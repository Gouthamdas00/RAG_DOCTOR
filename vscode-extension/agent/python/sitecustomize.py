"""Opt-in, process-local observer used by RAG Doctor's Run with Doctor mode.

Loaded only when the extension launches Python with RAG_DOCTOR_CAPTURE=1.
It observes selected requests and HTTPX LLM inference calls, including common
provider streaming formats, and forwards their content to the loopback analyzer.
It never writes a trace file.
"""

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
import weakref
from urllib.parse import urlparse


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        raise urllib.error.HTTPError(new_url, code, "Local RAG Doctor capture redirects are disabled.", headers, file_pointer)


def _direct_loopback_opener():
    # Ignore corporate proxy variables and refuse redirects so captured content
    # can only reach the local analyzer URL constructed below.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirectHandler())


_LOCAL_OPENER = _direct_loopback_opener()
MAX_CAPTURED_STREAM_BYTES = 1_000_000


def _provider(host: str, url: str):
    host = host.lower()
    path = url.lower()
    if "generativelanguage.googleapis.com" in host or ("aiplatform.googleapis.com" in host and "/publishers/google/" in path):
        return "google", "gemini"
    if "api.openai.com" in host or host.endswith(".openai.azure.com"):
        return "openai", "openai-compatible"
    if "api.anthropic.com" in host:
        return "anthropic", "anthropic"
    if host in {"127.0.0.1", "localhost", "::1"} and ("11434" in url or "/api/generate" in path or "/api/chat" in path):
        return "ollama", "ollama"
    return None, None


def _as_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (_as_text(item) for item in value)))
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), (str, list)):
            return _as_text(value["content"])
    return ""


def _request_prompt(body):
    if not isinstance(body, dict):
        return ""
    if isinstance(body.get("prompt"), str):
        return body["prompt"]
    blocks = []
    for message in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if isinstance(message, dict):
            role = str(message.get("role", "message"))
            content = _as_text(message.get("content", message.get("parts", "")))
            if content:
                blocks.append(f"{role.upper()}: {content}")
    for message in body.get("contents", []) if isinstance(body.get("contents"), list) else []:
        if isinstance(message, dict):
            role = str(message.get("role", "user"))
            parts = message.get("parts", [])
            text = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)) if isinstance(parts, list) else _as_text(parts)
            if text:
                blocks.append(f"{role.upper()}: {text}")
    system = body.get("system") or body.get("systemInstruction")
    system_text = _as_text(system)
    if system_text:
        blocks.insert(0, f"SYSTEM: {system_text}")
    return "\n\n".join(blocks)


def _response_answer(response, provider):
    try:
        status = getattr(response, "status_code", getattr(response, "status", 200))
        if status < 200 or status >= 300:
            return "", f"http_{status}"
        data = response.json()
    except Exception:
        return "", "unparseable_response"
    return _answer_from_data(data, provider)


def _answer_from_data(data, provider):
    try:
        if provider == "google":
            parts = data["candidates"][0]["content"]["parts"]
            return "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)), None
        if provider == "openai":
            choice = data["choices"][0]
            return _as_text(choice.get("message", {}).get("content", choice.get("text", ""))), None
        if provider == "anthropic":
            return _as_text(data.get("content", [])), None
        if provider == "ollama":
            return _as_text(data.get("response", data.get("message", {}).get("content", ""))), None
    except (KeyError, IndexError, TypeError):
        return "", "response_shape_not_recognized"
    return "", "provider_not_supported"


def _response_body_answer(raw, provider, status=200):
    if status < 200 or status >= 300:
        return "", f"http_{status}"
    try:
        data = json.loads(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw))
    except Exception:
        return "", "unparseable_response"
    return _answer_from_data(data, provider)


def _stream_delta(data, provider):
    """Extract one answer fragment from a provider SSE/JSONL event."""
    try:
        if provider == "google":
            candidates = data.get("candidates", [])
            if candidates:
                return _as_text(candidates[0].get("content", {}).get("parts", []))
        elif provider == "openai":
            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or choice.get("message") or {}
                return _as_text(delta.get("content", choice.get("text", "")))
        elif provider == "anthropic":
            if data.get("type") == "content_block_delta":
                return _as_text(data.get("delta", {}).get("text", ""))
            return _as_text(data.get("completion", ""))
        elif provider == "ollama":
            return _as_text(data.get("response", data.get("message", {}).get("content", "")))
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""
    return ""


def _stream_answer(raw, provider):
    """Reassemble common OpenAI, Gemini, Anthropic, and Ollama stream frames."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw or "")
    fragments = []
    if provider == "ollama":
        frames = text.splitlines()
        for frame in frames:
            try:
                data = json.loads(frame)
            except (json.JSONDecodeError, TypeError):
                continue
            fragment = _stream_delta(data, provider)
            if fragment:
                fragments.append(fragment)
    else:
        for frame in re.split(r"\r?\n\s*\r?\n", text):
            data_lines = [line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")]
            if not data_lines:
                continue
            data_text = "\n".join(data_lines).strip()
            if not data_text or data_text == "[DONE]":
                continue
            try:
                data = json.loads(data_text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict):
                fragment = _stream_delta(data, provider)
                if fragment:
                    fragments.append(fragment)
    answer = "".join(fragments)
    return (answer, None) if answer else ("", "stream_response_not_recognized")


def _supported_inference_url(url):
    parsed = urlparse(str(url))
    provider, model_family = _provider(parsed.hostname or "", str(url))
    path = parsed.path.lower()
    if not provider or not any(token in path for token in ("generatecontent", "chat/completions", "/messages", "/api/generate", "/api/chat")):
        return None, None
    return provider, model_family


def _request_model(body, url, fallback):
    if isinstance(body, dict) and isinstance(body.get("model"), str) and body["model"].strip():
        return body["model"].strip()[:120]
    if fallback == "gemini":
        match = re.search(r"/models/([^/:?]+)", str(url), re.I)
        if match:
            return match.group(1)[:120]
    return fallback


def _capture_payload(url, body, answer, error, latency_ms=None, streaming=False, capture_truncated=False):
    provider, model_family = _supported_inference_url(url)
    if not provider:
        return None
    full_prompt = _request_prompt(body)
    full_question = _question_from_prompt(full_prompt, body)
    truncated = capture_truncated or len(full_prompt) > 30000 or len(full_question) > 10000 or len(answer) > 20000
    prompt = full_prompt[:30000]
    question = full_question[:10000]
    answer = answer[:20000]
    chunks = _extract_finance_chunks(prompt)
    context_blocks, context_method = ([], "finance_source_labels") if chunks else _extract_prompt_context_blocks(prompt)
    metadata = {
        "provider": provider,
        "model": _request_model(body, url, model_family),
        "operation": "generate",
        "error": error,
        "streaming": bool(streaming),
        "capture_truncated": bool(truncated),
        "context_extraction_method": context_method,
        "prompt_context_block_count": len(context_blocks),
    }
    if isinstance(latency_ms, (int, float)) and latency_ms >= 0:
        metadata["latency_ms"] = round(float(latency_ms), 3)
    return {
        "question": question,
        "prompt": prompt,
        "retrieved_chunks": chunks,
        "prompt_context_blocks": context_blocks,
        "llm_answer": answer,
        "workspace_path": os.environ.get("RAG_DOCTOR_WORKSPACE", os.getcwd()),
        "retrieval_observed": True if chunks else None,
        "retrieval_count": len(chunks) if chunks else None,
        "metadata": metadata,
        "include_content": True,
    }


def _capture_inference(url, body, response, latency_ms=None):
    try:
        provider, _ = _supported_inference_url(url)
        if not provider:
            return
        answer, error = _response_answer(response, provider)
        payload = _capture_payload(url, body, answer, error, latency_ms)
        if payload:
            _send_capture(payload)
    except Exception:
        # Instrumentation must never turn a successful provider response into
        # an application failure.
        return


def _capture_inference_body(url, body, status, raw, latency_ms, overflow=False):
    try:
        provider, _ = _supported_inference_url(url)
        if not provider:
            return
        streaming = bool(isinstance(body, dict) and body.get("stream"))
        if status < 200 or status >= 300:
            answer, error = "", f"http_{status}"
        elif overflow:
            answer, error = "", "stream_capture_limit_exceeded"
        elif streaming:
            answer, error = _stream_answer(raw, provider)
        else:
            answer, error = _response_body_answer(raw, provider, status)
        payload = _capture_payload(
            url,
            body,
            answer,
            error,
            latency_ms,
            streaming=streaming,
            capture_truncated=overflow,
        )
        if payload:
            _send_capture(payload)
    except Exception:
        # Observation failures must never alter the application's HTTP result.
        return


def _request_body(request, data=None):
    raw = getattr(request, "data", None)
    if raw is None:
        raw = data
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeError:
            return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _observe_urllib_response(response, url, body, started_at):
    original_read = getattr(response, "read", None)
    if not callable(original_read):
        return
    state = {"captured": False, "overflow": False}
    captured = bytearray()

    def observed_read(*args, **kwargs):
        data = original_read(*args, **kwargs)
        _append_stream_bytes(captured, data, state)
        amount = kwargs.get("amt", args[0] if args else -1)
        complete = amount is None or amount < 0 or not data
        if complete and not state["captured"]:
            state["captured"] = True
            status = getattr(response, "status", getattr(response, "code", 200))
            _capture_inference_body(
                url,
                body,
                status,
                bytes(captured),
                (time.perf_counter() - started_at) * 1000,
                state["overflow"],
            )
        return data

    try:
        response.read = observed_read
    except Exception:
        return


def _capture_stream_inference(url, body, response, raw, latency_ms, overflow=False):
    try:
        provider, _ = _supported_inference_url(url)
        if not provider:
            return
        status_code = getattr(response, "status_code", 200)
        if status_code < 200 or status_code >= 300:
            answer, error = "", f"http_{status_code}"
        elif overflow:
            answer, error = "", "stream_capture_limit_exceeded"
        else:
            answer, error = _stream_answer(raw, provider)
        payload = _capture_payload(
            url,
            body,
            answer,
            error,
            latency_ms,
            streaming=True,
            capture_truncated=overflow,
        )
        if payload:
            _send_capture(payload)
    except Exception:
        return


def _append_stream_bytes(buffer, chunk, state):
    if state["overflow"]:
        return
    if isinstance(chunk, str):
        chunk = chunk.encode("utf-8", errors="replace")
    if not isinstance(chunk, bytes):
        return
    if len(buffer) + len(chunk) > MAX_CAPTURED_STREAM_BYTES:
        buffer.clear()
        state["overflow"] = True
    else:
        buffer.extend(chunk)


def _observe_requests_stream(response, url, body, started_at):
    original_iter_content = getattr(response, "iter_content", None)
    if not callable(original_iter_content):
        return
    state = {"captured": False, "overflow": False}

    def observed_iter_content(*args, **kwargs):
        captured = bytearray()
        complete = False
        try:
            for chunk in original_iter_content(*args, **kwargs):
                _append_stream_bytes(captured, chunk, state)
                yield chunk
            complete = True
        finally:
            if complete and not state["captured"]:
                state["captured"] = True
                _capture_stream_inference(
                    url, body, response, bytes(captured), (time.perf_counter() - started_at) * 1000, state["overflow"]
                )

    try:
        response.iter_content = observed_iter_content
    except Exception:
        return


def _observe_httpx_sync_stream(response, url, body, started_at):
    original_iter_bytes = getattr(response, "iter_bytes", None)
    if not callable(original_iter_bytes):
        return
    state = {"captured": False, "overflow": False}

    def observed_iter_bytes(*args, **kwargs):
        captured = bytearray()
        complete = False
        try:
            for chunk in original_iter_bytes(*args, **kwargs):
                _append_stream_bytes(captured, chunk, state)
                yield chunk
            complete = True
        finally:
            if complete and not state["captured"]:
                state["captured"] = True
                _capture_stream_inference(
                    url, body, response, bytes(captured), (time.perf_counter() - started_at) * 1000, state["overflow"]
                )

    try:
        response.iter_bytes = observed_iter_bytes
    except Exception:
        return


def _observe_httpx_async_stream(response, url, body, started_at):
    original_aiter_bytes = getattr(response, "aiter_bytes", None)
    if not callable(original_aiter_bytes):
        return
    state = {"captured": False, "overflow": False}

    async def observed_aiter_bytes(*args, **kwargs):
        captured = bytearray()
        complete = False
        try:
            async for chunk in original_aiter_bytes(*args, **kwargs):
                _append_stream_bytes(captured, chunk, state)
                yield chunk
            complete = True
        finally:
            if complete and not state["captured"]:
                state["captured"] = True
                latency_ms = (time.perf_counter() - started_at) * 1000
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        _capture_stream_inference,
                        url,
                        body,
                        response,
                        bytes(captured),
                        latency_ms,
                        state["overflow"],
                    )
                except Exception:
                    pass

    try:
        response.aiter_bytes = observed_aiter_bytes
    except Exception:
        return


def _httpx_body(request):
    try:
        raw = request.content
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        # Streaming request bodies are not consumed just for observation.
        return {}


def _extract_finance_chunks(prompt):
    """Parse the explicit source labels used by the finance app's prompt template."""
    pattern = re.compile(r"\[Source #(?P<rank>\d+)\s*\|\s*Chunk ID:\s*(?P<chunk>[^|\]]+)\|\s*Filing:\s*(?P<doc>[^\]]+)\]\s*\n", re.I)
    matches = list(pattern.finditer(prompt))
    chunks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        text = prompt[match.end():end]
        text = re.split(r"\n\s*--- USER QUESTION ---", text, maxsplit=1, flags=re.I)[0]
        chunks.append({
            "text": text.strip(),
            "chunk_id": match.group("chunk").strip(),
            "doc_id": match.group("doc").strip(),
            "final_rank": int(match.group("rank")),
        })
    return chunks


_PROMPT_SOURCE_MARKER = re.compile(
    r"(?im)^\s*(?:(?:SYSTEM|USER|ASSISTANT|HUMAN):\s*)?(?:\[(?P<bracket_kind>source|document|passage)\s*#?\s*(?P<bracket_number>\d+)[^\]]*\]|"
    r"(?P<plain_kind>source|document|passage)\s*#?\s*(?P<plain_number>\d+)\s*[:\-])\s*(?P<title>[^\n]*)$"
)
_PROMPT_CONTEXT_HEADING = re.compile(
    r"(?im)^\s*(?:(?:SYSTEM|USER|ASSISTANT|HUMAN):\s*)?(?:#{1,6}\s*)?(?:use the following pieces of context|context information is below|"
    r"(?:retrieved\s+)?context(?:\s+(?:information|documents|passages|chunks))?|"
    r"(?:relevant\s+)?(?:documents|passages|reference material)|"
    r"here are (?:the )?(?:relevant )?documents)\b[^\n]*$"
)
_PROMPT_INLINE_CONTEXT = re.compile(
    r"(?i)^\s*(?:(?:system|user|assistant|human):\s*)?(?:#{1,6}\s*)?"
    r"(?:retrieved\s+context|context(?:\s+(?:information|documents|passages|chunks))?|"
    r"(?:relevant\s+)?(?:documents|passages|reference material))\s*:\s*(?P<body>.*)$"
)
_PROMPT_CONTEXT_END = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:---\s*)?(?:user question|question|query|user input|answer|"
    r"assistant response|response|task|instruction|system prompt|user|assistant|human|system)\s*[:\-].*$"
)
_PROMPT_SEPARATOR = re.compile(r"^\s*(?:-{3,}|_{3,}|={3,})\s*$")


def _trim_prompt_context(value, stop_at_separator=False):
    lines = str(value or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and _PROMPT_SEPARATOR.match(lines[0]):
        lines.pop(0)
        stop_at_separator = True
    collected = []
    for line in lines:
        if _PROMPT_CONTEXT_END.match(line):
            break
        if stop_at_separator and _PROMPT_SEPARATOR.match(line):
            break
        collected.append(line)
    return "\n".join(collected).strip()


def _extract_prompt_context_blocks(prompt):
    """Find likely assembled context spans without inventing retrieval ranks or IDs."""
    prompt = str(prompt or "")
    if not prompt:
        return [], "none"

    source_matches = list(_PROMPT_SOURCE_MARKER.finditer(prompt))
    if source_matches:
        blocks = []
        for index, match in enumerate(source_matches[:10]):
            end = source_matches[index + 1].start() if index + 1 < len(source_matches) else len(prompt)
            text = _trim_prompt_context(prompt[match.end():end])
            if len(text) < 20:
                continue
            kind = match.group("bracket_kind") or match.group("plain_kind")
            number = match.group("bracket_number") or match.group("plain_number")
            title = match.group("title").strip()
            label = f"{kind.title()} {number}"
            if title:
                label += f": {title[:160]}"
            blocks.append({
                "text": text[:10000],
                "label": label,
                "prompt_position": index + 1,
                "extraction_method": "prompt_source_marker",
                "confidence": "high",
            })
        if blocks:
            return blocks, "prompt_source_marker"

    tag_pattern = re.compile(
        r"(?is)<(?P<tag>retrieved_context|context|retrieved_documents|documents|reference_documents)\b[^>]*>"
        r"(?P<body>.*?)</(?P=tag)\s*>"
    )
    tag_matches = list(tag_pattern.finditer(prompt))
    if tag_matches:
        blocks = []
        for index, match in enumerate(tag_matches[:10]):
            text = _trim_prompt_context(match.group("body"))
            if len(text) < 20:
                continue
            blocks.append({
                "text": text[:10000],
                "label": f"<{match.group('tag')}> section",
                "prompt_position": index + 1,
                "extraction_method": "prompt_context_tag",
                "confidence": "high",
            })
        if blocks:
            return blocks, "prompt_context_tag"

    heading_matches = list(_PROMPT_CONTEXT_HEADING.finditer(prompt))
    if heading_matches:
        blocks = []
        for index, match in enumerate(heading_matches[:10]):
            end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(prompt)
            line_end = prompt.find("\n", match.start())
            if line_end < 0 or line_end > match.end():
                line_end = match.end()
            heading_line = prompt[match.start():line_end]
            inline_match = _PROMPT_INLINE_CONTEXT.match(heading_line)
            inline_text = inline_match.group("body") if inline_match else ""
            text = _trim_prompt_context("\n".join(part for part in (inline_text, prompt[match.end():end]) if part), stop_at_separator=True)
            if len(text) < 20:
                continue
            label = " ".join(match.group(0).strip().split())[:160]
            blocks.append({
                "text": text[:10000],
                "label": label or "Context section",
                "prompt_position": index + 1,
                "extraction_method": "prompt_context_heading",
                "confidence": "moderate",
            })
        if blocks:
            return blocks, "prompt_context_heading"

    return [], "none"


def _question_from_prompt(prompt, body):
    match = re.search(r"--- USER QUESTION ---\s*(.*?)\s*(?:--- ANALYST RESPONSE|$)", prompt, re.I | re.S)
    if match:
        return match.group(1).strip()[:10000]
    if isinstance(body, dict):
        for key in ("question", "query", "input"):
            if isinstance(body.get(key), str):
                return body[key][:10000]
        for key in ("messages", "contents"):
            items = body.get(key)
            if not isinstance(items, list):
                continue
            for message in reversed(items):
                if not isinstance(message, dict) or str(message.get("role", "user")).lower() not in {"user", "human"}:
                    continue
                text = _as_text(message.get("content", message.get("parts", "")))
                if text:
                    return text[:10000]
    return ""


def _send_capture(payload):
    port = str(os.environ.get("RAG_DOCTOR_PORT", "8765"))
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/capture-live-run",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _LOCAL_OPENER.open(request, timeout=2):
            pass
    except Exception:
        # Diagnostics must never prevent the application's model request completing.
        pass


def _send_ready(observer_installed, detail=""):
    """Send a content-free local beacon so Run with Doctor knows injection worked."""
    port = str(os.environ.get("RAG_DOCTOR_PORT", "8765"))
    payload = {
        "workspace_path": os.environ.get("RAG_DOCTOR_WORKSPACE", os.getcwd()),
        "process_id": os.getpid(),
        "providers": ["google", "openai", "anthropic", "ollama"],
        "observer_installed": observer_installed,
        "detail": detail,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/runtime-agent-ready",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _LOCAL_OPENER.open(request, timeout=2):
            pass
    except Exception:
        pass


def _install_observer():
    if os.environ.get("RAG_DOCTOR_CAPTURE") != "1":
        return
    requests_sessions = None
    httpx_module = None
    aiohttp_module = None
    urllib_director = None
    urllib_installed = False
    try:
        import requests.sessions
        requests_sessions = requests.sessions
        original_request = requests_sessions.Session.request
    except Exception:
        original_request = None
    try:
        import httpx
        httpx_module = httpx
        original_httpx_send = httpx.Client.send
        original_httpx_async_send = httpx.AsyncClient.send
    except Exception:
        original_httpx_send = None
        original_httpx_async_send = None
    try:
        import aiohttp
        aiohttp_module = aiohttp
        original_aiohttp_request = aiohttp.ClientSession._request
        original_aiohttp_read = aiohttp.ClientResponse.read
        aiohttp_pending = weakref.WeakKeyDictionary()
    except Exception:
        original_aiohttp_request = None
        original_aiohttp_read = None
        aiohttp_pending = None

    try:
        urllib_director = urllib.request.OpenerDirector
        current_open = urllib_director.open
        if not getattr(current_open, "_rag_doctor_observer", False):
            original_urllib_open = current_open

            def observed_urllib_open(opener, fullurl, data=None, *args, **kwargs):
                url = fullurl.full_url if isinstance(fullurl, urllib.request.Request) else str(fullurl)
                if not _supported_inference_url(url)[0]:
                    return original_urllib_open(opener, fullurl, data, *args, **kwargs)
                body = _request_body(fullurl, data)
                started_at = time.perf_counter()
                response = original_urllib_open(opener, fullurl, data, *args, **kwargs)
                _observe_urllib_response(response, url, body, started_at)
                return response

            observed_urllib_open._rag_doctor_observer = True
            urllib_director.open = observed_urllib_open
        urllib_installed = True
    except Exception:
        urllib_director = None

    if requests_sessions is None and httpx_module is None and aiohttp_module is None and not urllib_installed:
        _send_ready(False, "No supported Python HTTP client could be instrumented (requests, HTTPX, aiohttp, or urllib).")
        return

    if requests_sessions is not None:
        def observed_request(session, method, url, **kwargs):
            if not _supported_inference_url(url)[0]:
                return original_request(session, method, url, **kwargs)
            body = kwargs.get("json")
            if body is None:
                raw = kwargs.get("data")
                try:
                    body = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw) if raw else {}
                except Exception:
                    body = {}
            started_at = time.perf_counter()
            response = original_request(session, method, url, **kwargs)
            latency_ms = (time.perf_counter() - started_at) * 1000
            if kwargs.get("stream", False):
                _observe_requests_stream(response, url, body, started_at)
            else:
                _capture_inference(url, body, response, latency_ms)
            return response

        requests_sessions.Session.request = observed_request

    if httpx_module is not None:
        def observed_httpx_send(client, request, **kwargs):
            url = str(request.url)
            if not _supported_inference_url(url)[0]:
                return original_httpx_send(client, request, **kwargs)
            body = _httpx_body(request)
            started_at = time.perf_counter()
            response = original_httpx_send(client, request, **kwargs)
            latency_ms = (time.perf_counter() - started_at) * 1000
            if kwargs.get("stream", False):
                _observe_httpx_sync_stream(response, url, body, started_at)
            else:
                _capture_inference(url, body, response, latency_ms)
            return response

        async def observed_httpx_async_send(client, request, **kwargs):
            url = str(request.url)
            if not _supported_inference_url(url)[0]:
                return await original_httpx_async_send(client, request, **kwargs)
            body = _httpx_body(request)
            started_at = time.perf_counter()
            response = await original_httpx_async_send(client, request, **kwargs)
            latency_ms = (time.perf_counter() - started_at) * 1000
            if kwargs.get("stream", False):
                _observe_httpx_async_stream(response, url, body, started_at)
            else:
                try:
                    loop = asyncio.get_running_loop()
                    # Preserve ordered capture before returning the model result,
                    # but do not block the application's async event loop on
                    # the local analyzer's HTTP request.
                    await loop.run_in_executor(None, _capture_inference, url, body, response, latency_ms)
                except Exception:
                    pass
            return response

        httpx_module.Client.send = observed_httpx_send
        httpx_module.AsyncClient.send = observed_httpx_async_send

    if aiohttp_module is not None:
        async def observed_aiohttp_request(session, method, str_or_url, *args, **kwargs):
            url = str(str_or_url)
            if not _supported_inference_url(url)[0]:
                return await original_aiohttp_request(session, method, str_or_url, *args, **kwargs)
            body = kwargs.get("json")
            if body is None:
                body = _request_body(None, kwargs.get("data"))
            started_at = time.perf_counter()
            response = await original_aiohttp_request(session, method, str_or_url, *args, **kwargs)
            try:
                aiohttp_pending[response] = (url, body, started_at)
            except (TypeError, AttributeError):
                pass
            return response

        async def observed_aiohttp_read(response):
            raw = await original_aiohttp_read(response)
            observation = aiohttp_pending.pop(response, None)
            if observation:
                url, body, started_at = observation
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        _capture_inference_body,
                        url,
                        body,
                        getattr(response, "status", 200),
                        raw,
                        (time.perf_counter() - started_at) * 1000,
                    )
                except Exception:
                    pass
            return raw

        aiohttp_module.ClientSession._request = observed_aiohttp_request
        aiohttp_module.ClientResponse.read = observed_aiohttp_read

    installed_clients = [
        name for name, installed in (
            ("requests", requests_sessions is not None),
            ("HTTPX", httpx_module is not None),
            ("aiohttp", aiohttp_module is not None),
            ("urllib", urllib_installed),
        ) if installed
    ]
    _send_ready(True, "Observed Python HTTP clients: " + ", ".join(installed_clients) + ".")


_install_observer()
