"""Offline checks for the opt-in Python observer used by Run with Doctor."""

import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[2]
OBSERVER_PATH = ROOT / "vscode-extension" / "agent" / "python" / "sitecustomize.py"
SPEC = importlib.util.spec_from_file_location("rag_doctor_runtime_observer", OBSERVER_PATH)
OBSERVER = importlib.util.module_from_spec(SPEC)
with patch.dict(os.environ, {"RAG_DOCTOR_CAPTURE": "0"}):
    SPEC.loader.exec_module(OBSERVER)


class FakeGoogleResponse:
    status_code = 200

    @staticmethod
    def json():
        return {
            "candidates": [{
                "content": {
                    "parts": [{"text": "The answer is supported by filing evidence."}],
                },
            }],
        }


class RuntimeObserverTests(unittest.TestCase):
    def test_capture_transport_ignores_proxy_environment_and_blocks_redirects(self):
        with (
            patch.dict(os.environ, {
                "HTTP_PROXY": "http://proxy.invalid:8080",
                "HTTPS_PROXY": "http://proxy.invalid:8080",
                "http_proxy": "http://proxy.invalid:8080",
                "https_proxy": "http://proxy.invalid:8080",
            }),
            patch("urllib.request.getproxies", side_effect=AssertionError("proxy environment was read")) as getproxies,
        ):
            opener = OBSERVER._direct_loopback_opener()

        getproxies.assert_not_called()
        redirect = next(handler for handler in opener.handlers if isinstance(handler, OBSERVER._NoRedirectHandler))
        with self.assertRaises(HTTPError):
            redirect.redirect_request(
                Request("http://127.0.0.1:8765/api/capture-live-run"),
                None,
                302,
                "Found",
                {},
                "https://example.com/collect",
            )

    def test_finance_prompt_parser_extracts_question_and_labeled_chunks(self):
        prompt = (
            "Use only filing excerpts.\n"
            "[Source #1 | Chunk ID: 3M_2018_10K_c4 | Filing: 3M_2018_10K]\n"
            "Net sales were $32,765 million.\n\n"
            "[Source #2 | Chunk ID: 3M_2018_10K_c8 | Filing: 3M_2018_10K]\n"
            "Capital expenditures were $1,200 million.\n\n"
            "--- USER QUESTION ---\nWhat were FY2018 capital expenditures?\n\n"
            "--- ANALYST RESPONSE (in clean Markdown) ---"
        )
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

        captured_prompt = OBSERVER._request_prompt(body)
        chunks = OBSERVER._extract_finance_chunks(captured_prompt)

        self.assertEqual(OBSERVER._question_from_prompt(captured_prompt, body), "What were FY2018 capital expenditures?")
        self.assertEqual([chunk["chunk_id"] for chunk in chunks], ["3M_2018_10K_c4", "3M_2018_10K_c8"])
        self.assertIn("$1,200 million", chunks[1]["text"])
        self.assertNotIn("What were FY2018", chunks[1]["text"])

    def test_generic_prompt_parser_extracts_context_without_inventing_retrieval_ranks(self):
        prompt = (
            "USER: Use the following documents to answer the question.\n"
            "Document 1: Annual report section 4\n"
            "Revenue for the year was 82 million dollars.\n"
            "Document 2: Annual report section 7\n"
            "Operating costs were 51 million dollars.\n"
            "Question: What was the reported revenue?\n"
            "Answer:"
        )
        blocks, method = OBSERVER._extract_prompt_context_blocks(prompt)

        self.assertEqual(method, "prompt_source_marker")
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["label"], "Document 1: Annual report section 4")
        self.assertIn("82 million dollars", blocks[0]["text"])
        self.assertNotIn("What was the reported revenue", "\n".join(block["text"] for block in blocks))
        self.assertTrue(all("rank" not in block and "doc_id" not in block for block in blocks))

    def test_context_headings_tags_and_llamaindex_delimiters_are_recognized(self):
        prompts = [
            (
                "Context:\nThe official filing says net income was 4 million dollars.\n\n"
                "Question: What was net income?",
                "prompt_context_heading",
                "4 million dollars",
            ),
            (
                "Context information is below.\n---------------------\n"
                "The official filing says net income was 4 million dollars.\n"
                "---------------------\nGiven the context above, answer carefully.\nQuery: What was net income?",
                "prompt_context_heading",
                "4 million dollars",
            ),
            (
                "<retrieved_context>The official filing says net income was 4 million dollars.</retrieved_context>\n"
                "Question: What was net income?",
                "prompt_context_tag",
                "4 million dollars",
            ),
        ]
        for prompt, expected_method, expected_phrase in prompts:
            with self.subTest(method=expected_method, prompt=prompt[:30]):
                blocks, method = OBSERVER._extract_prompt_context_blocks(prompt)
                self.assertEqual(method, expected_method)
                self.assertEqual(len(blocks), 1)
                self.assertIn(expected_phrase, blocks[0]["text"])
                self.assertNotIn("What was net income?", blocks[0]["text"])

    def test_capture_payload_keeps_generic_context_separate_from_retriever_results(self):
        prompt = (
            "Context:\nThe official filing says net income was 4 million dollars.\n\n"
            "Question: What was net income?"
        )
        payload = OBSERVER._capture_payload(
            "https://api.openai.com/v1/chat/completions",
            {"model": "generic-rag", "prompt": prompt},
            "Net income was 3 million dollars.",
            None,
        )

        self.assertEqual(payload["retrieved_chunks"], [])
        self.assertIsNone(payload["retrieval_observed"])
        self.assertIsNone(payload["retrieval_count"])
        self.assertEqual(len(payload["prompt_context_blocks"]), 1)
        self.assertEqual(payload["metadata"]["context_extraction_method"], "prompt_context_heading")

    def test_stream_parser_reassembles_supported_provider_formats(self):
        streams = {
            "openai": (
                'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"world."}}]}\n\n'
                "data: [DONE]\n\n"
            ),
            "google": (
                'data: {"candidates":[{"content":{"parts":[{"text":"Gemini "}]}}]}\n\n'
                'data: {"candidates":[{"content":{"parts":[{"text":"answer."}]}}]}\n\n'
            ),
            "anthropic": (
                'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Claude "}}\n\n'
                'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"answer."}}\n\n'
            ),
            "ollama": (
                '{"message":{"content":"Ollama "},"done":false}\n'
                '{"message":{"content":"answer."},"done":true}\n'
            ),
        }
        expected = {
            "openai": "Hello world.",
            "google": "Gemini answer.",
            "anthropic": "Claude answer.",
            "ollama": "Ollama answer.",
        }

        for provider, stream in streams.items():
            with self.subTest(provider=provider):
                answer, error = OBSERVER._stream_answer(stream.encode("utf-8"), provider)
                self.assertEqual(answer, expected[provider])
                self.assertIsNone(error)

    def test_requests_gemini_call_forwards_local_run_capture(self):
        prompt = (
            "[Source #1 | Chunk ID: 3M_2018_10K_c4 | Filing: 3M_2018_10K]\n"
            "Capital expenditures were $1,200 million.\n\n"
            "--- USER QUESTION ---\nWhat were FY2018 capital expenditures?\n\n"
            "--- ANALYST RESPONSE (in clean Markdown) ---"
        )
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        captures = []
        ready = []
        calls = []

        class FakeSession:
            def request(self, method, url, **kwargs):
                calls.append((method, url, kwargs))
                return FakeGoogleResponse()

            def post(self, url, **kwargs):
                return self.request("POST", url, **kwargs)

        fake_sessions = types.ModuleType("requests.sessions")
        fake_sessions.Session = FakeSession
        fake_requests = types.ModuleType("requests")
        fake_requests.__path__ = []
        fake_requests.sessions = fake_sessions
        fake_requests.post = lambda url, **kwargs: FakeSession().post(url, **kwargs)

        with (
            patch.dict(sys.modules, {"requests": fake_requests, "requests.sessions": fake_sessions}),
            patch.dict(os.environ, {
                "RAG_DOCTOR_CAPTURE": "1",
                "RAG_DOCTOR_PORT": "8765",
                "RAG_DOCTOR_WORKSPACE": r"C:\workspace\finance-app",
            }),
            patch.object(OBSERVER, "_send_capture", side_effect=captures.append),
            patch.object(OBSERVER, "_send_ready", side_effect=lambda installed, detail="": ready.append((installed, detail))),
        ):
            OBSERVER._install_observer()
            response = fake_requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent",
                json=body,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(ready), 1)
        self.assertTrue(ready[0][0])
        self.assertIn("requests", ready[0][1])
        self.assertIn("urllib", ready[0][1])
        self.assertEqual(len(captures), 1)
        captured = captures[0]
        self.assertEqual(captured["workspace_path"], r"C:\workspace\finance-app")
        self.assertEqual(captured["question"], "What were FY2018 capital expenditures?")
        self.assertEqual(captured["llm_answer"], "The answer is supported by filing evidence.")
        self.assertEqual(captured["retrieved_chunks"][0]["doc_id"], "3M_2018_10K")
        self.assertTrue(captured["include_content"])
        self.assertEqual(captured["metadata"]["provider"], "google")

    def test_urllib_gemini_call_captures_response_without_consuming_extra_bytes(self):
        captures = []
        ready = []
        prompt = {"contents": [{"role": "user", "parts": [{"text": "What does the report say?"}]}]}
        raw_request = json.dumps(prompt).encode("utf-8")
        raw_response = json.dumps({"candidates": [{"content": {"parts": [{"text": "The report says 42."}]}}]}).encode("utf-8")
        calls = []

        class FakeResponse:
            status = 200

            def __init__(self):
                self.remaining = raw_response

            def read(self, amount=-1):
                if amount is None or amount < 0:
                    result, self.remaining = self.remaining, b""
                    return result
                result, self.remaining = self.remaining[:amount], self.remaining[amount:]
                return result

        def fake_open(opener, fullurl, data=None, *args, **kwargs):
            calls.append((fullurl.full_url, fullurl.data, data))
            return FakeResponse()

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
        request = Request(url, data=raw_request, headers={"Content-Type": "application/json"})
        with (
            patch.dict(sys.modules, {"requests": None, "requests.sessions": None, "httpx": None, "aiohttp": None}),
            patch.dict(os.environ, {"RAG_DOCTOR_CAPTURE": "1", "RAG_DOCTOR_WORKSPACE": "/workspace/urllib-rag"}),
            patch.object(OBSERVER.urllib.request.OpenerDirector, "open", fake_open),
            patch.object(OBSERVER, "_send_capture", side_effect=captures.append),
            patch.object(OBSERVER, "_send_ready", side_effect=lambda installed, detail="": ready.append((installed, detail))),
        ):
            OBSERVER._install_observer()
            response = OBSERVER.urllib.request.OpenerDirector().open(request)
            received = response.read()

        self.assertEqual(received, raw_response)
        self.assertEqual(len(calls), 1)
        self.assertTrue(ready[0][0])
        self.assertIn("urllib", ready[0][1])
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]["workspace_path"], "/workspace/urllib-rag")
        self.assertEqual(captures[0]["question"], "What does the report say?")
        self.assertEqual(captures[0]["llm_answer"], "The report says 42.")

    def test_aiohttp_openai_call_captures_answer_when_response_is_read(self):
        captures = []
        ready = []
        request_body = {
            "model": "openai-aiohttp-test",
            "messages": [{"role": "user", "content": "What does the report say?"}],
        }
        raw_response = json.dumps({"choices": [{"message": {"content": "The report says 42."}}]}).encode("utf-8")

        class FakeResponse:
            status = 200

            async def read(self):
                return raw_response

        class FakeSession:
            async def _request(self, method, url, **kwargs):
                return FakeResponse()

        class ImmediateAwaitable:
            def __await__(self):
                if False:
                    yield None
                return None

        class FakeLoop:
            def run_in_executor(self, executor, function, *args):
                function(*args)
                return ImmediateAwaitable()

        fake_aiohttp = types.ModuleType("aiohttp")
        fake_aiohttp.ClientSession = FakeSession
        fake_aiohttp.ClientResponse = FakeResponse
        url = "https://api.openai.com/v1/chat/completions"

        async def send_and_read():
            response = await fake_aiohttp.ClientSession()._request("POST", url, json=request_body)
            return await response.read()

        with (
            patch.dict(sys.modules, {"requests": None, "requests.sessions": None, "httpx": None, "aiohttp": fake_aiohttp}),
            patch.dict(os.environ, {"RAG_DOCTOR_CAPTURE": "1", "RAG_DOCTOR_WORKSPACE": "/workspace/aiohttp-rag"}),
            patch.object(OBSERVER, "_send_capture", side_effect=captures.append),
            patch.object(OBSERVER, "_send_ready", side_effect=lambda installed, detail="": ready.append((installed, detail))),
            patch.object(OBSERVER.asyncio, "get_running_loop", return_value=FakeLoop()),
        ):
            OBSERVER._install_observer()
            pending = send_and_read()
            try:
                pending.send(None)
            except StopIteration as completed:
                received = completed.value
            else:
                self.fail("the fake aiohttp request should complete without a real event loop")

        self.assertEqual(received, raw_response)
        self.assertTrue(ready[0][0])
        self.assertIn("aiohttp", ready[0][1])
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]["workspace_path"], "/workspace/aiohttp-rag")
        self.assertEqual(captures[0]["metadata"]["provider"], "openai")
        self.assertEqual(captures[0]["metadata"]["model"], "openai-aiohttp-test")
        self.assertEqual(captures[0]["question"], "What does the report say?")
        self.assertEqual(captures[0]["llm_answer"], "The report says 42.")

    def test_httpx_sync_and_async_calls_capture_openai_runs(self):
        captures = []
        ready = []
        prompt_body = {
            "model": "gpt-test-local-client",
            "messages": [
                {"role": "system", "content": "Use retrieved evidence."},
                {"role": "user", "content": "What does the report say?"},
            ],
        }
        request_content = json.dumps(prompt_body).encode("utf-8")

        class FakeRequest:
            url = "https://api.openai.com/v1/chat/completions"
            content = request_content

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "The report says 42."}}]}

        class FakeClient:
            def send(self, request, **kwargs):
                return FakeResponse()

        class FakeAsyncClient:
            async def send(self, request, **kwargs):
                return FakeResponse()

        class ImmediateAwaitable:
            def __await__(self):
                if False:
                    yield None
                return None

        class FakeLoop:
            def run_in_executor(self, executor, function, *args):
                function(*args)
                return ImmediateAwaitable()

        fake_httpx = types.ModuleType("httpx")
        fake_httpx.Client = FakeClient
        fake_httpx.AsyncClient = FakeAsyncClient

        with (
            patch.dict(sys.modules, {"requests": None, "requests.sessions": None, "httpx": fake_httpx}),
            patch.dict(os.environ, {
                "RAG_DOCTOR_CAPTURE": "1",
                "RAG_DOCTOR_PORT": "8765",
                "RAG_DOCTOR_WORKSPACE": r"C:\workspace\rag-app",
            }),
            patch.object(OBSERVER, "_send_capture", side_effect=captures.append),
            patch.object(OBSERVER, "_send_ready", side_effect=lambda installed, detail="": ready.append((installed, detail))),
            patch.object(OBSERVER.asyncio, "get_running_loop", return_value=FakeLoop()),
        ):
            OBSERVER._install_observer()
            sync_response = fake_httpx.Client().send(FakeRequest())
            async_call = fake_httpx.AsyncClient().send(FakeRequest())
            try:
                async_call.send(None)
            except StopIteration as completed:
                async_response = completed.value
            else:
                self.fail("the fake async HTTP client should complete without scheduling a real event loop")

        self.assertEqual(sync_response.status_code, 200)
        self.assertEqual(async_response.status_code, 200)
        self.assertEqual(len(ready), 1)
        self.assertTrue(ready[0][0])
        self.assertIn("HTTPX", ready[0][1])
        self.assertIn("urllib", ready[0][1])
        self.assertEqual(len(captures), 2)
        for captured in captures:
            self.assertEqual(captured["workspace_path"], r"C:\workspace\rag-app")
            self.assertEqual(captured["question"], "What does the report say?")
            self.assertEqual(captured["llm_answer"], "The report says 42.")
            self.assertEqual(captured["metadata"]["provider"], "openai")
            self.assertEqual(captured["metadata"]["model"], "gpt-test-local-client")
            self.assertIsInstance(captured["metadata"]["latency_ms"], float)
            self.assertGreaterEqual(captured["metadata"]["latency_ms"], 0)
            self.assertTrue(captured["include_content"])

    def test_httpx_streaming_calls_preserve_bytes_and_capture_completed_answers(self):
        captures = []
        ready = []
        prompt_body = {
            "model": "gpt-stream-test",
            "messages": [{"role": "user", "content": "What is the streamed answer?"}],
        }
        request_content = json.dumps(prompt_body).encode("utf-8")
        response_body = (
            b'data: {"choices":[{"delta":{"content":"First "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"second."}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        class FakeRequest:
            url = "https://api.openai.com/v1/chat/completions"
            content = request_content

        class FakeResponse:
            status_code = 200

            def iter_bytes(self, chunk_size=None):
                for index in range(0, len(response_body), 11):
                    yield response_body[index:index + 11]

            async def aiter_bytes(self, chunk_size=None):
                for index in range(0, len(response_body), 11):
                    yield response_body[index:index + 11]

        class FakeClient:
            def send(self, request, **kwargs):
                return FakeResponse()

        class FakeAsyncClient:
            async def send(self, request, **kwargs):
                return FakeResponse()

        class ImmediateAwaitable:
            def __await__(self):
                if False:
                    yield None
                return None

        class FakeLoop:
            def run_in_executor(self, executor, function, *args):
                function(*args)
                return ImmediateAwaitable()

        fake_httpx = types.ModuleType("httpx")
        fake_httpx.Client = FakeClient
        fake_httpx.AsyncClient = FakeAsyncClient
        with (
            patch.dict(sys.modules, {"requests": None, "requests.sessions": None, "httpx": fake_httpx}),
            patch.dict(os.environ, {
                "RAG_DOCTOR_CAPTURE": "1",
                "RAG_DOCTOR_PORT": "8765",
                "RAG_DOCTOR_WORKSPACE": r"C:\workspace\rag-app",
            }),
            patch.object(OBSERVER, "_send_capture", side_effect=captures.append),
            patch.object(OBSERVER, "_send_ready", side_effect=lambda installed, detail="": ready.append((installed, detail))),
            patch.object(OBSERVER.asyncio, "get_running_loop", return_value=FakeLoop()),
        ):
            OBSERVER._install_observer()
            sync_response = fake_httpx.Client().send(FakeRequest(), stream=True)
            sync_bytes = b"".join(sync_response.iter_bytes())
            async_call = fake_httpx.AsyncClient().send(FakeRequest(), stream=True)
            try:
                async_response = async_call.send(None).value
            except StopIteration as completed:
                async_response = completed.value

            async def read_async_stream():
                return b"".join([chunk async for chunk in async_response.aiter_bytes()])

            async_stream = read_async_stream()
            try:
                async_stream.send(None)
            except StopIteration as completed:
                async_bytes = completed.value
            else:
                self.fail("the fake asynchronous response should complete without a real event loop")

        self.assertEqual(sync_bytes, response_body)
        self.assertEqual(async_bytes, response_body)
        self.assertEqual(len(ready), 1)
        self.assertTrue(ready[0][0])
        self.assertIn("HTTPX", ready[0][1])
        self.assertIn("urllib", ready[0][1])
        self.assertEqual(len(captures), 2)
        for captured in captures:
            self.assertEqual(captured["llm_answer"], "First second.")
            self.assertTrue(captured["metadata"]["streaming"])
            self.assertIsInstance(captured["metadata"]["latency_ms"], float)

    def test_real_httpx_stream_objects_are_wrapped_without_changing_payloads(self):
        try:
            import httpx
        except ImportError:
            self.skipTest("HTTPX is optional in the analyzer environment.")

        captures = []
        body = {"model": "real-httpx", "messages": [{"role": "user", "content": "question"}]}
        raw = b'data: {"choices":[{"delta":{"content":"Actual HTTPX stream."}}]}\n\ndata: [DONE]\n\n'

        class AsyncBytes(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield raw

        class ImmediateAwaitable:
            def __await__(self):
                if False:
                    yield None
                return None

        class FakeLoop:
            def run_in_executor(self, executor, function, *args):
                function(*args)
                return ImmediateAwaitable()

        with patch.object(OBSERVER, "_send_capture", side_effect=captures.append):
            sync_response = httpx.Response(200, stream=httpx.ByteStream(raw))
            OBSERVER._observe_httpx_sync_stream(
                sync_response,
                "https://api.openai.com/v1/chat/completions",
                body,
                OBSERVER.time.perf_counter(),
            )
            sync_bytes = b"".join(sync_response.iter_bytes())

            async_response = httpx.Response(200, stream=AsyncBytes())
            OBSERVER._observe_httpx_async_stream(
                async_response,
                "https://api.openai.com/v1/chat/completions",
                body,
                OBSERVER.time.perf_counter(),
            )

            async def collect():
                return b"".join([chunk async for chunk in async_response.aiter_bytes()])

            with patch.object(OBSERVER.asyncio, "get_running_loop", return_value=FakeLoop()):
                coroutine = collect()
                try:
                    coroutine.send(None)
                except StopIteration as completed:
                    async_bytes = completed.value
                else:
                    self.fail("HTTPX async byte stream unexpectedly required a network event loop")

        self.assertEqual(sync_bytes, raw)
        self.assertEqual(async_bytes, raw)
        self.assertEqual([capture["llm_answer"] for capture in captures], ["Actual HTTPX stream."] * 2)

    def test_stream_capture_limit_is_reported_without_truncating_app_bytes(self):
        raw = b"stream response larger than the diagnostic limit"
        captured = []

        class FakeResponse:
            status_code = 200

            def iter_bytes(self, chunk_size=None):
                yield raw

        response = FakeResponse()
        body = {"model": "bounded-stream", "messages": [{"role": "user", "content": "question"}]}
        with patch.object(OBSERVER, "MAX_CAPTURED_STREAM_BYTES", 8), patch.object(OBSERVER, "_send_capture", side_effect=captured.append):
            OBSERVER._observe_httpx_sync_stream(
                response,
                "https://api.openai.com/v1/chat/completions",
                body,
                OBSERVER.time.perf_counter(),
            )
            returned = b"".join(response.iter_bytes())

        self.assertEqual(returned, raw)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["llm_answer"], "")
        self.assertEqual(captured[0]["metadata"]["error"], "stream_capture_limit_exceeded")
        self.assertTrue(captured[0]["metadata"]["capture_truncated"])


if __name__ == "__main__":
    unittest.main()
