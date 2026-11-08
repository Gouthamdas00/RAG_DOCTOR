"""Bounded, read-only discovery of common local RAG log records."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.security import SecurityGuard


class LocalLogInspector:
    MAX_FILES = 40
    MAX_FILE_BYTES = 256 * 1024
    MAX_TOTAL_BYTES = 4 * 1024 * 1024
    MAX_RECORDS_PER_FILE = 3000
    MAX_DISCOVERED_FILES = 1000
    EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cache", "dist", "build"}

    FIELD_NAMES = {
        "question": {"question", "query", "userquery", "inputquestion", "userquestion", "querytext", "genairetrievalquerytext"},
        "prompt": {"prompt", "prompts", "assembledprompt", "llmprompt", "inputprompt", "formattedprompt", "fullprompt", "genaiinputmessages", "genaisysteminstructions"},
        "answer": {"answer", "response", "output", "llmanswer", "modelresponse", "completion", "generatedtext", "llmoutput", "genaioutputmessages"},
        "chunks": {"retrievedchunks", "retrievalresults", "retrieveddocs", "contexts", "contextchunks", "documents", "retrieveddocuments", "context", "sources", "genairetrievaldocuments"},
        "expected": {"expectedanswer", "goldanswer", "referenceanswer", "groundtruth"},
        "oracle": {"oraclepassed", "counterfactualpassed"},
        "coverage": {"maxindexgoldcoverage", "goldcoverage", "answercoverage"},
        "topk": {"topkfinal", "topk"},
        "trace_id": {"traceid", "trace_id"},
        "operation": {"operation", "spanname", "genaioperationname"},
        "provider": {"genaiprovidername", "provider", "providername"},
        "model": {"genairequestmodel", "genairesponsemodel", "model", "modelname"},
        "input_tokens": {"genaiusageinputtokens", "inputtokens", "prompttokens"},
        "output_tokens": {"genaiusageoutputtokens", "outputtokens", "completiontokens"},
        "top_k": {"genairetrievaltopk", "topk"},
        "retrieval_observed": {"retrievalobserved", "retrievalspanobserved", "retrievalexecuted"},
        "retrieval_count": {"retrievalcount", "candidatecount", "retrievedcount", "retrievalresultcount", "documentcount"},
        "error": {"error.type", "errortype", "exceptiontype"},
    }

    def scan(self, root: Path, include_content: bool = False) -> Dict[str, Any]:
        candidates: List[Path] = []
        visited = 0
        for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in self.EXCLUDED_DIRS]
            for name in names:
                visited += 1
                if visited > 30000:
                    break
                path = Path(current) / name
                suffix = path.suffix.lower()
                if suffix not in {".log", ".jsonl", ".ndjson"} and not (suffix == ".json" and any(k in path.name.lower() for k in ("rag", "trace", "run"))):
                    continue
                try:
                    path.resolve().relative_to(root)
                    candidates.append(path)
                    if len(candidates) >= self.MAX_DISCOVERED_FILES:
                        break
                except (OSError, ValueError):
                    continue
            if visited > 30000 or len(candidates) >= self.MAX_DISCOVERED_FILES:
                break
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

        scanned: List[str] = []
        latest: Optional[Dict[str, Any]] = None
        event_count = 0
        total_bytes = 0
        for path in candidates[: self.MAX_FILES]:
            try:
                size = path.stat().st_size
                if size <= 0 or size > self.MAX_FILE_BYTES or total_bytes + size > self.MAX_TOTAL_BYTES:
                    continue
                total_bytes += size
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                scanned.append(str(path.relative_to(root)))
                lines = text.splitlines()[-self.MAX_RECORDS_PER_FILE :]
                if path.suffix.lower() == ".json":
                    try:
                        whole_document = json.loads(text)
                        if isinstance(whole_document, (dict, list)):
                            lines = [text]
                    except json.JSONDecodeError:
                        pass
                trace_records: Dict[str, Dict[str, Any]] = {}
                file_latest: Optional[Dict[str, Any]] = None
                for line_no, line in enumerate(lines, start=max(1, len(text.splitlines()) - len(lines) + 1)):
                    for record in self._parse_records(line):
                        if not include_content:
                            record = self._metadata_only(record)
                        event_count += 1
                        trace_id = str(record.get("trace_id") or "").strip()
                        if trace_id:
                            aggregate = trace_records.setdefault(trace_id, {
                                "trace_id": trace_id,
                                "file": str(path.relative_to(root)),
                                "line": line_no,
                                "span_count": 0,
                            })
                            aggregate["span_count"] += 1
                            aggregate["line"] = line_no
                            for key, value in record.items():
                                if key == "trace_id" or value is None or value == "":
                                    continue
                                # Preserve an earlier populated retrieval result if a later
                                # span carries only model or timing metadata.
                                if key == "chunks" and not value and aggregate.get("chunks"):
                                    continue
                                aggregate[key] = value
                            file_latest = aggregate
                        else:
                            starts_run = bool(record.get("question") or record.get("prompt"))
                            if file_latest is None or starts_run:
                                file_latest = {"file": str(path.relative_to(root)), "line": line_no}
                            file_latest.update(record)
                if trace_records:
                    file_latest = max(trace_records.values(), key=lambda record: record.get("line", 0))
                if latest is None and file_latest:
                    latest = file_latest
            except (OSError, UnicodeError, ValueError):
                continue

        if latest:
            latest = self._bound_and_sanitize(latest)
        return {
            "files_scanned": scanned,
            "events_found": event_count,
            "latest_event": latest,
            "bytes_read": total_bytes,
            "storage": "memory_only",
            "content_policy": "included_by_user_setting" if include_content else "prompt_response_and_chunk_text_excluded",
        }

    @staticmethod
    def _metadata_only(record: Dict[str, Any]) -> Dict[str, Any]:
        safe = dict(record)
        had_content = any(bool(safe.get(key)) for key in ("question", "prompt", "answer", "expected"))
        for key in ("question", "prompt", "answer", "expected"):
            safe.pop(key, None)
        chunks = safe.get("chunks")
        if isinstance(chunks, list):
            metadata_chunks = []
            for chunk in chunks[:50]:
                if not isinstance(chunk, dict):
                    had_content = had_content or bool(chunk)
                    continue
                had_content = had_content or bool(chunk.get("text"))
                metadata_chunks.append({key: value for key, value in chunk.items() if key != "text"})
            safe["chunks"] = metadata_chunks
            safe["retrieval_observed"] = True
            safe["retrieval_count"] = safe.get("retrieval_count", len(metadata_chunks))
        safe["content_excluded"] = had_content
        return safe

    def _parse_records(self, line: str) -> List[Dict[str, Any]]:
        """Read a JSON event, newline-delimited record, or OTLP JSON span batch."""
        if not line.strip():
            return []
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            value = None
        if isinstance(value, dict):
            spans: List[Dict[str, Any]] = []
            for resource_group in value.get("resourceSpans", []) if isinstance(value.get("resourceSpans"), list) else []:
                if not isinstance(resource_group, dict):
                    continue
                resource = resource_group.get("resource", {})
                resource_attrs = resource.get("attributes", []) if isinstance(resource, dict) else []
                scope_groups = resource_group.get("scopeSpans", resource_group.get("instrumentationLibrarySpans", []))
                for scope_group in scope_groups if isinstance(scope_groups, list) else []:
                    if not isinstance(scope_group, dict):
                        continue
                    for span in scope_group.get("spans", []) if isinstance(scope_group.get("spans"), list) else []:
                        if not isinstance(span, dict):
                            continue
                        combined = {
                            "traceId": span.get("traceId"),
                            "spanId": span.get("spanId"),
                            "name": span.get("name"),
                            "startTimeUnixNano": span.get("startTimeUnixNano"),
                            "endTimeUnixNano": span.get("endTimeUnixNano"),
                            "attributes": (resource_attrs if isinstance(resource_attrs, list) else []) + (span.get("attributes", []) if isinstance(span.get("attributes"), list) else []),
                            "events": span.get("events", []),
                        }
                        record = self._extract_fields(combined)
                        if record and any(record.get(key) is not None for key in ("trace_id", "question", "prompt", "answer", "chunks", "operation", "input_tokens", "output_tokens")):
                            spans.append(record)
            if spans:
                return spans
        elif isinstance(value, list):
            records = []
            for item in value[:1000]:
                if isinstance(item, dict):
                    record = self._extract_fields(item)
                    if record and any(record.get(key) is not None for key in ("trace_id", "question", "prompt", "answer", "chunks", "operation")):
                        records.append(record)
            if records:
                return records
        record = self._parse_record(line, parsed=value)
        return [record] if record else []

    def _parse_record(self, line: str, parsed: Any = None) -> Optional[Dict[str, Any]]:
        if not line.strip():
            return None
        try:
            value = parsed if parsed is not None else json.loads(line)
            if isinstance(value, dict):
                record = self._extract_fields(value)
                if record and any(record.get(key) for key in ("question", "prompt", "answer", "chunks", "trace_id", "retrieval_observed")):
                    return record
        except (json.JSONDecodeError, TypeError):
            pass

        # Accept simple key=value / key: value log lines. These are lower-confidence
        # captures; we don't infer retrieved chunks from arbitrary prose.
        found: Dict[str, Any] = {}
        aliases = {
            "question": r"(?:question|query|user_input)",
            "answer": r"(?:answer|response|llm_output)",
            "prompt": r"(?:prompt|assembled_prompt)",
            "chunks": r"(?:retrieved_chunks|contexts|retrieved_context)",
        }
        for field, pattern in aliases.items():
            match = re.search(rf"\b{pattern}\s*[:=]\s*(.+)$", line, re.IGNORECASE)
            if match:
                value = match.group(1).strip().strip('"\'')
                found[field] = [{"text": value}] if field == "chunks" else value
        return found or None

    def _extract_fields(self, value: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        self._flatten_fields(value, normalized)
        result: Dict[str, Any] = {}
        for output_key, aliases in self.FIELD_NAMES.items():
            match = next((normalized[a] for a in aliases if a in normalized and normalized[a] is not None), None)
            if match is not None:
                if output_key == "chunks":
                    result[output_key] = self._normalize_chunks(match)
                elif output_key in {"question", "prompt", "answer"}:
                    result[output_key] = self._message_text(match)
                else:
                    result[output_key] = match
        if result.get("trace_id") is None and "traceid" in normalized:
            result["trace_id"] = normalized["traceid"]
        if "chunks" in result:
            result["retrieval_observed"] = True
            result["retrieval_count"] = len(result["chunks"])
        elif result.get("retrieval_count") is not None:
            result["retrieval_observed"] = True
        if "starttimeunixnano" in normalized and "endtimeunixnano" in normalized:
            try:
                start = int(normalized["starttimeunixnano"])
                end = int(normalized["endtimeunixnano"])
                if end >= start:
                    result["latency_ms"] = round((end - start) / 1_000_000, 3)
            except (TypeError, ValueError, OverflowError):
                pass
        if result.get("operation") is None and isinstance(value.get("name"), str):
            result["operation"] = value["name"]
        return result

    @classmethod
    def _flatten_fields(cls, value: Any, output: Dict[str, Any]) -> None:
        """Flatten normal JSON and OpenTelemetry attribute-list encodings."""
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
                if normalized_key == "attributes" and isinstance(item, list):
                    for attribute in item:
                        if isinstance(attribute, dict) and isinstance(attribute.get("key"), str):
                            attr_key = re.sub(r"[^a-z0-9]", "", attribute["key"].lower())
                            output[attr_key] = cls._unwrap_otel_value(attribute.get("value"))
                            cls._flatten_fields(attribute.get("value"), output)
                    continue
                decoded = cls._unwrap_otel_value(item)
                # Preserve a direct field over a nested duplicate, but let a later
                # attribute-list value replace an earlier one in source order.
                output[normalized_key] = decoded
                if isinstance(item, (dict, list)):
                    cls._flatten_fields(item, output)
        elif isinstance(value, list):
            for item in value:
                cls._flatten_fields(item, output)

    @classmethod
    def _unwrap_otel_value(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        for wrapper in ("stringValue", "intValue", "doubleValue", "boolValue", "bytesValue"):
            if wrapper in value:
                return value[wrapper]
        array = value.get("arrayValue")
        if isinstance(array, dict):
            return [cls._unwrap_otel_value(item) for item in array.get("values", [])]
        kvlist = value.get("kvlistValue")
        if isinstance(kvlist, dict):
            return {
                item.get("key"): cls._unwrap_otel_value(item.get("value"))
                for item in kvlist.get("values", [])
                if isinstance(item, dict) and isinstance(item.get("key"), str)
            }
        return value

    @classmethod
    def _message_text(cls, value: Any) -> str:
        """Extract text from common chat-message and OpenTelemetry AnyValue shapes."""
        value = cls._unwrap_otel_value(value)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [cls._message_text(item) for item in value]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in ("text", "content", "result"):
                item = value.get(key)
                if isinstance(item, (str, list, dict)):
                    extracted = cls._message_text(item)
                    if extracted:
                        return extracted
            return "\n".join(
                text for key, item in value.items()
                if key in {"parts", "messages", "message", "content"}
                for text in [cls._message_text(item)] if text
            )
        return ""

    @staticmethod
    def _normalize_chunks(value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, str):
            return [{"text": value}]
        if not isinstance(value, list):
            return []
        chunks: List[Dict[str, Any]] = []
        for index, item in enumerate(value[:10], start=1):
            if isinstance(item, dict):
                text = item.get("text") or item.get("page_content") or item.get("content") or item.get("document") or ""
                chunk = {"text": str(text), "final_rank": item.get("final_rank", item.get("rank", index))}
                chunk_id = item.get("chunk_id", item.get("id"))
                doc_id = item.get("doc_id", item.get("source"))
                if chunk_id is not None:
                    chunk["chunk_id"] = chunk_id
                if doc_id is not None:
                    chunk["doc_id"] = doc_id
                if "score" in item:
                    chunk["score"] = item["score"]
                for key in ("chunk_id", "doc_id", "score", "dense_score", "bm25_score", "rerank_score", "pre_rerank_rank"):
                    if key in item:
                        chunk[key] = item[key]
                chunks.append(chunk)
            else:
                chunks.append({"text": str(item), "final_rank": index})
        return chunks

    @staticmethod
    def _bound_and_sanitize(record: Dict[str, Any]) -> Dict[str, Any]:
        safe = dict(record)
        for key in ("question", "prompt", "answer", "expected"):
            if key in safe:
                safe[key] = SecurityGuard.sanitize_text(str(safe[key])[:20000])
        for key in ("trace_id", "operation", "provider", "model", "error"):
            if key in safe and safe[key] is not None:
                safe[key] = SecurityGuard.sanitize_text(str(safe[key])[:500])
        if isinstance(safe.get("chunks"), list):
            safe["chunks"] = [
                {**chunk, "text": SecurityGuard.sanitize_text(str(chunk.get("text", ""))[:6000])}
                for chunk in safe["chunks"][:10]
            ]
        if "chunks" in safe:
            safe["retrieval_count"] = safe.get("retrieval_count", len(safe["chunks"]))
            safe["retrieval_observed"] = True
        return safe
