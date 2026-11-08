"""Tiny stdlib-only connector from a local RAG app to the RAG Doctor extension."""

import json
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


class RAGDoctorProbe:
    """Send one run to a RAG Doctor service bound to this machine's loopback."""

    def __init__(self, server_url: str = "http://127.0.0.1:8765"):
        parsed = urlparse(server_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("RAG Doctor probe only supports an HTTP loopback endpoint.")
        self.server_url = server_url.rstrip("/")

    def capture(
        self,
        question: str,
        retrieved_chunks: List[Any],
        llm_answer: str,
        workspace_path: str = ".",
        expected_answer: Optional[str] = None,
        source_document_text: Optional[str] = None,
        prompt: Optional[str] = None,
        oracle_passed: Optional[bool] = None,
        max_index_gold_coverage: Optional[float] = None,
        top_k_final: Optional[int] = None,
        data_transfer_events: Optional[List[Dict[str, Any]]] = None,
        retrieval_observed: bool = True,
        retrieval_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_content: bool = False,
        include_retrieved_text: bool = False,
    ) -> Dict[str, Any]:
        chunk_fields = ("chunk_id", "doc_id", "id", "source", "final_rank", "rank", "score", "dense_score", "bm25_score", "rerank_score", "pre_rerank_rank")
        captured_chunks = []
        for item in retrieved_chunks[:50]:
            if not isinstance(item, dict):
                continue
            safe_chunk = {key: item[key] for key in chunk_fields if key in item}
            if include_content or include_retrieved_text:
                safe_chunk["text"] = item.get("text", item.get("page_content", item.get("content", "")))
            captured_chunks.append(item if include_content else safe_chunk)
        payload = {
            "question": question if include_content else "",
            "retrieved_chunks": captured_chunks,
            "llm_answer": str(llm_answer) if include_content else "",
            "workspace_path": str(workspace_path),
            "expected_answer": expected_answer if include_content else None,
            "source_document_text": source_document_text if include_content else None,
            "prompt": prompt if include_content else None,
            "oracle_passed": oracle_passed,
            "max_index_gold_coverage": max_index_gold_coverage,
            "top_k_final": top_k_final,
            "data_transfer_events": [
                {key: event[key] for key in ("method", "host", "status_code", "bytes_sent", "bytes_received", "duration_ms", "protocol", "operation") if key in event and isinstance(event[key], (str, int, float, bool, type(None)))}
                for event in (data_transfer_events or [])[:20] if isinstance(event, dict)
            ],
            "retrieval_observed": retrieval_observed,
            "retrieval_count": retrieval_count if retrieval_count is not None else len(retrieved_chunks),
            "metadata": metadata or {},
            "include_content": include_content,
            "include_retrieved_text": include_retrieved_text,
        }
        request = urllib.request.Request(
            f"{self.server_url}/api/capture-live-run",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            # Avoid returning exception text: request URLs can contain credentials.
            return {"status": "offline", "detail": "Local RAG Doctor service is unavailable."}


probe = RAGDoctorProbe()
