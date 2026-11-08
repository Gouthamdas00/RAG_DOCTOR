import json
import urllib.request
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional


class RAGDoctorProbe:
    """
    Lightweight, zero-dependency probe you can call inside your own RAG script
    to push the latest execution (query, retrieved chunks, answer, and source path)
    into RAG Doctor's volatile memory for instant root-cause diagnosis.
    """

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
    ) -> Dict[str, Any]:
        payload = {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "llm_answer": str(llm_answer),
            "workspace_path": str(workspace_path),
            "expected_answer": expected_answer,
            "source_document_text": source_document_text,
            "prompt": prompt,
            "oracle_passed": oracle_passed,
            "max_index_gold_coverage": max_index_gold_coverage,
            "top_k_final": top_k_final,
            "data_transfer_events": data_transfer_events or [],
        }
        req = urllib.request.Request(
            f"{self.server_url}/api/capture-live-run",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            # URL errors can contain credentials; do not return them to application logs.
            return {"status": "offline", "detail": "Local RAG Doctor service is unavailable."}


# Singleton convenience helper
probe = RAGDoctorProbe()
