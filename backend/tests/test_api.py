from fastapi.testclient import TestClient
from backend.app.api.server import DOCUMENTS, app
from backend.app.core.security import SecurityGuard
from backend.app.pipeline.testgen import SyntheticBenchmarkGenerator
from backend.app.schemas.contracts import PipelineConfig

client = TestClient(app)


def test_security_guard_redacts_secrets_and_pii():
    dirty = "Contact admin@corp.com or use key sk-1234567890abcdefghijklmno and SSN 123-45-6789"
    clean = SecurityGuard.sanitize_text(dirty)
    assert "[REDACTED_EMAIL]" in clean
    assert "[REDACTED_API_KEY]" in clean
    assert "[REDACTED_SSN]" in clean


def test_ast_workspace_inspector_finds_exact_file_and_line():
    resp = client.get("/api/inspect-workspace")
    assert resp.status_code == 200
    data = resp.json()
    assert data["storage_mode"] == "ZERO_STORAGE_VOLATILE_RAM"
    assert data["security_audit"]["pii_redaction"] == "ACTIVE"

    # Verify AST scanner caught the bug in my_local_rag.py with exact line numbers!
    warnings = data["ast_code_scan"]["static_warnings"]
    assert len(warnings) >= 1
    assert any("my_local_rag.py" in w["file"] and w["line"] > 0 for w in warnings)
    assert data["last_response_diagnosis"]["root_cause"] in ("CHUNKING", "RETRIEVAL", "RANKING_K")


def test_fastapi_diagnose_and_autotune_endpoints():
    bad_cfg = PipelineConfig(chunk_size=64, chunk_overlap=0, top_k_final=1)
    resp = client.post("/api/diagnose", json=bad_cfg.model_dump())
    assert resp.status_code == 200
    data = resp.json()
    assert data["storage_mode"] == "ZERO_STORAGE_VOLATILE_RAM"
    assert data["summary"]["accuracy"] < 1.0

    tune_resp = client.post("/api/autotune", json=bad_cfg.model_dump())
    assert tune_resp.status_code == 200
    tune_data = tune_resp.json()
    assert tune_data["best_recommended"]["accuracy"] == 1.0


def test_workspace_scanner_pins_chunk_step_that_ignores_overlap():
    import tempfile
    from pathlib import Path
    from backend.app.diagnose.inspector import WorkspaceASTInspector

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "src" / "ingestion" / "chunker.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "class Chunker:\n"
            "    def __init__(self, chunk_size=800, chunk_overlap=100):\n"
            "        self.chunk_size = chunk_size\n"
            "        self.chunk_overlap = chunk_overlap\n"
            "    def split(self, text):\n"
            "        step = self.chunk_size\n"
            "        start = 0\n"
            "        while start < len(text):\n"
            "            yield text[start:start + self.chunk_size]\n"
            "            start += step\n",
            encoding="utf-8",
        )
        report = WorkspaceASTInspector().scan_workspace(str(root))

    finding = next(warning for warning in report["static_warnings"] if "overlap is not applied" in warning["problem"])
    assert Path(finding["file"]).as_posix() == "src/ingestion/chunker.py"
    assert finding["line"] == 6
    assert finding["root_cause"] == "CHUNKING"


def test_workspace_scanner_finds_zero_uppercase_retrieval_limit_at_assignment():
    import tempfile
    from pathlib import Path
    from backend.app.diagnose.inspector import WorkspaceASTInspector

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "src" / "config.py"
        source.parent.mkdir(parents=True)
        source.write_text("TOP_K_RETRIEVAL: int = 0\n", encoding="utf-8")
        report = WorkspaceASTInspector().scan_workspace(str(root))

    finding = next(warning for warning in report["static_warnings"] if warning["root_cause"] == "RETRIEVAL")
    assert Path(finding["file"]).as_posix() == "src/config.py"
    assert finding["line"] == 1
    assert "positive value" in finding["fix_suggestion"]


def test_reranker_location_requires_matching_runtime_score_order():
    import tempfile
    from pathlib import Path
    from backend.app.api.server import _runtime_correlated_reranker_sort

    diagnosis = {
        "root_cause": "RERANKER",
        "expected_answer": "TARGET_ANSWER",
        "runtime_parameters": {"top_k_final": 1},
        "retrieved_chunks": [
            {"text": "Distractor", "final_rank": 1, "pre_rerank_rank": 2, "rerank_score": 0.1},
            {"text": "TARGET_ANSWER", "final_rank": 2, "pre_rerank_rank": 1, "rerank_score": 0.9},
        ],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reranker = root / "reranker.py"
        reranker.write_text(
            "rescored = []\nrescored.sort(key=lambda item: item[1], reverse=False)\n",
            encoding="utf-8",
        )
        reference = _runtime_correlated_reranker_sort(root, diagnosis)
        assert reference is not None
        assert Path(reference["file"]).as_posix() == "reranker.py"
        assert reference["line"] == 2
        assert reference["match_type"] == "runtime_score_order"

        reranker.write_text(
            "rescored = []\nrescored.sort(key=lambda item: item[0], reverse=False)\n",
            encoding="utf-8",
        )
        assert _runtime_correlated_reranker_sort(root, diagnosis) is None


if __name__ == "__main__":
    test_security_guard_redacts_secrets_and_pii()
    test_ast_workspace_inspector_finds_exact_file_and_line()
    test_fastapi_diagnose_and_autotune_endpoints()
    print("All Zero-Storage Stateless Mode, Security Guard, and AST Code Line Scanner tests passed!")
