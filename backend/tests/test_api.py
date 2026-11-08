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


if __name__ == "__main__":
    test_security_guard_redacts_secrets_and_pii()
    test_ast_workspace_inspector_finds_exact_file_and_line()
    test_fastapi_diagnose_and_autotune_endpoints()
    print("All Zero-Storage Stateless Mode, Security Guard, and AST Code Line Scanner tests passed!")
