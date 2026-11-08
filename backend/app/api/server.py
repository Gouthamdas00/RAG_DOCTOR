from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.app.core.cache import DiskCache
from backend.app.core.security import SecurityGuard
from backend.app.diagnose.agentic_investigator import investigate_case
from backend.app.diagnose.arithmetic_audit import audit_arithmetic
from backend.app.diagnose.clustering import FailureClusterer
from backend.app.diagnose.inspector import WorkspaceASTInspector
from backend.app.diagnose.live_diagnoser import diagnose_live_run
from backend.app.diagnose.log_inspector import LocalLogInspector
from backend.app.diagnose.ml_diagnoser import MLTraceDiagnoser
from backend.app.pipeline.runner import RAGPipelineRunner
from backend.app.schemas.contracts import (
    BenchmarkItem,
    CharSpan,
    ChunkStrategy,
    Document,
    FailureLabel,
    PipelineConfig,
    RetrieverType,
)
from backend.app.tuner.optimizer import ParetoAutoTuner

app = FastAPI(title="RAG Doctor — Local IDE Diagnostic & Code Inspector", version="2.4.64")

# Latest run per workspace, retained only in this local process's RAM.
VOLATILE_EXTERNAL_RUNS: Dict[str, Dict[str, Any]] = {}
RUNTIME_AGENT_READY: Dict[str, Dict[str, Any]] = {}
WORKSPACE_CODE_REPORTS: Dict[str, Tuple[float, Dict[str, Any]]] = {}
LOG_INSPECTOR = LocalLogInspector()


class LiveCaptureRequest(BaseModel):
    question: str = Field(default="", max_length=10000)
    retrieved_chunks: Optional[List[Any]] = Field(default=None, max_length=50)
    prompt_context_blocks: Optional[List[Dict[str, Any]]] = Field(default=None, max_length=10)
    llm_answer: str = Field(default="", max_length=20000)
    workspace_path: str = "."
    prompt: Optional[str] = Field(default=None, max_length=30000)
    expected_answer: Optional[str] = Field(default=None, max_length=5000)
    source_document_text: Optional[str] = Field(default=None, max_length=100000)
    oracle_passed: Optional[bool] = None
    max_index_gold_coverage: Optional[float] = Field(default=None, ge=0, le=1)
    top_k_final: Optional[int] = Field(default=None, ge=1, le=100)
    data_transfer_events: List[Dict[str, Any]] = Field(default_factory=list, max_length=20)
    retrieval_observed: Optional[bool] = None
    retrieval_count: Optional[int] = Field(default=None, ge=0, le=10000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    include_content: bool = False
    include_retrieved_text: bool = False


class EvaluateCapturedRunRequest(BaseModel):
    workspace_path: str
    expected_answer: str = Field(min_length=1, max_length=5000)
    case_id: Optional[str] = Field(default=None, max_length=120)
    gold_document_id: Optional[str] = Field(default=None, max_length=300)
    question_type: Optional[str] = Field(default=None, max_length=100)
    top_k_final: Optional[int] = Field(default=None, ge=1, le=100)


class AgentInvestigationRequest(BaseModel):
    workspace_path: str
    case_id: Optional[str] = Field(default=None, max_length=120)
    question: str = Field(default="", max_length=10000)
    expected_answer: str = Field(default="", max_length=5000)
    gold_document_id: Optional[str] = Field(default=None, max_length=300)
    question_type: Optional[str] = Field(default=None, max_length=100)
    diagnosis: Dict[str, Any] = Field(default_factory=dict)
    local_model_endpoint: str = Field(default="http://127.0.0.1:11434/api/chat", max_length=500)
    local_model_name: str = Field(default="", max_length=120)
    local_model_thinking: bool = True


class RuntimeAgentRequest(BaseModel):
    workspace_path: str
    process_id: Optional[int] = None
    providers: List[str] = Field(default_factory=list, max_length=20)
    observer_installed: bool = True
    detail: Optional[str] = Field(default=None, max_length=300)


def _configured_workspace_roots() -> List[Path]:
    configured = os.getenv("RAG_DOCTOR_ALLOWED_ROOTS", "").strip()
    if configured:
        try:
            values = json.loads(configured)
            if isinstance(values, list):
                roots = [Path(str(value)).resolve() for value in values]
                if roots:
                    return roots
        except (json.JSONDecodeError, OSError, TypeError):
            raise HTTPException(status_code=500, detail="Invalid local workspace root configuration.")
    return [Path.cwd().resolve()]


def _resolve_workspace(target_path: str) -> Path:
    target = Path(target_path).expanduser().resolve()
    matches: List[Path] = []
    for allowed_root in _configured_workspace_roots():
        try:
            SecurityGuard.validate_workspace_path(str(target), str(allowed_root))
            matches.append(allowed_root)
        except PermissionError:
            continue
    if matches:
        # Canonicalize app subdirectories to the open VS Code workspace root, so
        # captures and later scans address the same in-memory run entry.
        return max(matches, key=lambda candidate: len(candidate.parts))
    raise HTTPException(status_code=403, detail="Requested path is outside the VS Code workspace folders.")


def _sanitize_run(req: LiveCaptureRequest, workspace: Path) -> Dict[str, Any]:
    include_content = req.include_content is True
    include_retrieved_text = include_content or req.include_retrieved_text is True
    chunks: List[Dict[str, Any]] = []
    retrieved_chunks = req.retrieved_chunks or []
    for index, chunk in enumerate(retrieved_chunks[:50], start=1):
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("page_content") or chunk.get("content") or ""
            rank = chunk.get("final_rank", chunk.get("rank", index))
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
                rank = index
            safe_chunk: Dict[str, Any] = {
                "text": SecurityGuard.sanitize_text(str(text)[:10000]) if include_retrieved_text else "",
                "final_rank": rank,
            }
            for target, aliases in {
                "chunk_id": ("chunk_id", "id"),
                "doc_id": ("doc_id", "source"),
                "dense_score": ("dense_score", "score", "similarity"),
                "bm25_score": ("bm25_score",),
                "rerank_score": ("rerank_score",),
                "pre_rerank_rank": ("pre_rerank_rank",),
            }.items():
                value = next((chunk[key] for key in aliases if key in chunk), None)
                if isinstance(value, (str, int, float, type(None))):
                    safe_chunk[target] = SecurityGuard.sanitize_text(value[:300]) if isinstance(value, str) else value
            chunks.append(safe_chunk)
        else:
            chunks.append({"text": SecurityGuard.sanitize_text(str(chunk)[:10000]), "final_rank": index})
    prompt_context_blocks: List[Dict[str, Any]] = []
    for index, block in enumerate((req.prompt_context_blocks or [])[:10], start=1):
        if not isinstance(block, dict):
            continue
        raw_text = block.get("text") or ""
        label = SecurityGuard.sanitize_text(str(block.get("label") or "Context block")[:200])
        method = str(block.get("extraction_method") or "").strip().lower()
        confidence = str(block.get("confidence") or "").strip().lower()
        position = block.get("prompt_position", index)
        if not isinstance(position, int) or isinstance(position, bool) or not 1 <= position <= 10:
            position = index
        prompt_context_blocks.append({
            "text": SecurityGuard.sanitize_text(str(raw_text)[:10000]) if include_content else "",
            "label": label,
            "prompt_position": position,
            "extraction_method": method if method in {"prompt_source_marker", "prompt_context_tag", "prompt_context_heading"} else "unknown",
            "confidence": confidence if confidence in {"high", "moderate"} else "limited",
        })
    clean_expected = SecurityGuard.sanitize_text(req.expected_answer[:5000]) if include_content and req.expected_answer else None
    source_text = SecurityGuard.sanitize_text(req.source_document_text[:100000]) if include_content and req.source_document_text else ""
    expected_in_source = None
    if clean_expected and source_text:
        expected_terms = re.findall(r"[a-z]+|\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d)", "", clean_expected.lower()))
        source_terms = re.findall(r"[a-z]+|\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d)", "", source_text.lower()))
        width = len(expected_terms)
        # A missing literal phrase is not proof the fact is absent; paraphrases and
        # domain-specific formatting need a semantic evaluator or human label.
        expected_in_source = bool(width) and any(source_terms[index:index + width] == expected_terms for index in range(len(source_terms) - width + 1))
        if not expected_in_source:
            expected_in_source = None
    allowed_metadata = {"trace_id", "operation", "provider", "model", "input_tokens", "output_tokens", "latency_ms", "error", "span_count", "collection", "streaming", "capture_truncated", "context_extraction_method", "prompt_context_block_count"}
    safe_metadata: Dict[str, Any] = {}
    for key, value in list(req.metadata.items())[:30]:
        normalized_key = str(key).strip().lower()
        if normalized_key not in allowed_metadata or not isinstance(value, (str, int, float, bool, type(None))):
            continue
        safe_metadata[normalized_key] = SecurityGuard.sanitize_text(str(value)[:300]) if isinstance(value, str) else value
    return {
        "workspace_path": str(workspace),
        "question": SecurityGuard.sanitize_text(req.question[:10000]) if include_content else "",
        "prompt": SecurityGuard.sanitize_text((req.prompt or "")[:30000]) if include_content else "",
        "retrieved_chunks": chunks,
        "prompt_context_blocks": prompt_context_blocks,
        "prompt_context_count": len(req.prompt_context_blocks or []),
        # An empty list from an inference observer often means it only saw the
        # model request and could not extract context. Treat that as unknown;
        # explicit zero-result retrieval telemetry must set retrieval_observed
        # and retrieval_count itself.
        "retrieval_observed": req.retrieval_observed if req.retrieval_observed is not None else (True if retrieved_chunks else None),
        "retrieval_count": req.retrieval_count if req.retrieval_count is not None else (len(retrieved_chunks) if retrieved_chunks else None),
        "metadata": safe_metadata,
        "content_excluded": not include_retrieved_text,
        "retrieved_text_included": include_retrieved_text,
        "prompt_context_content_included": bool(include_content and any(block.get("text") for block in prompt_context_blocks)),
        "prompt_response_excluded": not include_content,
        "llm_answer": SecurityGuard.sanitize_text(req.llm_answer[:20000]) if include_content else "",
        "include_content": include_content,
        "expected_answer": clean_expected,
        "source_document_provided": bool(source_text),
        "expected_answer_in_source": expected_in_source,
        "oracle_passed": req.oracle_passed,
        "max_index_gold_coverage": req.max_index_gold_coverage,
        "top_k_final": req.top_k_final,
        # Keep only transport metadata; request/response bodies can contain prompts,
        # customer data, or credentials and are not needed for this signal.
        "data_transfer_events": [
            {
                key: SecurityGuard.sanitize_text(str(event[key])[:300]) if isinstance(event[key], str) else event[key]
                for key in ("method", "host", "status_code", "bytes_sent", "bytes_received", "duration_ms", "protocol", "operation")
                if key in event and isinstance(event[key], (str, int, float, bool, type(None)))
            }
            for event in req.data_transfer_events[:20]
        ],
    }


def _normalized_numeric_values(value: Any) -> List[str]:
    text = re.sub(r"(?<=\d),(?=\d)", "", str(value or ""))
    values = []
    for token in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text):
        try:
            values.append(format(Decimal(token).normalize(), "f"))
        except InvalidOperation:
            continue
    return values


def _code_references_for_diagnosis(workspace: Path, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Attach bounded, workspace-relative code locations to a live finding."""
    key = str(workspace)
    now = time.monotonic()
    cached = WORKSPACE_CODE_REPORTS.get(key)
    if cached and now - cached[0] < 30:
        report = cached[1]
    else:
        report = AST_INSPECTOR.scan_workspace(key)
        WORKSPACE_CODE_REPORTS[key] = (now, report)
        while len(WORKSPACE_CODE_REPORTS) > 16:
            WORKSPACE_CODE_REPORTS.pop(next(iter(WORKSPACE_CODE_REPORTS)))

    root_cause = diagnosis.get("root_cause")
    status = diagnosis.get("status")
    if root_cause == "NO_FAILURE" or status in {"no_obvious_issue", "reference_supported"}:
        return []
    component_map = {
        "CHUNKING": ("chunking",),
        "RETRIEVAL": ("retrieval",),
        "RANKING_K": ("retrieval",),
        "RERANKER": ("reranking",),
        "CONTEXT_NOISE": ("prompt", "generation"),
        "GENERATOR": ("generation", "prompt"),
        "ABSTENTION": ("generation", "prompt"),
    }
    components = component_map.get(root_cause)
    if not components and status == "needs_review":
        finding = diagnosis.get("finding")
        if finding == "TARGET_TERMS_MISSING_FROM_CONTEXT":
            components = ("retrieval",)
        elif finding == "DERIVED_RESULT_UNVERIFIED":
            components = ("generation", "prompt", "retrieval")
        elif finding == "TOPIC_PRESENT_REFERENCE_NOT_VERIFIED":
            components = ("generation", "retrieval")
        elif finding == "PARTIAL_REFERENCE_MATCH":
            components = ("generation", "prompt")
        elif diagnosis.get("gold_document_retrieved") is True:
            components = ("retrieval", "generation")
        else:
            components = ("retrieval", "generation")
    if not components:
        return []

    parameter_map = {
        "chunking": {"chunk_size", "chunk_overlap"},
        "retrieval": {"top_k", "top_k_final", "top_n_candidates"},
        "reranking": {"top_k", "top_k_final", "top_n_candidates"},
        "generation": {"temperature"},
    }
    refs: List[Dict[str, Any]] = []
    has_causal_stage = root_cause in component_map
    warnings = [warning for warning in report.get("static_warnings", []) if has_causal_stage and warning.get("root_cause") == root_cause]
    for warning in warnings:
        refs.append({
            "kind": "configuration_finding",
            "file": warning.get("file"),
            "line": warning.get("line"),
            "label": f"{warning.get('problem', 'Detected configuration finding')} Suggested experiment: {warning.get('fix_suggestion', 'review this setting on a labeled case.')}",
            "code_snippet": warning.get("code_snippet", ""),
        })

    locations = report.get("code_locations", [])
    for component in components:
        # Show parameter experiments only when a stage has evidence-backed
        # attribution. A review-only case must not receive an implied tuning fix.
        if has_causal_stage and status != "needs_review":
            for knob in locations:
                if knob.get("parameter") not in parameter_map.get(component, set()):
                    continue
                value = knob.get("value")
                parameter = knob.get("parameter")
                runtime_top_k = (diagnosis.get("runtime_parameters") or {}).get("top_k_final")
                if parameter in {"top_k", "top_k_final", "top_n_candidates"} and root_cause == "RETRIEVAL" and diagnosis.get("gold_document_retrieved") is False:
                    current = runtime_top_k if isinstance(runtime_top_k, int) else (value if isinstance(value, int) else None)
                    if current is not None and current < 100:
                        trial = min(100, max(current + 3, current * 2))
                        label = f"Detected `{parameter}={value}` here; the replay used top_k={current}. For this case only, compare top_k={current} vs top_k={trial}; verify whether the labeled source enters the final prompt, then compare latency and token use before changing the default."
                    else:
                        label = f"Detected `{parameter}={value}`. Record the effective candidate limit at this call, then compare it with a larger limit on this labeled case; keep a change only if the gold source enters the final prompt and latency remains acceptable."
                else:
                    label = f"Detected `{parameter}={value}` in {knob.get('call', 'configuration')} code. Use this as the baseline for a one-variable A/B run; keep the change only if the labeled case improves without unacceptable latency or token cost."
                refs.append({
                    "kind": "parameter",
                    "parameter": parameter,
                    "value": value,
                    "file": knob.get("file"),
                    "line": knob.get("line"),
                    "label": label,
                    "code_snippet": knob.get("code_snippet", ""),
                })
                if len(refs) >= 4:
                    break
        for location in report.get("pipeline_locations", []):
            if location.get("component") != component:
                continue
            snippet = str(location.get("code_snippet", ""))
            if component == "retrieval":
                runtime_top_k = (diagnosis.get("runtime_parameters") or {}).get("top_k_final")
                if diagnosis.get("finding") == "TARGET_TERMS_MISSING_FROM_CONTEXT" and isinstance(runtime_top_k, int):
                    current = runtime_top_k
                    trial = min(100, max(current + 3, current * 2))
                    label = f"The filing was returned but the distinctive query terms were mostly absent from captured chunks. A/B test top_k={current} vs top_k={trial} for this case; check whether the target statement section enters the final prompt before changing the default."
                elif diagnosis.get("gold_document_retrieved") is True or diagnosis.get("derived_question"):
                    label = "The labeled source was retrieved, so this run does not justify changing top_k. Inspect the retrieved values passed onward from this call only to confirm the required inputs reached generation."
                elif root_cause == "RETRIEVAL" and diagnosis.get("gold_document_retrieved") is False and isinstance(runtime_top_k, int):
                    label = f"The labeled source was missing from this run. The replay passed top_k={runtime_top_k} here; A/B test a larger value for this case and check whether the target document enters the final context."
                else:
                    label = "Inspect the query and candidate limit at this call to confirm which inputs and filters were applied. Current evidence does not justify a retrieval setting change."
            elif component in {"generation", "prompt"} and status == "needs_review":
                label = "This is a relevant prompt/generation call site. Check the retrieved values passed into it and verify the calculation or interpretation before tuning retrieval."
            elif component in {"generation", "prompt"}:
                label = "Inspect the prompt and answer constraints at this call. Compare the same retrieved evidence with an isolated-context run before changing models or settings."
            else:
                label = f"Inspect this {component} call against the captured run evidence before changing its configuration."
            refs.append({
                "kind": "pipeline_call",
                "component": component,
                "file": location.get("file"),
                "line": location.get("line"),
                "label": label,
                "code_snippet": snippet,
            })
            if len(refs) >= 4:
                break
        if len(refs) >= 4:
            break

    # Keep the report compact and avoid duplicate file/line entries from overlapping patterns.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for ref in refs:
        identity = (ref.get("file"), ref.get("line"))
        if ref.get("file") and identity not in seen:
            seen.add(identity)
            deduped.append(ref)
        if len(deduped) == 4:
            break
    return deduped


def build_default_workspace() -> Tuple[List[Document], List[BenchmarkItem]]:
    doc1_text = (
        "Section 1: Cloud Gateway Rate Limiting.\n"
        "Standard tier accounts are limited to 1200 requests per minute across all regional API endpoints.\n\n"
        "Section 2: Database Failover SLA.\n"
        "In the event of a primary node outage, automated replica promotion completes within 45 seconds.\n\n"
        "Section 3: Backup Retention Policy.\n"
        "Automated encrypted snapshots are retained for 35 days in cold object storage vaults.\n\n"
        "Section 4: Authentication Token Lifecycle.\n"
        "Short-lived OAuth access tokens expire every 15 minutes and require refresh token rotation.\n\n"
        "Section 5: Webhook Delivery Retries.\n"
        "Failed webhook deliveries are retried up to 8 times using exponential backoff jitter.\n\n"
        "Section 6: Pod Eviction Memory Thresholds.\n"
        "The kubelet evicts burstable pods when available node memory drops below 150Mi."
    )
    doc = Document(
        doc_id="cloud_architecture_spec",
        title="Cloud Production Architecture & SLA Spec",
        text=doc1_text,
    )

    def span_of(substr: str) -> CharSpan:
        s = doc1_text.index(substr)
        return CharSpan(start_char=s, end_char=s + len(substr))

    benchmarks = [
        BenchmarkItem(
            question_id="q_rate",
            question="How many requests per minute are allowed for standard tier accounts across regional API endpoints?",
            gold_answer="1200 requests per minute",
            gold_doc_id=doc.doc_id,
            gold_span=span_of("Standard tier accounts are limited to 1200 requests per minute across all regional API endpoints."),
            question_type="numeric",
        ),
        BenchmarkItem(
            question_id="q_failover",
            question="Within how many seconds does automated replica promotion complete during a primary node outage?",
            gold_answer="45 seconds",
            gold_doc_id=doc.doc_id,
            gold_span=span_of("In the event of a primary node outage, automated replica promotion completes within 45 seconds."),
            question_type="numeric",
        ),
        BenchmarkItem(
            question_id="q_backup",
            question="For how many days are automated encrypted snapshots retained in cold object storage vaults?",
            gold_answer="35 days",
            gold_doc_id=doc.doc_id,
            gold_span=span_of("Automated encrypted snapshots are retained for 35 days in cold object storage vaults."),
            question_type="numeric",
        ),
        BenchmarkItem(
            question_id="q_auth",
            question="Every how many minutes do short-lived OAuth access tokens expire?",
            gold_answer="15 minutes",
            gold_doc_id=doc.doc_id,
            gold_span=span_of("Short-lived OAuth access tokens expire every 15 minutes and require refresh token rotation."),
            question_type="numeric",
        ),
        BenchmarkItem(
            question_id="q_webhook",
            question="Up to how many times are failed webhook deliveries retried using exponential backoff?",
            gold_answer="8 times",
            gold_doc_id=doc.doc_id,
            gold_span=span_of("Failed webhook deliveries are retried up to 8 times using exponential backoff jitter."),
            question_type="numeric",
        ),
        BenchmarkItem(
            question_id="q_evict",
            question="Below what node memory threshold does the kubelet evict burstable pods?",
            gold_answer="150Mi",
            gold_doc_id=doc.doc_id,
            gold_span=span_of("The kubelet evicts burstable pods when available node memory drops below 150Mi."),
            question_type="numeric",
        ),
        BenchmarkItem(
            question_id="q_unans",
            question="What is the satellite uplink bandwidth quota for lunar edge data centers?",
            gold_answer="UNANSWERABLE",
            gold_doc_id=doc.doc_id,
            gold_span=CharSpan(start_char=0, end_char=0),
            question_type="unanswerable",
        ),
    ]
    return [doc], benchmarks


DOCUMENTS, BENCHMARKS = build_default_workspace()
AST_INSPECTOR = WorkspaceASTInspector()


@app.get("/")
def serve_dashboard():
    return FileResponse(Path("frontend/index.html"))


@app.post("/api/capture-live-run")
def capture_live_external_rag_run(req: LiveCaptureRequest) -> Dict[str, Any]:
    """Capture one local run. Only the latest run per open workspace stays in RAM."""
    workspace = _resolve_workspace(req.workspace_path)
    run = _sanitize_run(req, workspace)
    run["security_audit"] = {
        "pii_redaction": "ACTIVE",
        "prompt_injection_detected": SecurityGuard.scan_for_prompt_injection(
            run["prompt"] + "\n" + "\n".join(c["text"] for c in run["retrieved_chunks"])
        )["injection_detected"],
        "workspace_jail": "ENFORCED",
        "storage": "VOLATILE_RAM_ONLY",
        "network_uploads_by_rag_doctor": False,
    }
    VOLATILE_EXTERNAL_RUNS[str(workspace)] = run
    while len(VOLATILE_EXTERNAL_RUNS) > 16:
        VOLATILE_EXTERNAL_RUNS.pop(next(iter(VOLATILE_EXTERNAL_RUNS)))
    return {
        "status": "captured",
        "storage": "VOLATILE_RAM_ONLY",
        "workspace_root": str(workspace),
        "message": "Run captured locally. Open RAG Doctor in VS Code to inspect it.",
    }


@app.post("/api/evaluate-captured-run")
def evaluate_captured_run(req: EvaluateCapturedRunRequest) -> Dict[str, Any]:
    """Attach a local test-set reference to the latest intercepted app run."""
    workspace = _resolve_workspace(req.workspace_path)
    captured = VOLATILE_EXTERNAL_RUNS.get(str(workspace))
    if not captured:
        raise HTTPException(status_code=404, detail="No runtime capture was received for this workspace.")
    run = dict(captured)
    run["expected_answer"] = SecurityGuard.sanitize_text(req.expected_answer[:5000])
    run["top_k_final"] = req.top_k_final
    retrieved_docs = [
        str(chunk.get("doc_id") or chunk.get("source") or "").strip().lower()
        for chunk in run.get("retrieved_chunks", []) if isinstance(chunk, dict)
    ]
    expected_doc = (req.gold_document_id or "").strip().lower()
    matching_ranks = [
        int(chunk.get("final_rank", index))
        for index, chunk in enumerate(run.get("retrieved_chunks", []), start=1)
        if isinstance(chunk, dict)
        and expected_doc
        and expected_doc in str(chunk.get("doc_id") or chunk.get("source") or "").strip().lower()
    ]
    gold_document_retrieved = any(expected_doc in doc for doc in retrieved_docs) if expected_doc else None
    run["gold_document_retrieved"] = gold_document_retrieved
    run["gold_document_rank"] = min(matching_ranks) if matching_ranks else None
    run["question_type"] = req.question_type
    diagnosis = diagnose_live_run(run)
    diagnosis["source"] = "run_with_doctor"
    diagnosis["case_id"] = req.case_id
    diagnosis["gold_document_id"] = req.gold_document_id
    diagnosis["gold_document_retrieved"] = gold_document_retrieved
    diagnosis["runtime_parameters"] = {"top_k_final": req.top_k_final} if req.top_k_final is not None else {}
    diagnosis["gold_document_rank"] = min(matching_ranks) if matching_ranks else None
    diagnosis["question_type"] = req.question_type
    expected_numbers = _normalized_numeric_values(req.expected_answer)
    answer_numbers = _normalized_numeric_values(run.get("llm_answer"))
    context_numbers = _normalized_numeric_values("\n".join(
        str(chunk.get("text", "")) for chunk in run.get("retrieved_chunks", []) if isinstance(chunk, dict)
    ))
    if (
        req.question_type == "metrics-generated"
        and len(set(expected_numbers)) == 1
        and expected_numbers[0] in answer_numbers
        and expected_numbers[0] in context_numbers
        and (not expected_doc or diagnosis["gold_document_retrieved"] is True)
    ):
        diagnosis.update(
            root_cause=FailureLabel.NO_FAILURE.value,
            status="reference_supported",
            finding="REFERENCE_SUPPORTED_NUMERIC",
            finding_label="Labeled numeric fact supported",
            evidence_level="strong",
            explanation="The labeled numeric value appears in both the answer and retrieved context, and the expected source document was retrieved. This supports the requested numeric fact; it does not validate unrelated claims or unit formatting.",
            recommended_fix="No RAG parameter change is indicated for this labeled numeric fact. Review units and any additional claims separately.",
        )
        diagnosis["evidence"].append("Normalized expected numeric value matched the response and retrieved passage text.")
    arithmetic_checks = audit_arithmetic(run.get("llm_answer", ""))
    if arithmetic_checks:
        diagnosis.update(
            root_cause=FailureLabel.GENERATOR.value,
            status="diagnosed",
            finding="ARITHMETIC_MISMATCH",
            finding_label="Response arithmetic failed a deterministic check",
            evidence_level="strong",
            explanation="The response contains an explicit numerical equation whose operands do not produce the displayed result. This is direct evidence of an answer-generation/calculation error.",
            recommended_fix="Move this calculation to deterministic application code or correct the formula, then rerun the same labeled case and verify each intermediate value.",
            arithmetic_checks=arithmetic_checks,
        )
        diagnosis["evidence"].extend(
            f"Arithmetic mismatch: {item['expression']} computes to {item['computed']}, but the response states {item['reported']}."
            for item in arithmetic_checks
        )
    # Build references only after the evaluator has applied its final status and
    # root-cause overrides, so a passing case cannot inherit stale tuning advice.
    diagnosis["code_references"] = _code_references_for_diagnosis(workspace, diagnosis)
    return {
        "case_id": req.case_id,
        "diagnosis": diagnosis,
        "captured_attribution": "latest_local_runtime_capture",
    }


@app.post("/api/agentic-investigate")
def agentic_investigate(req: AgentInvestigationRequest) -> Dict[str, Any]:
    """Use a configured loopback model to inspect one captured case and local code."""
    workspace = _resolve_workspace(req.workspace_path)
    case = {
        "case_id": req.case_id,
        "question": SecurityGuard.sanitize_text(req.question[:10000]),
        "expected_answer": SecurityGuard.sanitize_text(req.expected_answer[:5000]),
        "gold_document_id": req.gold_document_id,
        "question_type": req.question_type,
        "diagnosis": req.diagnosis,
    }
    try:
        result = investigate_case(
            workspace,
            case,
            req.local_model_endpoint,
            req.local_model_name,
            thinking=req.local_model_thinking,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"Local investigation failed: {str(exc)[:300]}") from exc
    return {
        "case_id": req.case_id,
        "investigation": result,
        "workspace_root": str(workspace),
        "storage": "VOLATILE_RAM_ONLY",
        "network_policy": "LOOPBACK_MODEL_ONLY",
    }


@app.post("/api/runtime-agent-reset")
def reset_runtime_agent(req: RuntimeAgentRequest) -> Dict[str, Any]:
    workspace = _resolve_workspace(req.workspace_path)
    RUNTIME_AGENT_READY.pop(str(workspace), None)
    VOLATILE_EXTERNAL_RUNS.pop(str(workspace), None)
    return {"status": "reset", "workspace_root": str(workspace)}


@app.post("/api/captured-run-reset")
def reset_captured_run(req: RuntimeAgentRequest) -> Dict[str, Any]:
    """Clear one workspace's last run before replaying the next labeled case."""
    workspace = _resolve_workspace(req.workspace_path)
    VOLATILE_EXTERNAL_RUNS.pop(str(workspace), None)
    return {"status": "reset", "workspace_root": str(workspace)}


@app.post("/api/runtime-agent-ready")
def runtime_agent_ready(req: RuntimeAgentRequest) -> Dict[str, Any]:
    workspace = _resolve_workspace(req.workspace_path)
    RUNTIME_AGENT_READY[str(workspace)] = {
        "process_id": req.process_id,
        "providers": req.providers,
        "observer_installed": req.observer_installed,
        "detail": req.detail,
        "status": "ready" if req.observer_installed else "unsupported",
    }
    return {"status": RUNTIME_AGENT_READY[str(workspace)]["status"]}


@app.get("/api/runtime-agent-status")
def runtime_agent_status(workspace_path: str = Query(...)) -> Dict[str, Any]:
    workspace = _resolve_workspace(workspace_path)
    return RUNTIME_AGENT_READY.get(str(workspace), {"status": "not_ready"})


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "storage": "VOLATILE_RAM_ONLY"}


@app.get("/api/analyze-workspace")
def analyze_workspace(workspace_path: str = Query(...), include_content: bool = Query(default=False)) -> Dict[str, Any]:
    workspace = _resolve_workspace(workspace_path)
    ast_report = AST_INSPECTOR.scan_workspace(str(workspace))
    log_report = LOG_INSPECTOR.scan(workspace, include_content=include_content)
    latest_run = VOLATILE_EXTERNAL_RUNS.get(str(workspace))

    # An in-memory SDK capture takes precedence over a parsed log event.
    run = latest_run
    source = "live_capture" if latest_run else None
    if run is None and log_report.get("latest_event"):
        event = log_report["latest_event"]
        run = {
            "question": event.get("question", ""),
            "prompt": event.get("prompt", ""),
            "llm_answer": event.get("answer", ""),
            "retrieved_chunks": event.get("chunks", []),
            "retrieval_observed": event.get("retrieval_observed", "chunks" in event),
            "retrieval_count": event.get("retrieval_count", len(event.get("chunks", []))),
            "expected_answer": event.get("expected"),
            "oracle_passed": event.get("oracle"),
            "max_index_gold_coverage": event.get("coverage"),
            "top_k_final": event.get("top_k", event.get("topk")),
            "metadata": {
                key: event[key]
                for key in ("trace_id", "operation", "provider", "model", "input_tokens", "output_tokens", "latency_ms", "error", "span_count", "collection")
                if event.get(key) is not None
            },
            "content_excluded": event.get("content_excluded", False),
            "workspace_path": str(workspace),
        }
        source = "local_log"

    diagnosis = diagnose_live_run(run, ast_report["static_warnings"]) if run else None
    if diagnosis:
        diagnosis["source"] = source
        diagnosis["data_transfer_events"] = run.get("data_transfer_events", [])
        if not diagnosis.get("source_location"):
            component = {
                "CHUNKING": "chunking",
                "RETRIEVAL": "retrieval",
                "RANKING_K": "retrieval",
                "RERANKER": "reranking",
                "CONTEXT_NOISE": "prompt",
                "GENERATOR": "generation",
                "ABSTENTION": "generation",
            }.get(diagnosis.get("root_cause"))
            location = next((item for item in ast_report.get("pipeline_locations", []) if item.get("component") == component), None)
            if location:
                diagnosis["source_location"] = {"file": location["file"], "line": location["line"]}
                diagnosis["source_location_note"] = f"Related {component} call site found by static pattern matching; review this location."
        if source == "local_log":
            diagnosis["log_location"] = {"file": log_report["latest_event"].get("file"), "line": log_report["latest_event"].get("line")}

    # Scrub code excerpts and filesystem path fields before returning anything to the webview.
    for finding in ast_report.get("code_locations", []):
        finding["code_snippet"] = SecurityGuard.sanitize_text(str(finding.get("code_snippet", "")))
    for warning in ast_report.get("static_warnings", []):
        for key in ("code_snippet", "problem", "fix_suggestion"):
            if key in warning:
                warning[key] = SecurityGuard.sanitize_text(str(warning[key]))

    return {
        "workspace_root": str(workspace),
        "storage": "VOLATILE_RAM_ONLY",
        "security_audit": {
            "workspace_jail": "ENFORCED",
            "pii_redaction": "ACTIVE",
            "rag_doctor_network_uploads": False,
        },
        "code_scan": ast_report,
        "log_scan": log_report,
        "diagnosis": diagnosis,
        "message": "No supported RAG run was found. Add a JSON/JSONL log event or the local probe hook to diagnose prompt, retrieved context, and response.",
    }


@app.get("/api/inspect-workspace")
def inspect_workspace_code_and_last_run(workspace_path: str = Query(default=".")) -> Dict[str, Any]:
    workspace = _resolve_workspace(workspace_path)
    captured = VOLATILE_EXTERNAL_RUNS.get(str(workspace))
    if captured:
        diagnosis = diagnose_live_run(captured)
        return {
            "storage_mode": "VOLATILE_RAM_ONLY",
            "security_audit": captured["security_audit"],
            "ast_code_scan": AST_INSPECTOR.scan_workspace(str(workspace)),
            "last_response_diagnosis": {
                "question": diagnosis["question"],
                "llm_answer": diagnosis["llm_answer"],
                "gold_expected": diagnosis["expected_answer"] or "(No reference supplied)",
                "root_cause": diagnosis["root_cause"] or "INSUFFICIENT_EVIDENCE",
                "explanation": diagnosis["explanation"],
                "recommended_fix": diagnosis["recommended_fix"],
                "oracle_passed": captured.get("oracle_passed"),
                "max_index_gold_coverage": captured.get("max_index_gold_coverage"),
                "candidates": captured["retrieved_chunks"],
            },
        }

    ast_report = AST_INSPECTOR.scan_workspace(str(workspace))
    knobs = ast_report["detected_knobs"]

    raw_cs = int(knobs.get("chunk_size", {}).get("value", 64))
    c_size = max(32, min(4096, raw_cs))
    raw_ov = int(knobs.get("chunk_overlap", {}).get("value", 0))
    c_overlap = max(0, min(c_size - 1, raw_ov))
    raw_tk = int(knobs.get("top_k_final", {}).get("value", 1))
    top_k = max(1, min(15, raw_tk))

    cfg = PipelineConfig(
        chunk_strategy=ChunkStrategy.FIXED if c_size < 100 else ChunkStrategy.RECURSIVE,
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        retriever_type=RetrieverType.DENSE,
        top_n_candidates=max(top_k, 5),
        top_k_final=top_k,
    )

    stateless_cache = DiskCache(enabled=False)
    runner = RAGPipelineRunner(DOCUMENTS, cfg, cache=stateless_cache)
    target_bm = BENCHMARKS[1]
    trace, diagnosis = runner.run_single(target_bm.question, target_bm)

    sanitized_answer = SecurityGuard.sanitize_text(trace.llm_answer)
    injection_check = SecurityGuard.scan_for_prompt_injection(trace.assembled_prompt)

    return {
        "storage_mode": "ZERO_STORAGE_VOLATILE_RAM",
        "security_audit": {
            "pii_redaction": "ACTIVE",
            "prompt_injection_detected": injection_check["injection_detected"],
            "workspace_jail": "ENFORCED",
        },
        "ast_code_scan": ast_report,
        "last_response_diagnosis": {
            "question": trace.question,
            "llm_answer": sanitized_answer,
            "gold_expected": target_bm.gold_answer,
            "is_correct": trace.is_answer_correct,
            "root_cause": diagnosis.primary_root_cause.value,
            "explanation": diagnosis.explanation,
            "recommended_fix": diagnosis.recommended_fix,
            "oracle_passed": trace.oracle_passed,
            "max_index_gold_coverage": trace.max_index_gold_coverage,
            "candidates": [c.model_dump() for c in trace.candidates[:4]],
        },
    }


@app.post("/api/diagnose")
def run_diagnostic_suite(config: PipelineConfig) -> Dict[str, Any]:
    stateless_cache = DiskCache(enabled=False)
    runner = RAGPipelineRunner(DOCUMENTS, config, cache=stateless_cache)
    results = runner.run_benchmark_suite(BENCHMARKS)

    traces = [t for t, _ in results]
    diagnoses = [d for _, d in results if d is not None]
    labels = [d.primary_root_cause for d in diagnoses]

    total = max(1, len(traces))
    correct = sum(1 for t in traces if t.is_answer_correct)
    recall_k = sum(
        1
        for t in traces
        if any(c.gold_coverage_ratio >= 0.75 for c in t.get_final_context_chunks())
    )
    avg_lat = round(sum(t.total_latency_ms for t in traces) / total, 2)
    avg_tok = round(sum(t.prompt_tokens for t in traces) / total, 1)

    cause_counts: Dict[str, int] = {}
    for d in diagnoses:
        k = d.primary_root_cause.value
        cause_counts[k] = cause_counts.get(k, 0) + 1

    feature_importances = []
    if len(set(labels)) >= 2:
        ml_diag = MLTraceDiagnoser()
        ml_diag.fit_and_evaluate(traces, labels)
        feature_importances = [
            {"feature": name, "importance": imp}
            for name, imp in ml_diag.get_feature_importances(traces, labels)[:6]
        ]

    clusterer = FailureClusterer(max_clusters=3)
    cluster_report = clusterer.discover_themes(traces, diagnoses)
    ast_report = AST_INSPECTOR.scan_workspace(".")

    return {
        "storage_mode": "ZERO_STORAGE_VOLATILE_RAM",
        "config": config.model_dump(),
        "config_hash": config.config_hash,
        "ast_code_scan": ast_report,
        "summary": {
            "total_questions": total,
            "accuracy": round(correct / total, 4),
            "recall_at_k": round(recall_k / total, 4),
            "avg_latency_ms": avg_lat,
            "avg_prompt_tokens": avg_tok,
            "failure_breakdown": cause_counts,
        },
        "feature_importances": feature_importances,
        "clustering": cluster_report,
        "runs": [
            {
                "trace": t.model_dump(),
                "diagnosis": d.model_dump() if d else None,
            }
            for t, d in results
        ],
    }


@app.post("/api/autotune")
def run_pareto_autotune(baseline_config: PipelineConfig) -> Dict[str, Any]:
    tuner = ParetoAutoTuner(DOCUMENTS, BENCHMARKS)
    return tuner.run_auto_tune(baseline_config, n_trials=6)
