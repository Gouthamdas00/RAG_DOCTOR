from backend.app.diagnose.clustering import FailureClusterer
from backend.app.diagnose.ml_diagnoser import MLTraceDiagnoser, extract_trace_features
from backend.app.schemas.contracts import (
    CharSpan,
    DiagnosisResult,
    FailureLabel,
    PipelineConfig,
    PipelineRunTrace,
    ScoredChunkTrace,
)


def make_synthetic_trace(
    idx: int,
    question: str,
    chunk_size: int,
    dense_top1: float,
    dense_top2: float,
    abstained: bool,
) -> PipelineRunTrace:
    c1 = ScoredChunkTrace(
        chunk_id=f"c1_{idx}",
        doc_id="d1",
        span=CharSpan(start_char=0, end_char=chunk_size),
        text=question if dense_top1 > 0.6 else "unrelated text block",
        dense_score=dense_top1,
        dense_rank=1,
        bm25_score=dense_top1 * 10,
        bm25_rank=1,
        pre_rerank_rank=1,
        rerank_score=dense_top1,
        final_rank=1,
    )
    c2 = ScoredChunkTrace(
        chunk_id=f"c2_{idx}",
        doc_id="d1",
        span=CharSpan(start_char=100, end_char=100 + chunk_size),
        text="secondary chunk text",
        dense_score=dense_top2,
        dense_rank=2,
        bm25_score=dense_top2 * 10,
        bm25_rank=2,
        pre_rerank_rank=2,
        rerank_score=dense_top2,
        final_rank=2,
    )
    return PipelineRunTrace(
        trace_id=f"tr_{idx}",
        question_id=f"q_{idx}",
        question=question,
        config=PipelineConfig(chunk_size=chunk_size, top_k_final=2),
        candidates=[c1, c2],
        retrieval_latency_ms=10.0,
        rerank_latency_ms=5.0,
        assembled_prompt="Prompt...",
        prompt_tokens=120,
        completion_tokens=20,
        llm_answer="INSUFFICIENT_CONTEXT" if abstained else question,
        abstained=abstained,
        generation_latency_ms=80.0,
    )


def test_ml_diagnoser_and_clustering():
    traces = []
    labels = []
    diagnoses = []

    # 1. Create 10 CHUNKING failure traces (tiny chunk_size=64, low scores)
    for i in range(10):
        q = f"What is the database failover timeout SLA for region {i}?"
        t = make_synthetic_trace(i, q, chunk_size=64, dense_top1=0.35, dense_top2=0.33, abstained=False)
        traces.append(t)
        labels.append(FailureLabel.CHUNKING)
        diagnoses.append(
            DiagnosisResult(
                trace_id=t.trace_id,
                question_id=t.question_id,
                primary_root_cause=FailureLabel.CHUNKING,
                explanation="Chunking split",
                recommended_fix="Increase chunk_size",
            )
        )

    # 2. Create 10 RETRIEVAL failure traces (normal chunk_size=512, very low dense scores + abstained)
    for i in range(10, 20):
        q = f"How do I configure mTLS certificate rotation in ingress gateway {i}?"
        t = make_synthetic_trace(i, q, chunk_size=512, dense_top1=0.12, dense_top2=0.11, abstained=True)
        traces.append(t)
        labels.append(FailureLabel.RETRIEVAL)
        diagnoses.append(
            DiagnosisResult(
                trace_id=t.trace_id,
                question_id=t.question_id,
                primary_root_cause=FailureLabel.RETRIEVAL,
                explanation="Retrieval missed",
                recommended_fix="Use hybrid search",
            )
        )

    # 3. Create 10 NO_FAILURE traces (normal chunk_size=512, high dense margin, not abstained)
    for i in range(20, 30):
        q = f"What is the default worker connection limit on node {i}?"
        t = make_synthetic_trace(i, q, chunk_size=512, dense_top1=0.92, dense_top2=0.30, abstained=False)
        traces.append(t)
        labels.append(FailureLabel.NO_FAILURE)
        diagnoses.append(
            DiagnosisResult(
                trace_id=t.trace_id,
                question_id=t.question_id,
                primary_root_cause=FailureLabel.NO_FAILURE,
                explanation="Healthy run",
                recommended_fix="None",
            )
        )

    # Test Reference-Free Feature Extractor
    feats = extract_trace_features(traces[0])
    assert len(feats) == 14

    # Train & Evaluate ML Diagnoser
    ml_diag = MLTraceDiagnoser()
    metrics = ml_diag.fit_and_evaluate(traces, labels)
    assert metrics["rf_macro_f1"] >= 0.90

    # Predict on a live unlabelled trace (no gold_span!)
    pred = ml_diag.predict_diagnosis(traces[0])
    assert pred.primary_root_cause == FailureLabel.CHUNKING
    assert FailureLabel.CHUNKING in pred.failure_probabilities

    # Check Permutation Feature Importance
    importances = ml_diag.get_feature_importances(traces, labels)
    assert len(importances) == 14

    # Test Failure-Theme Clustering on the 20 failed traces
    clusterer = FailureClusterer(max_clusters=4)
    theme_report = clusterer.discover_themes(traces, diagnoses)
    assert theme_report["num_failed"] == 20
    assert theme_report["optimal_k"] >= 2
    assert len(theme_report["clusters"]) >= 2
    assert len(theme_report["points_2d"]) == 20


if __name__ == "__main__":
    test_ml_diagnoser_and_clustering()
    print("All ML root-cause classifier and unsupervised failure-theme clustering tests passed!")
