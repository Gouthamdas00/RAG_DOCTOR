from backend.app.schemas.contracts import (
    BenchmarkItem,
    CharSpan,
    ChunkStrategy,
    Document,
    PipelineConfig,
    RetrieverType,
)
from backend.app.tuner.optimizer import ParetoAutoTuner


def build_tuning_benchmark():
    text = (
        "Section 1: Authentication Tokens.\n"
        "Access tokens expire every 15 minutes and require refresh token rotation.\n\n"
        "Section 2: Database Failover SLA.\n"
        "In the event of a primary node outage, automated replica promotion completes within 45 seconds.\n\n"
        "Section 3: Webhook Retry Policy.\n"
        "Failed webhook deliveries are retried up to 8 times using exponential backoff."
    )
    doc = Document(doc_id="platform_sla", title="Platform SLA Spec", text=text)

    q1_sent = "Access tokens expire every 15 minutes and require refresh token rotation."
    q2_sent = "In the event of a primary node outage, automated replica promotion completes within 45 seconds."
    q3_sent = "Failed webhook deliveries are retried up to 8 times using exponential backoff."

    benchmarks = [
        BenchmarkItem(
            question_id="q1",
            question="After how many minutes do access tokens expire?",
            gold_answer="15 minutes",
            gold_doc_id="platform_sla",
            gold_span=CharSpan(start_char=text.index(q1_sent), end_char=text.index(q1_sent) + len(q1_sent)),
        ),
        BenchmarkItem(
            question_id="q2",
            question="Within how many seconds does automated replica promotion complete during a primary node outage?",
            gold_answer="45 seconds",
            gold_doc_id="platform_sla",
            gold_span=CharSpan(start_char=text.index(q2_sent), end_char=text.index(q2_sent) + len(q2_sent)),
        ),
        BenchmarkItem(
            question_id="q3",
            question="Up to how many times are failed webhook deliveries retried?",
            gold_answer="8 times",
            gold_doc_id="platform_sla",
            gold_span=CharSpan(start_char=text.index(q3_sent), end_char=text.index(q3_sent) + len(q3_sent)),
        ),
    ]
    return [doc], benchmarks


def test_pareto_auto_tuner_heals_broken_pipeline():
    docs, benchmarks = build_tuning_benchmark()

    # Start with a deliberately broken baseline config (tiny 64-char fixed chunks, no overlap)
    broken_baseline = PipelineConfig(
        chunk_strategy=ChunkStrategy.FIXED,
        chunk_size=64,
        chunk_overlap=0,
        retriever_type=RetrieverType.DENSE,
        top_n_candidates=3,
        top_k_final=1,
        reranker_model=None,
    )

    tuner = ParetoAutoTuner(docs, benchmarks)
    report = tuner.run_auto_tune(broken_baseline, n_trials=6)

    # Verify the broken baseline performed poorly, and the auto-tuner found a 100% accurate config!
    assert report["baseline"]["accuracy"] < 1.0
    assert report["best_recommended"]["accuracy"] == 1.0
    assert report["accuracy_lift"] > 0.0
    assert len(report["pareto_frontier"]) >= 1

    # Verify Regression Surrogates were fitted and can predict on unseen configs
    pred_acc, pred_tokens = tuner.surrogates.predict(broken_baseline)
    assert 0.0 <= pred_acc <= 1.0
    assert pred_tokens > 0.0


if __name__ == "__main__":
    test_pareto_auto_tuner_heals_broken_pipeline()
    print("All regression surrogate and Pareto auto-tuner tests passed!")
