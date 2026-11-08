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


def build_test_corpus_and_benchmark():
    text = (
        "Chapter 1: Cluster Networking.\n"
        "Internal service mesh routing uses mTLS with rotation every 24 hours.\n\n"
        "Chapter 2: Database Failover SLA.\n"
        "In the event of a primary node outage, automated replica promotion completes within 45 seconds.\n\n"
        "Chapter 3: Log Archival.\n"
        "Audit logs are compressed with zstd and uploaded to glacier vaults weekly.\n\n"
        "Chapter 4: Pod Eviction Thresholds.\n"
        "Kubelet evicts burstable pods when node memory drops below 150Mi.\n\n"
        "Chapter 5: Ingress Controller Tuning.\n"
        "Worker connections default to 10240 per node.\n\n"
        "Chapter 6: Secret Encryption.\n"
        "Envelope encryption uses AES-256-GCM keys managed by KMS."
    )
    doc = Document(doc_id="k8s_ops", title="Kubernetes Operations Guide", text=text)

    gold_sentence = "In the event of a primary node outage, automated replica promotion completes within 45 seconds."
    start = text.index(gold_sentence)
    end = start + len(gold_sentence)

    bm = BenchmarkItem(
        question_id="q_failover",
        question="Within how many seconds does automated replica promotion complete during a primary node outage?",
        gold_answer="45 seconds",
        gold_doc_id="k8s_ops",
        gold_span=CharSpan(start_char=start, end_char=end),
    )
    return [doc], bm


def test_end_to_end_runner_diagnoses():
    docs, bm = build_test_corpus_and_benchmark()

    # 1. Healthy Config (chunk_size=200 keeps Chapter 2 intact) -> Should pass with NO_FAILURE
    healthy_cfg = PipelineConfig(
        chunk_strategy=ChunkStrategy.RECURSIVE,
        chunk_size=200,
        chunk_overlap=30,
        retriever_type=RetrieverType.HYBRID_RRF,
        top_n_candidates=6,
        top_k_final=2,
    )
    runner_ok = RAGPipelineRunner(docs, healthy_cfg)
    trace_ok, diag_ok = runner_ok.run_single(bm.question, bm)
    assert trace_ok.is_answer_correct is True
    assert diag_ok.primary_root_cause == FailureLabel.NO_FAILURE

    # 2. Weak Generator Config -> Gold chunk is #1, but weak model fails even the Oracle test -> GENERATOR failure
    weak_gen_cfg = PipelineConfig(
        chunk_strategy=ChunkStrategy.RECURSIVE,
        chunk_size=200,
        chunk_overlap=30,
        top_n_candidates=6,
        top_k_final=2,
        llm_model="weak-tiny-llm",
    )
    runner_weak = RAGPipelineRunner(docs, weak_gen_cfg)
    trace_weak, diag_weak = runner_weak.run_single(bm.question, bm)
    assert trace_weak.is_answer_correct is False
    assert trace_weak.oracle_passed is False
    assert diag_weak.primary_root_cause == FailureLabel.GENERATOR


if __name__ == "__main__":
    test_end_to_end_runner_diagnoses()
    print("All end-to-end RAG runner and Oracle counterfactual tests passed!")
