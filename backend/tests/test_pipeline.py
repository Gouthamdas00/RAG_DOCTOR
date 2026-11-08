from backend.app.diagnose.waterfall import WaterfallDiagnoser
from backend.app.pipeline.chunker import DocumentChunker
from backend.app.pipeline.retriever import HybridRetrieverEngine
from backend.app.schemas.contracts import (
    BenchmarkItem,
    CharSpan,
    ChunkStrategy,
    Document,
    FailureLabel,
    PipelineConfig,
    PipelineRunTrace,
    RetrieverType,
)


def build_sample_corpus():
    doc_text = (
        "Section 1: Overview of Cloud Gateway.\n"
        "The Cloud Gateway manages incoming traffic across regional clusters.\n\n"
        "Section 2: Rate Limiting and Quotas.\n"
        "Standard tier accounts are limited to 1200 requests per minute across all endpoints. "
        "Burst traffic exceeding this threshold receives HTTP 429 Too Many Requests.\n\n"
        "Section 3: Backup Retention Policy.\n"
        "Automated snapshots are retained for 35 days in cold object storage."
    )
    doc = Document(doc_id="doc_gateway", title="Gateway Manual", text=doc_text)

    target_sentence = "Standard tier accounts are limited to 1200 requests per minute across all endpoints."
    start_idx = doc_text.index(target_sentence)
    end_idx = start_idx + len(target_sentence)

    benchmark = BenchmarkItem(
        question_id="q_rate_limit",
        question="How many requests per minute are allowed for standard tier accounts?",
        gold_answer="1200 requests per minute",
        gold_doc_id="doc_gateway",
        gold_span=CharSpan(start_char=start_idx, end_char=end_idx),
    )
    return [doc], benchmark


def test_healthy_hybrid_retrieval():
    docs, bm = build_sample_corpus()
    cfg = PipelineConfig(
        chunk_strategy=ChunkStrategy.RECURSIVE,
        chunk_size=180,
        chunk_overlap=30,
        retriever_type=RetrieverType.HYBRID_RRF,
        top_n_candidates=5,
        top_k_final=2,
    )

    chunker = DocumentChunker()
    chunks = chunker.chunk_corpus(docs, cfg)
    max_cov = chunker.compute_max_gold_coverage(chunks, bm)

    assert max_cov == 1.0

    for c in chunks:
        assert docs[0].text[c.span.start_char : c.span.end_char] == c.text

    retriever = HybridRetrieverEngine(chunks, cfg, use_neural_models=False)
    candidates, ret_ms, rerank_ms = retriever.retrieve_and_rerank(bm.question, bm)

    assert candidates[0].gold_coverage_ratio == 1.0
    assert candidates[0].final_rank == 1


def test_real_chunking_fault_injection():
    docs, bm = build_sample_corpus()
    bad_cfg = PipelineConfig(
        chunk_strategy=ChunkStrategy.FIXED,
        chunk_size=64,
        chunk_overlap=0,
        top_n_candidates=5,
        top_k_final=2,
    )

    chunker = DocumentChunker()
    chunks = chunker.chunk_corpus(docs, bad_cfg)
    max_cov = chunker.compute_max_gold_coverage(chunks, bm)

    assert max_cov < 0.80

    retriever = HybridRetrieverEngine(chunks, bad_cfg, use_neural_models=False)
    candidates, ret_ms, rerank_ms = retriever.retrieve_and_rerank(bm.question, bm)

    trace = PipelineRunTrace(
        trace_id="trace_broken_chunker",
        question_id=bm.question_id,
        question=bm.question,
        config=bad_cfg,
        max_index_gold_coverage=max_cov,
        candidates=candidates,
        retrieval_latency_ms=ret_ms,
        rerank_latency_ms=rerank_ms,
        assembled_prompt="Prompt...",
        prompt_tokens=100,
        completion_tokens=15,
        llm_answer="I am not sure of the exact limit.",
        generation_latency_ms=145.0,
        is_answer_correct=False,
    )

    diagnoser = WaterfallDiagnoser(min_gold_coverage=0.80)
    diagnosis = diagnoser.diagnose(trace, bm)
    assert diagnosis.primary_root_cause == FailureLabel.CHUNKING


if __name__ == "__main__":
    test_healthy_hybrid_retrieval()
    test_real_chunking_fault_injection()
    print("All chunking, hybrid RRF retrieval, and fault-injection tests passed!")
