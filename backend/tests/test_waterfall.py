import os
from backend.app.core.cache import DiskCache
from backend.app.diagnose.waterfall import WaterfallDiagnoser
from backend.app.schemas.contracts import (
    BenchmarkItem,
    CharSpan,
    FailureLabel,
    PipelineConfig,
    PipelineRunTrace,
    ScoredChunkTrace,
)


def make_candidate(pre_rank: int, final_rank: int, coverage: float) -> ScoredChunkTrace:
    return ScoredChunkTrace(
        chunk_id=f"c_{pre_rank}",
        doc_id="doc1",
        span=CharSpan(start_char=0, end_char=100),
        text="Sample chunk text",
        pre_rerank_rank=pre_rank,
        final_rank=final_rank,
        gold_coverage_ratio=coverage,
    )


def make_trace(**overrides) -> PipelineRunTrace:
    defaults = dict(
        trace_id="t1",
        question_id="q1",
        question="What is the API rate limit?",
        config=PipelineConfig(top_n_candidates=20, top_k_final=5),
        max_index_gold_coverage=1.0,
        candidates=[make_candidate(pre_rank=1, final_rank=1, coverage=1.0)],
        retrieval_latency_ms=12.0,
        rerank_latency_ms=18.0,
        assembled_prompt="Context: ... Question: ...",
        prompt_tokens=150,
        completion_tokens=20,
        llm_answer="100 requests per minute",
        abstained=False,
        generation_latency_ms=220.0,
        is_answer_correct=False,
        oracle_passed=None,
    )
    defaults.update(overrides)
    return PipelineRunTrace(**defaults)


def test_all_waterfall_diagnoses():
    diagnoser = WaterfallDiagnoser(min_gold_coverage=0.75)

    # 1. CHUNKING Failure: Max coverage in the entire index is only 40%
    t_chunk = make_trace(max_index_gold_coverage=0.40)
    res = diagnoser.diagnose(t_chunk)
    assert res.primary_root_cause == FailureLabel.CHUNKING

    # 2. RETRIEVAL Failure: Index has intact chunk (1.0), but candidates have 0.0 coverage
    t_ret = make_trace(
        max_index_gold_coverage=1.0,
        candidates=[make_candidate(pre_rank=1, final_rank=1, coverage=0.0)],
    )
    res = diagnoser.diagnose(t_ret)
    assert res.primary_root_cause == FailureLabel.RETRIEVAL

    # 3. RERANKER Failure: Gold chunk was rank #2 pre-rerank, demoted to #8 (top_k_final=5)
    t_rerank = make_trace(
        candidates=[make_candidate(pre_rank=2, final_rank=8, coverage=1.0)]
    )
    res = diagnoser.diagnose(t_rerank)
    assert res.primary_root_cause == FailureLabel.RERANKER

    # 4. RANKING_K Failure: Gold chunk was rank #12 pre-rerank and #7 post-rerank (never in top-5)
    t_rank_k = make_trace(
        candidates=[make_candidate(pre_rank=12, final_rank=7, coverage=1.0)]
    )
    res = diagnoser.diagnose(t_rank_k)
    assert res.primary_root_cause == FailureLabel.RANKING_K

    # 5. CONTEXT_NOISE Failure: Gold chunk in prompt (rank #3 <= 5), wrong answer, but Oracle PASSES
    t_noise = make_trace(
        candidates=[make_candidate(pre_rank=4, final_rank=3, coverage=1.0)],
        oracle_passed=True,
    )
    res = diagnoser.diagnose(t_noise)
    assert res.primary_root_cause == FailureLabel.CONTEXT_NOISE

    # 6. GENERATOR Failure: Gold chunk in prompt (rank #1 <= 5), wrong answer, and Oracle FAILS
    t_gen = make_trace(
        candidates=[make_candidate(pre_rank=1, final_rank=1, coverage=1.0)],
        oracle_passed=False,
    )
    res = diagnoser.diagnose(t_gen)
    assert res.primary_root_cause == FailureLabel.GENERATOR

    # 7. ABSTENTION Failure: Unanswerable benchmark question, but LLM did not abstain
    unanswerable_bm = BenchmarkItem(
        question_id="q_unans",
        question="What is the CEO's favorite color?",
        gold_answer="UNANSWERABLE",
        gold_doc_id="doc1",
        gold_span=CharSpan(start_char=0, end_char=0),
        question_type="unanswerable",
    )
    t_abst = make_trace(abstained=False)
    res = diagnoser.diagnose(t_abst, benchmark=unanswerable_bm)
    assert res.primary_root_cause == FailureLabel.ABSTENTION


def test_disk_cache():
    test_db = "data/test_cache.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    cache = DiskCache(db_path=test_db)
    call_count = 0

    def expensive_llm_call():
        nonlocal call_count
        call_count += 1
        return {"answer": "42", "tokens": 12}

    payload = {"model": "gemini-2.5-flash", "prompt": "What is 6x7?"}
    res1 = cache.get_or_compute("llm", payload, expensive_llm_call)
    res2 = cache.get_or_compute("llm", payload, expensive_llm_call)

    assert res1 == res2
    assert call_count == 1  # Second call was served instantly from SQLite!
    os.remove(test_db)


if __name__ == "__main__":
    test_all_waterfall_diagnoses()
    test_disk_cache()
    print("All waterfall diagnosis and cache tests passed!")