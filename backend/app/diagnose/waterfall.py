from typing import Optional
from backend.app.schemas.contracts import (
    BenchmarkItem,
    DiagnosisResult,
    FailureLabel,
    PipelineRunTrace,
    ScoredChunkTrace,
)


class WaterfallDiagnoser:
    """
    Deterministic root-cause diagnoser + Oracle counterfactual analyzer.
    """

    def __init__(self, min_gold_coverage: float = 0.75):
        self.min_gold_coverage = min_gold_coverage

    def _find_best_candidate(
        self, trace: PipelineRunTrace
    ) -> Optional[ScoredChunkTrace]:
        if not trace.candidates:
            return None
        best = max(trace.candidates, key=lambda c: c.gold_coverage_ratio)
        if best.gold_coverage_ratio >= self.min_gold_coverage:
            return best
        return None

    def diagnose(
        self, trace: PipelineRunTrace, benchmark: Optional[BenchmarkItem] = None
    ) -> DiagnosisResult:
        cfg = trace.config

        # 1. Check for Unanswerable / Abstention Failures First
        if benchmark and benchmark.question_type == "unanswerable":
            if trace.abstained:
                return DiagnosisResult(
                    trace_id=trace.trace_id,
                    question_id=trace.question_id,
                    primary_root_cause=FailureLabel.NO_FAILURE,
                    failure_probabilities={FailureLabel.NO_FAILURE: 1.0},
                    explanation="Question was unanswerable and the pipeline properly abstained.",
                    recommended_fix="None required.",
                )
            else:
                return DiagnosisResult(
                    trace_id=trace.trace_id,
                    question_id=trace.question_id,
                    primary_root_cause=FailureLabel.ABSTENTION,
                    failure_probabilities={FailureLabel.ABSTENTION: 1.0},
                    explanation="Question is unanswerable from the corpus, but the generator hallucinated an answer instead of abstaining.",
                    recommended_fix="Add explicit negative-constraint instructions in the system prompt or enforce a minimum retrieval/reranker score threshold.",
                )

        # 2. If Answer is Correct -> NO_FAILURE
        if trace.is_answer_correct is True:
            return DiagnosisResult(
                trace_id=trace.trace_id,
                question_id=trace.question_id,
                primary_root_cause=FailureLabel.NO_FAILURE,
                failure_probabilities={FailureLabel.NO_FAILURE: 1.0},
                explanation="Pipeline produced the correct answer.",
                recommended_fix="None required.",
            )

        # 3. Stage 1 Check: CHUNKING Failure
        if (
            trace.max_index_gold_coverage is not None
            and trace.max_index_gold_coverage < self.min_gold_coverage
        ):
            pct = int(trace.max_index_gold_coverage * 100)
            return DiagnosisResult(
                trace_id=trace.trace_id,
                question_id=trace.question_id,
                primary_root_cause=FailureLabel.CHUNKING,
                failure_probabilities={FailureLabel.CHUNKING: 1.0},
                explanation=(
                    f"The ground-truth answer span was fragmented across multiple chunks. "
                    f"The best chunk in the entire index only captured {pct}% of the answer "
                    f"(chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap})."
                ),
                recommended_fix=(
                    f"Increase chunk_size (currently {cfg.chunk_size}) or increase chunk_overlap "
                    f"(currently {cfg.chunk_overlap}), or switch chunk_strategy to 'recursive'/'semantic'."
                ),
            )

        # 4. Stage 2 Check: RETRIEVAL Failure
        gold_candidate = self._find_best_candidate(trace)
        if gold_candidate is None:
            return DiagnosisResult(
                trace_id=trace.trace_id,
                question_id=trace.question_id,
                primary_root_cause=FailureLabel.RETRIEVAL,
                failure_probabilities={FailureLabel.RETRIEVAL: 1.0},
                explanation=(
                    f"An intact gold chunk exists in the index, but the '{cfg.retriever_type.value}' "
                    f"retriever failed to surface it inside the top-{cfg.top_n_candidates} candidate pool."
                ),
                recommended_fix=(
                    f"Switch retriever_type to 'hybrid_rrf' (currently '{cfg.retriever_type.value}'), "
                    f"increase top_n_candidates (currently {cfg.top_n_candidates}), or upgrade the embedding model."
                ),
            )

        # 5. Stage 3 Check: RANKING_K vs. RERANKER Failure
        if gold_candidate.final_rank > cfg.top_k_final:
            if (
                cfg.reranker_model is not None
                and gold_candidate.pre_rerank_rank <= cfg.top_k_final
            ):
                return DiagnosisResult(
                    trace_id=trace.trace_id,
                    question_id=trace.question_id,
                    primary_root_cause=FailureLabel.RERANKER,
                    failure_probabilities={FailureLabel.RERANKER: 1.0},
                    explanation=(
                        f"Initial retrieval ranked the gold chunk at #{gold_candidate.pre_rerank_rank} "
                        f"(inside top-{cfg.top_k_final}), but reranker '{cfg.reranker_model}' demoted it "
                        f"to #{gold_candidate.final_rank} (rank_delta={gold_candidate.rank_delta})."
                    ),
                    recommended_fix=(
                        f"Disable or replace reranker '{cfg.reranker_model}', or increase top_k_final "
                        f"from {cfg.top_k_final} to {gold_candidate.final_rank}."
                    ),
                )
            else:
                return DiagnosisResult(
                    trace_id=trace.trace_id,
                    question_id=trace.question_id,
                    primary_root_cause=FailureLabel.RANKING_K,
                    failure_probabilities={FailureLabel.RANKING_K: 1.0},
                    explanation=(
                        f"The gold chunk was retrieved in the candidate pool (pre-rerank #{gold_candidate.pre_rerank_rank}, "
                        f"final rank #{gold_candidate.final_rank}), but missed the top_k_final={cfg.top_k_final} prompt cutoff."
                    ),
                    recommended_fix=(
                        f"Increase top_k_final (currently {cfg.top_k_final}) or add/upgrade a cross-encoder "
                        f"reranker to promote the candidate from #{gold_candidate.pre_rerank_rank} into the top-{cfg.top_k_final}."
                    ),
                )

        # 6. Stage 4 Check: CONTEXT_NOISE vs. GENERATOR (The Oracle Test)
        if trace.oracle_passed is True:
            return DiagnosisResult(
                trace_id=trace.trace_id,
                question_id=trace.question_id,
                primary_root_cause=FailureLabel.CONTEXT_NOISE,
                failure_probabilities={FailureLabel.CONTEXT_NOISE: 1.0},
                explanation=(
                    f"The gold chunk was present in the prompt at rank #{gold_candidate.final_rank}, and the "
                    f"Oracle Test PASSED when the LLM was given only the gold chunk. Distractor chunks or "
                    f"middle-of-prompt placement caused the generator to miss the answer."
                ),
                recommended_fix=(
                    f"Decrease top_k_final (currently {cfg.top_k_final}) to reduce irrelevant context noise, "
                    f"or reorder chunks so the highest-confidence chunks appear at the very beginning/end of the prompt."
                ),
            )
        else:
            return DiagnosisResult(
                trace_id=trace.trace_id,
                question_id=trace.question_id,
                primary_root_cause=FailureLabel.GENERATOR,
                failure_probabilities={FailureLabel.GENERATOR: 1.0},
                explanation=(
                    f"The gold chunk was present in the prompt at rank #{gold_candidate.final_rank}, and even "
                    f"in the isolated Oracle Test (with zero distractor chunks) the LLM failed to answer accurately."
                ),
                recommended_fix=(
                    f"Refine the system prompt instructions (e.g., chain-of-thought extraction for tables/numbers) "
                    f"or upgrade llm_model from '{cfg.llm_model}' to a stronger reasoning model."
                ),
            )
