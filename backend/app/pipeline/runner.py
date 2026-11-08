import uuid
from typing import Dict, List, Optional, Tuple

from backend.app.core.cache import DiskCache
from backend.app.diagnose.waterfall import WaterfallDiagnoser
from backend.app.pipeline.chunker import DocumentChunker
from backend.app.pipeline.generator import LLMGeneratorAndJudge
from backend.app.pipeline.retriever import HybridRetrieverEngine
from backend.app.schemas.contracts import (
    BenchmarkItem,
    DiagnosisResult,
    Document,
    PipelineConfig,
    PipelineRunTrace,
)


class RAGPipelineRunner:
    """
    End-to-End Traced RAG Orchestrator.
    Runs: Ingest/Chunk -> Hybrid Retrieve -> Rerank -> Prompt -> Generate
          -> Evaluate -> Counterfactual Oracle Probe -> Root-Cause Diagnosis.
    """

    def __init__(
        self,
        documents: List[Document],
        config: PipelineConfig,
        cache: Optional[DiskCache] = None,
        use_neural_models: bool = False,
        use_live_api: bool = False,
    ):
        self.documents = documents
        self.doc_map: Dict[str, Document] = {d.doc_id: d for d in documents}
        self.config = config
        self.cache = cache or DiskCache()

        self.chunker = DocumentChunker()
        self.chunks = self.chunker.chunk_corpus(documents, config)
        self.retriever = HybridRetrieverEngine(
            self.chunks, config, use_neural_models=use_neural_models
        )
        self.generator = LLMGeneratorAndJudge(
            cache=self.cache, use_live_api=use_live_api
        )
        self.diagnoser = WaterfallDiagnoser(min_gold_coverage=0.75)

    def run_single(
        self, question: str, benchmark: Optional[BenchmarkItem] = None
    ) -> Tuple[PipelineRunTrace, Optional[DiagnosisResult]]:
        """
        Executes a single question through the configured RAG pipeline, collects full
        stage-by-stage telemetry, runs the Oracle counterfactual if needed, and diagnoses the run.
        """
        q_id = benchmark.question_id if benchmark else f"q_{uuid.uuid4().hex[:8]}"
        trace_id = f"tr_{self.config.config_hash}_{q_id}"

        # 1. Check index-level chunk coverage
        max_cov = self.chunker.compute_max_gold_coverage(self.chunks, benchmark)

        # 2. Retrieve & Rerank
        candidates, ret_ms, rerank_ms = self.retriever.retrieve_and_rerank(
            question, benchmark
        )
        final_context = [
            c for c in candidates if c.final_rank <= self.config.top_k_final
        ]

        # 3. Generate Answer
        (
            prompt,
            llm_answer,
            prompt_tokens,
            completion_tokens,
            abstained,
            gen_ms,
        ) = self.generator.generate(question, final_context, self.config)

        # 4. Offline CI Evaluation & Smart Counterfactual Oracle Probe
        is_correct: Optional[bool] = None
        oracle_answer: Optional[str] = None
        oracle_passed: Optional[bool] = None

        if benchmark is not None:
            is_correct = self.generator.evaluate_correctness(llm_answer, benchmark)

            # Check if the gold chunk made it into the final prompt context
            gold_in_prompt = any(
                c.gold_coverage_ratio >= self.diagnoser.min_gold_coverage
                for c in final_context
            )

            # Only invoke the Oracle Probe when the answer failed DESPITE the gold chunk being in the prompt!
            if (
                not is_correct
                and gold_in_prompt
                and benchmark.question_type != "unanswerable"
            ):
                source_doc = self.doc_map.get(benchmark.gold_doc_id)
                if source_doc:
                    gold_text = source_doc.text[
                        benchmark.gold_span.start_char : benchmark.gold_span.end_char
                    ]
                    oracle_answer = self.generator.run_oracle_probe(
                        question, gold_text, self.config
                    )
                    oracle_passed = self.generator.evaluate_correctness(
                        oracle_answer, benchmark
                    )

        trace = PipelineRunTrace(
            trace_id=trace_id,
            question_id=q_id,
            question=question,
            config=self.config,
            max_index_gold_coverage=max_cov,
            candidates=candidates,
            retrieval_latency_ms=ret_ms,
            rerank_latency_ms=rerank_ms,
            assembled_prompt=prompt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_answer=llm_answer,
            abstained=abstained,
            generation_latency_ms=gen_ms,
            is_answer_correct=is_correct,
            oracle_answer=oracle_answer,
            oracle_passed=oracle_passed,
        )

        diagnosis = (
            self.diagnoser.diagnose(trace, benchmark) if benchmark is not None else None
        )
        return trace, diagnosis

    def run_benchmark_suite(
        self, benchmarks: List[BenchmarkItem]
    ) -> List[Tuple[PipelineRunTrace, DiagnosisResult]]:
        results = []
        for bm in benchmarks:
            trace, diag = self.run_single(bm.question, bm)
            results.append((trace, diag))
        return results
