import random
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.pipeline.runner import RAGPipelineRunner
from backend.app.schemas.contracts import (
    BenchmarkItem,
    ChunkStrategy,
    Document,
    FailureLabel,
    PipelineConfig,
    RetrieverType,
)


def config_to_numeric_vector(cfg: PipelineConfig) -> List[float]:
    """Converts a PipelineConfig into a numeric feature vector for regression surrogates."""
    strategy_val = 1.0 if cfg.chunk_strategy == ChunkStrategy.RECURSIVE else 0.0
    hybrid_val = 1.0 if cfg.retriever_type == RetrieverType.HYBRID_RRF else 0.0
    rerank_val = 1.0 if cfg.reranker_model is not None else 0.0
    return [
        float(cfg.chunk_size),
        float(cfg.chunk_overlap),
        strategy_val,
        hybrid_val,
        float(cfg.top_n_candidates),
        rerank_val,
        float(cfg.top_k_final),
        float(cfg.rrf_k),
    ]


class RegressionSurrogateSuite:
    """
    Implements the Regression & Surrogate Predictors (Module 9 & 10):
      1. Ridge Regression (L2 regularized) predicting pipeline latency/token cost from config knobs.
      2. Random Forest Regressor surrogate predicting answer accuracy score from config knobs.
    """

    def __init__(self, random_state: int = 42):
        self.cost_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )
        self.quality_surrogate = RandomForestRegressor(
            n_estimators=80, max_depth=6, random_state=random_state
        )
        self.is_fitted = False

    def fit(self, trial_records: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(trial_records) < 3:
            return {"cost_r2": 0.0, "quality_r2": 0.0}

        X = np.array(
            [config_to_numeric_vector(r["config_obj"]) for r in trial_records],
            dtype=np.float32,
        )
        y_cost = np.array([r["avg_prompt_tokens"] for r in trial_records], dtype=np.float32)
        y_qual = np.array([r["accuracy"] for r in trial_records], dtype=np.float32)

        self.cost_model.fit(X, y_cost)
        self.quality_surrogate.fit(X, y_qual)
        self.is_fitted = True

        pred_cost = self.cost_model.predict(X)
        pred_qual = self.quality_surrogate.predict(X)

        cost_r2 = float(r2_score(y_cost, pred_cost)) if np.var(y_cost) > 0 else 1.0
        qual_r2 = float(r2_score(y_qual, pred_qual)) if np.var(y_qual) > 0 else 1.0

        return {
            "cost_r2": round(cost_r2, 4),
            "quality_r2": round(qual_r2, 4),
        }

    def predict(self, cfg: PipelineConfig) -> Tuple[float, float]:
        """Returns (predicted_accuracy, predicted_prompt_tokens)."""
        if not self.is_fitted:
            return 0.5, 200.0
        x = np.array([config_to_numeric_vector(cfg)], dtype=np.float32)
        pred_acc = float(self.quality_surrogate.predict(x)[0])
        pred_tok = float(self.cost_model.predict(x)[0])
        return round(pred_acc, 4), round(pred_tok, 1)


class ParetoAutoTuner:
    """
    Multi-Objective Auto-Tuner (Module 10).
    Evaluates baseline config, diagnoses dominant root causes, searches candidate configs,
    fits Regression Surrogates, and computes the Pareto Frontier (Accuracy vs. Token Cost/Latency).
    """

    def __init__(
        self,
        documents: List[Document],
        benchmarks: List[BenchmarkItem],
        random_state: int = 42,
    ):
        self.documents = documents
        self.benchmarks = benchmarks
        self.rng = random.Random(random_state)
        self.surrogates = RegressionSurrogateSuite(random_state=random_state)

    def evaluate_config(self, cfg: PipelineConfig) -> Dict[str, Any]:
        runner = RAGPipelineRunner(self.documents, cfg)
        suite_results = runner.run_benchmark_suite(self.benchmarks)

        total = max(1, len(suite_results))
        correct_count = sum(1 for t, _ in suite_results if t.is_answer_correct)
        recall_at_k_count = sum(
            1
            for t, _ in suite_results
            if any(c.gold_coverage_ratio >= 0.75 for c in t.get_final_context_chunks())
        )

        avg_latency = float(np.mean([t.total_latency_ms for t, _ in suite_results]))
        avg_tokens = float(np.mean([t.prompt_tokens for t, _ in suite_results]))

        failure_counts: Dict[str, int] = {}
        for _, diag in suite_results:
            if diag is not None:
                lbl = diag.primary_root_cause.value
                failure_counts[lbl] = failure_counts.get(lbl, 0) + 1

        return {
            "config_hash": cfg.config_hash,
            "config": cfg.model_dump(),
            "config_obj": cfg,
            "accuracy": round(correct_count / total, 4),
            "recall_at_k": round(recall_at_k_count / total, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_prompt_tokens": round(avg_tokens, 1),
            "failure_breakdown": failure_counts,
        }

    def generate_candidate_space(
        self, baseline_cfg: PipelineConfig, n_trials: int = 8
    ) -> List[PipelineConfig]:
        """
        Generates a diverse set of candidate configurations spanning chunk sizes,
        strategies, retriever modes, rerankers, and top_k cutoffs.
        """
        candidates: List[PipelineConfig] = [baseline_cfg]
        seen_hashes = {baseline_cfg.config_hash}

        # Curated high-signal engineering configurations + randomized exploration
        preset_knobs = [
            dict(chunk_strategy=ChunkStrategy.RECURSIVE, chunk_size=220, chunk_overlap=40, retriever_type=RetrieverType.HYBRID_RRF, top_n_candidates=8, top_k_final=2, reranker_model="BAAI/bge-reranker-base"),
            dict(chunk_strategy=ChunkStrategy.RECURSIVE, chunk_size=280, chunk_overlap=50, retriever_type=RetrieverType.HYBRID_RRF, top_n_candidates=10, top_k_final=3, reranker_model="BAAI/bge-reranker-base"),
            dict(chunk_strategy=ChunkStrategy.RECURSIVE, chunk_size=180, chunk_overlap=30, retriever_type=RetrieverType.BM25, top_n_candidates=6, top_k_final=1, reranker_model=None),
            dict(chunk_strategy=ChunkStrategy.FIXED, chunk_size=120, chunk_overlap=0, retriever_type=RetrieverType.DENSE, top_n_candidates=5, top_k_final=2, reranker_model=None),
            dict(chunk_strategy=ChunkStrategy.RECURSIVE, chunk_size=350, chunk_overlap=60, retriever_type=RetrieverType.HYBRID_RRF, top_n_candidates=12, top_k_final=4, reranker_model="BAAI/bge-reranker-base"),
        ]

        for knobs in preset_knobs:
            if len(candidates) >= n_trials:
                break
            cfg = PipelineConfig(**knobs)
            if cfg.config_hash not in seen_hashes:
                seen_hashes.add(cfg.config_hash)
                candidates.append(cfg)

        while len(candidates) < n_trials:
            c_size = self.rng.choice([96, 160, 220, 320, 480])
            c_overlap = self.rng.choice([0, 20, 40])
            strat = self.rng.choice([ChunkStrategy.FIXED, ChunkStrategy.RECURSIVE])
            ret_type = self.rng.choice([RetrieverType.DENSE, RetrieverType.BM25, RetrieverType.HYBRID_RRF])
            top_k = self.rng.choice([1, 2, 3, 5])
            top_n = max(top_k + 2, self.rng.choice([5, 8, 12]))
            rerank = self.rng.choice([None, "BAAI/bge-reranker-base"])

            cfg = PipelineConfig(
                chunk_strategy=strat,
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                retriever_type=ret_type,
                top_n_candidates=top_n,
                top_k_final=top_k,
                reranker_model=rerank,
            )
            if cfg.config_hash not in seen_hashes:
                seen_hashes.add(cfg.config_hash)
                candidates.append(cfg)

        return candidates

    @staticmethod
    def compute_pareto_frontier(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Finds the Pareto-optimal configurations that maximize `accuracy`
        and minimize `avg_prompt_tokens` (cost/latency proxy).
        """
        pareto: List[Dict[str, Any]] = []
        for candidate in trials:
            dominated = False
            for other in trials:
                if other["config_hash"] == candidate["config_hash"]:
                    continue
                better_or_equal_acc = other["accuracy"] >= candidate["accuracy"]
                lower_or_equal_cost = other["avg_prompt_tokens"] <= candidate["avg_prompt_tokens"]
                strictly_better = (
                    other["accuracy"] > candidate["accuracy"]
                    or other["avg_prompt_tokens"] < candidate["avg_prompt_tokens"]
                )
                if better_or_equal_acc and lower_or_equal_cost and strictly_better:
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)

        pareto.sort(key=lambda r: (-r["accuracy"], r["avg_prompt_tokens"]))
        return pareto

    def run_auto_tune(
        self, baseline_cfg: PipelineConfig, n_trials: int = 6
    ) -> Dict[str, Any]:
        configs = self.generate_candidate_space(baseline_cfg, n_trials=n_trials)
        trials: List[Dict[str, Any]] = []

        for cfg in configs:
            res = self.evaluate_config(cfg)
            trials.append(res)

        surrogate_metrics = self.surrogates.fit(trials)
        pareto_front = self.compute_pareto_frontier(trials)

        # Pick best overall config (highest accuracy, tie-broken by lowest prompt tokens)
        best_trial = max(
            trials, key=lambda r: (r["accuracy"], r["recall_at_k"], -r["avg_prompt_tokens"])
        )
        baseline_trial = trials[0]

        # Clean out raw Pydantic object before JSON serialization
        clean_trials = [{k: v for k, v in t.items() if k != "config_obj"} for t in trials]
        clean_pareto = [{k: v for k, v in t.items() if k != "config_obj"} for t in pareto_front]
        clean_best = {k: v for k, v in best_trial.items() if k != "config_obj"}
        clean_base = {k: v for k, v in baseline_trial.items() if k != "config_obj"}

        return {
            "baseline": clean_base,
            "best_recommended": clean_best,
            "accuracy_lift": round(clean_best["accuracy"] - clean_base["accuracy"], 4),
            "recall_lift": round(clean_best["recall_at_k"] - clean_base["recall_at_k"], 4),
            "surrogate_metrics": surrogate_metrics,
            "pareto_frontier": clean_pareto,
            "all_trials": clean_trials,
        }
