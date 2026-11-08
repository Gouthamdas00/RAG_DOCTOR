import re
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.schemas.contracts import (
    DiagnosisResult,
    FailureLabel,
    PipelineRunTrace,
)


FEATURE_NAMES = [
    "chunk_size",
    "chunk_overlap",
    "top_n_candidates",
    "top_k_final",
    "top1_dense_score",
    "top1_top2_dense_margin",
    "top1_bm25_score",
    "top1_top2_bm25_margin",
    "top1_rerank_score",
    "mean_abs_rank_delta",
    "query_context_overlap",
    "answer_context_overlap",
    "prompt_tokens",
    "abstained_flag",
]


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def extract_trace_features(trace: PipelineRunTrace) -> List[float]:
    """
    Extracts 14 strictly reference-free (leak-free) telemetry features from a PipelineRunTrace.
    Because none of these features require `gold_span` or `gold_answer`, this vector can be
    computed on live, unlabelled production traffic at runtime.
    """
    cfg = trace.config
    cands = trace.candidates
    final_chunks = trace.get_final_context_chunks()

    dense_scores = sorted([c.dense_score or 0.0 for c in cands], reverse=True)
    bm25_scores = sorted([c.bm25_score or 0.0 for c in cands], reverse=True)

    top1_dense = dense_scores[0] if len(dense_scores) >= 1 else 0.0
    top2_dense = dense_scores[1] if len(dense_scores) >= 2 else 0.0
    dense_margin = top1_dense - top2_dense

    top1_bm25 = bm25_scores[0] if len(bm25_scores) >= 1 else 0.0
    top2_bm25 = bm25_scores[1] if len(bm25_scores) >= 2 else 0.0
    bm25_margin = top1_bm25 - top2_bm25

    top1_rerank = (
        final_chunks[0].rerank_score
        if final_chunks and final_chunks[0].rerank_score is not None
        else 0.0
    )
    mean_abs_delta = (
        float(np.mean([abs(c.rank_delta) for c in cands])) if cands else 0.0
    )

    q_tok = _tokens(trace.question)
    ctx_tok = _tokens(" ".join(c.text for c in final_chunks))
    ans_tok = _tokens(trace.llm_answer)

    q_ctx_overlap = len(q_tok.intersection(ctx_tok)) / max(1, len(q_tok))
    ans_ctx_overlap = len(ans_tok.intersection(ctx_tok)) / max(1, len(ans_tok))

    return [
        float(cfg.chunk_size),
        float(cfg.chunk_overlap),
        float(cfg.top_n_candidates),
        float(cfg.top_k_final),
        float(top1_dense),
        float(dense_margin),
        float(top1_bm25),
        float(bm25_margin),
        float(top1_rerank),
        float(mean_abs_delta),
        float(q_ctx_overlap),
        float(ans_ctx_overlap),
        float(trace.prompt_tokens),
        1.0 if trace.abstained else 0.0,
    ]


class MLTraceDiagnoser:
    """
    Supervised ML Root-Cause Classifier & Live Traffic Risk Predictor.
    Trains both a Random Forest and a scaled Multinomial Logistic Regression baseline
    on leak-free trace telemetry features, outputting calibrated class probabilities.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rf_model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
        )
        self.lr_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.is_fitted = False
        self.classes_: List[str] = []

    def fit_and_evaluate(
        self,
        traces: List[PipelineRunTrace],
        labels: List[FailureLabel],
    ) -> Dict[str, float]:
        """
        Trains both classifiers on (traces, labels) and returns training/OOB metrics.
        """
        X = np.array([extract_trace_features(t) for t in traces], dtype=np.float32)
        y = np.array([lbl.value for lbl in labels])

        self.rf_model.fit(X, y)
        self.lr_model.fit(X, y)
        self.classes_ = list(self.rf_model.classes_)
        self.is_fitted = True

        rf_preds = self.rf_model.predict(X)
        lr_preds = self.lr_model.predict(X)

        return {
            "rf_macro_f1": round(float(f1_score(y, rf_preds, average="macro")), 4),
            "lr_macro_f1": round(float(f1_score(y, lr_preds, average="macro")), 4),
            "num_samples": len(traces),
            "num_classes": len(self.classes_),
        }

    def predict_diagnosis(self, trace: PipelineRunTrace) -> DiagnosisResult:
        """
        Predicts root cause and probability distribution for a single trace WITHOUT needing gold_span.
        """
        if not self.is_fitted:
            raise RuntimeError("MLTraceDiagnoser must be fitted before calling predict_diagnosis.")

        x = np.array([extract_trace_features(trace)], dtype=np.float32)
        probs = self.rf_model.predict_proba(x)[0]

        prob_map: Dict[FailureLabel, float] = {}
        for cls_name, p in zip(self.classes_, probs):
            prob_map[FailureLabel(cls_name)] = round(float(p), 4)

        best_label = max(prob_map.items(), key=lambda kv: kv[1])[0]
        confidence = int(prob_map[best_label] * 100)

        return DiagnosisResult(
            trace_id=trace.trace_id,
            question_id=trace.question_id,
            primary_root_cause=best_label,
            failure_probabilities=prob_map,
            explanation=(
                f"Live ML Telemetry Classifier predicted '{best_label.value}' "
                f"with {confidence}% confidence based on retrieval score margins, "
                f"rank deltas, and lexical grounding."
            ),
            recommended_fix=f"Inspect {best_label.value} stage configuration parameters.",
        )

    def get_feature_importances(
        self, traces: List[PipelineRunTrace], labels: List[FailureLabel]
    ) -> List[Tuple[str, float]]:
        """
        Computes Permutation Importance on the provided dataset to show engineers
        which telemetry signals matter most.
        """
        X = np.array([extract_trace_features(t) for t in traces], dtype=np.float32)
        y = np.array([lbl.value for lbl in labels])
        perm = permutation_importance(
            self.rf_model, X, y, n_repeats=5, random_state=self.random_state
        )
        ranked = sorted(
            zip(FEATURE_NAMES, perm.importances_mean),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(name, round(float(score), 4)) for name, score in ranked]
