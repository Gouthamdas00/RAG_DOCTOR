from typing import Any, Dict, List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from backend.app.schemas.contracts import DiagnosisResult, FailureLabel, PipelineRunTrace


class FailureClusterer:
    """
    Unsupervised Failure-Theme Discovery Engine (Module 8).
    Groups failed RAG traces into human-interpretable failure themes using TF-IDF + K-Means++,
    automatically selects k via Silhouette Score, and computes 2D PCA coordinates for the UI.
    """

    def __init__(self, max_clusters: int = 5, random_state: int = 42):
        self.max_clusters = max_clusters
        self.random_state = random_state

    def discover_themes(
        self,
        traces: List[PipelineRunTrace],
        diagnoses: List[DiagnosisResult],
    ) -> Dict[str, Any]:
        # Filter to failed runs only
        failed_pairs = [
            (t, d)
            for t, d in zip(traces, diagnoses)
            if d.primary_root_cause != FailureLabel.NO_FAILURE
        ]

        if len(failed_pairs) < 3:
            return {"num_failed": len(failed_pairs), "optimal_k": 1, "clusters": [], "points_2d": []}

        questions = [t.question for t, _ in failed_pairs]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200)
        X_tfidf = vectorizer.fit_transform(questions).toarray()
        vocab = np.array(vectorizer.get_feature_names_out())

        # Choose optimal k between 2 and min(max_clusters, n_samples - 1) using Silhouette Score
        n_samples = len(failed_pairs)
        max_k = min(self.max_clusters, n_samples - 1)
        best_k = 2
        best_sil = -1.0
        best_labels = np.zeros(n_samples, dtype=int)

        for k in range(2, max(3, max_k + 1)):
            if k >= n_samples:
                break
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=self.random_state)
            lbls = km.fit_predict(X_tfidf)
            if len(set(lbls)) > 1:
                sil = float(silhouette_score(X_tfidf, lbls))
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = lbls

        # Compute 2D PCA projection for dashboard visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        coords_2d = pca.fit_transform(X_tfidf) if X_tfidf.shape[1] >= 2 else np.zeros((n_samples, 2))

        clusters_summary = []
        for cluster_id in range(best_k):
            mask = best_labels == cluster_id
            if not np.any(mask):
                continue

            cluster_indices = np.where(mask)[0].tolist()
            mean_tfidf = X_tfidf[mask].mean(axis=0)
            top_term_indices = np.argsort(mean_tfidf)[::-1][:4]
            top_keywords = [str(vocab[i]) for i in top_term_indices if mean_tfidf[i] > 0]

            # Find dominant root cause in this cluster
            cause_counts: Dict[str, int] = {}
            for idx in cluster_indices:
                cause = failed_pairs[idx][1].primary_root_cause.value
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
            dominant_cause = max(cause_counts.items(), key=lambda kv: kv[1])[0]

            clusters_summary.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(cluster_indices),
                    "theme_keywords": top_keywords,
                    "dominant_root_cause": dominant_cause,
                    "cause_breakdown": cause_counts,
                    "sample_questions": [failed_pairs[i][0].question for i in cluster_indices[:3]],
                }
            )

        points_2d = [
            {
                "trace_id": failed_pairs[i][0].trace_id,
                "question": failed_pairs[i][0].question,
                "root_cause": failed_pairs[i][1].primary_root_cause.value,
                "cluster_id": int(best_labels[i]),
                "x": round(float(coords_2d[i, 0]), 4),
                "y": round(float(coords_2d[i, 1]), 4),
            }
            for i in range(n_samples)
        ]

        return {
            "num_failed": n_samples,
            "optimal_k": best_k,
            "silhouette_score": round(best_sil, 4),
            "clusters": clusters_summary,
            "points_2d": points_2d,
        }
