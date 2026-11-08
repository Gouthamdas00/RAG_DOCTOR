import hashlib
import re
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.schemas.contracts import (
    BenchmarkItem,
    Chunk,
    PipelineConfig,
    RetrieverType,
    ScoredChunkTrace,
)


def tokenize(text: str) -> List[str]:
    """Clean alphanumeric + word-ngram tokenizer for BM25 and local embeddings."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class HybridRetrieverEngine:
    """
    Production-style Two-Stage Retrieval & Reranking Engine:
      Stage 1: Dense Vector Search, Sparse BM25 Search, or Hybrid Reciprocal Rank Fusion (RRF)
      Stage 2: Cross-Encoder Reranking (with full rank_delta and latency telemetry)
    """

    def __init__(self, chunks: List[Chunk], config: PipelineConfig, use_neural_models: bool = False):
        """
        use_neural_models=False uses deterministic character/token n-gram projections so
        tests and local sweeps run in <10ms without downloading 500MB HuggingFace weights.
        Set use_neural_models=True to load SentenceTransformer and CrossEncoder weights.
        """
        self.chunks = chunks
        self.config = config
        self.use_neural_models = use_neural_models
        self._embedder = None
        self._reranker = None

        # 1. Build BM25 Index
        self.tokenized_corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if chunks else None

        # 2. Build Dense Vector Index
        self.chunk_Embeddings = self._encode_texts([c.text for c in chunks]) if chunks else np.zeros((0, 256))

    def _encode_texts(self, texts: List[str], dim: int = 256) -> np.ndarray:
        if self.use_neural_models:
            from sentence_transformers import SentenceTransformer
            if self._embedder is None:
                self._embedder = SentenceTransformer(self.config.embedding_model)
            vecs = self._embedder.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)

        # Fast deterministic L2-normalized feature-hashing embedding (unigrams + bigrams)
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
            for feat in features:
                h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
                bucket = h % dim
                sign = 1.0 if (h & 1) == 0 else -1.0
                matrix[row_idx, bucket] += sign
            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm
        return matrix

    def _score_dense(self, query: str) -> Tuple[List[float], List[int]]:
        q_vec = self._encode_texts([query])[0]
        scores = np.dot(self.chunk_Embeddings, q_vec).tolist()
        sorted_indices = np.argsort(scores)[::-1]
        ranks = [0] * len(self.chunks)
        for rank_pos, chunk_idx in enumerate(sorted_indices, start=1):
            ranks[chunk_idx] = rank_pos
        return scores, ranks

    def _score_bm25(self, query: str) -> Tuple[List[float], List[int]]:
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens).tolist() if self.bm25 else [0.0] * len(self.chunks)
        sorted_indices = np.argsort(scores)[::-1]
        ranks = [0] * len(self.chunks)
        for rank_pos, chunk_idx in enumerate(sorted_indices, start=1):
            ranks[chunk_idx] = rank_pos
        return scores, ranks

    def _rerank_candidates(
        self, query: str, candidate_indices: List[int]
    ) -> Dict[int, float]:
        if not candidate_indices:
            return {}

        if self.use_neural_models and self.config.reranker_model:
            from sentence_transformers import CrossEncoder
            if self._reranker is None:
                self._reranker = CrossEncoder(self.config.reranker_model)
            pairs = [(query, self.chunks[idx].text) for idx in candidate_indices]
            scores = self._reranker.predict(pairs).tolist()
            return {idx: float(s) for idx, s in zip(candidate_indices, scores)}

        # Fast deterministic Cross-Scorer: exact token + bigram phrase precision
        q_tokens = set(tokenize(query))
        scores_map: Dict[int, float] = {}
        for idx in candidate_indices:
            c_tokens = set(tokenize(self.chunks[idx].text))
            if not q_tokens:
                scores_map[idx] = 0.0
            else:
                overlap = len(q_tokens.intersection(c_tokens)) / len(q_tokens)
                scores_map[idx] = round(float(overlap), 4)
        return scores_map

    def retrieve_and_rerank(
        self, query: str, benchmark: Optional[BenchmarkItem] = None
    ) -> Tuple[List[ScoredChunkTrace], float, float]:
        """
        Executes Stage 1 (Retrieval) and Stage 2 (Reranking), returning:
          (candidates_trace_list, retrieval_latency_ms, rerank_latency_ms)
        """
        if not self.chunks:
            return [], 0.0, 0.0

        # -----------------------------------------------------
        # Stage 1: Retrieval (Dense, BM25, or Hybrid RRF)
        # -----------------------------------------------------
        t0 = time.perf_counter()
        dense_scores, dense_ranks = self._score_dense(query)
        bm25_scores, bm25_ranks = self._score_bm25(query)

        rtype = self.config.retriever_type
        rrf_k = self.config.rrf_k
        primary_scores: List[float] = []

        for i in range(len(self.chunks)):
            if rtype == RetrieverType.DENSE:
                primary_scores.append(dense_scores[i])
            elif rtype == RetrieverType.BM25:
                primary_scores.append(bm25_scores[i])
            else:
                # Hybrid Reciprocal Rank Fusion (RRF)
                fused = (1.0 / (rrf_k + dense_ranks[i])) + (1.0 / (rrf_k + bm25_ranks[i]))
                primary_scores.append(fused)

        # Rank all chunks by primary_scores descending and slice top_n_candidates
        ordered_indices = np.argsort(primary_scores)[::-1].tolist()
        top_n_indices = ordered_indices[: self.config.top_n_candidates]
        retrieval_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # -----------------------------------------------------
        # Stage 2: Optional Cross-Encoder Reranking
        # -----------------------------------------------------
        t1 = time.perf_counter()
        rerank_scores_map: Dict[int, float] = {}
        if self.config.reranker_model is not None:
            rerank_scores_map = self._rerank_candidates(query, top_n_indices)
            # Sort top_n_indices by reranker score descending (tie-breaking by pre-rerank order)
            reranked_order = sorted(
                top_n_indices,
                key=lambda idx: (rerank_scores_map[idx], primary_scores[idx]),
                reverse=True,
            )
            final_rank_map = {idx: pos for pos, idx in enumerate(reranked_order, start=1)}
        else:
            final_rank_map = {idx: pos for pos, idx in enumerate(top_n_indices, start=1)}
        rerank_latency_ms = round((time.perf_counter() - t1) * 1000.0, 2)

        # -----------------------------------------------------
        # Build ScoredChunkTrace list for telemetry & diagnosis
        # -----------------------------------------------------
        traces: List[ScoredChunkTrace] = []
        for pre_rank, chunk_idx in enumerate(top_n_indices, start=1):
            chunk = self.chunks[chunk_idx]
            coverage = (
                chunk.gold_coverage(benchmark.gold_doc_id, benchmark.gold_span)
                if benchmark and benchmark.question_type != "unanswerable"
                else 0.0
            )
            traces.append(
                ScoredChunkTrace(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    span=chunk.span,
                    text=chunk.text,
                    dense_score=round(float(dense_scores[chunk_idx]), 4),
                    dense_rank=dense_ranks[chunk_idx],
                    bm25_score=round(float(bm25_scores[chunk_idx]), 4),
                    bm25_rank=bm25_ranks[chunk_idx],
                    fused_score=(
                        round(float(primary_scores[chunk_idx]), 6)
                        if rtype == RetrieverType.HYBRID_RRF
                        else None
                    ),
                    pre_rerank_rank=pre_rank,
                    rerank_score=rerank_scores_map.get(chunk_idx),
                    final_rank=final_rank_map[chunk_idx],
                    gold_coverage_ratio=round(coverage, 4),
                )
            )

        traces.sort(key=lambda t: t.final_rank)
        return traces, retrieval_latency_ms, rerank_latency_ms
