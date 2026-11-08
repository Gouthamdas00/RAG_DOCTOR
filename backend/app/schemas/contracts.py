import hashlib
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, computed_field


# ==========================================
# 1. ENUMS (Standardized Labels & Options)
# ==========================================

class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RetrieverType(str, Enum):
    DENSE = "dense"              # Vector similarity search (meaning-based)
    BM25 = "bm25"                # Exact keyword matching
    HYBRID_RRF = "hybrid_rrf"    # Combines Dense + BM25 using Reciprocal Rank Fusion


class FailureLabel(str, Enum):
    """
    The 8 standardized outcomes when diagnosing a RAG pipeline run.
    """
    CHUNKING = "CHUNKING"            # The answer was sliced in half or buried during document splitting
    RETRIEVAL = "RETRIEVAL"          # The search engine failed to find the right chunk in the top-N pool
    RANKING_K = "RANKING_K"          # Found in top-N (e.g. top 20), but missed the final top-k (e.g. top 5) cutoff
    RERANKER = "RERANKER"            # The reranker model actively demoted the good chunk out of the top-k
    CONTEXT_NOISE = "CONTEXT_NOISE"  # The right chunk was in the prompt, but extra noisy chunks confused the LLM
    GENERATOR = "GENERATOR"          # Even when given ONLY the perfect chunk, the LLM still failed (hallucinated/incapable)
    ABSTENTION = "ABSTENTION"        # Question was unanswerable, or LLM wrongly refused/answered an unanswerable query
    NO_FAILURE = "NO_FAILURE"        # Pipeline answered accurately


# ==========================================
# 2. PIPELINE CONFIGURATION SCHEMA
# ==========================================

class PipelineConfig(BaseModel):
    """
    Represents a specific set of RAG hyperparameters.
    When the Auto-Tuner optimizes a pipeline, it mutates this object.
    """
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    retriever_type: RetrieverType = RetrieverType.HYBRID_RRF
    top_n_candidates: int = Field(default=20, ge=1)
    reranker_model: Optional[str] = "BAAI/bge-reranker-base"
    top_k_final: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    llm_model: str = "gemini-2.5-flash"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @computed_field
    @property
    def config_hash(self) -> str:
        """
        Creates a unique 12-character fingerprint for this exact configuration.
        Allows us to group traces by config and cache embeddings/results on disk.
        """
        payload = self.model_dump_json(exclude={"config_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ==========================================
# 3. DOCUMENT, CHUNK & BENCHMARK SCHEMAS
# ==========================================

class CharSpan(BaseModel):
    """Tracks the exact character start and end index [start_char, end_char] in a raw document."""
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)

    @property
    def length(self) -> int:
        return max(0, self.end_char - self.start_char)

    def coverage_of(self, target_span: "CharSpan") -> float:
        """
        Calculates what fraction (0.0 to 1.0) of `target_span` lies inside `self`.
        """
        if target_span.length == 0:
            return 0.0
        overlap_start = max(self.start_char, target_span.start_char)
        overlap_end = min(self.end_char, target_span.end_char)
        overlap_length = max(0, overlap_end - overlap_start)
        return overlap_length / target_span.length


class Document(BaseModel):
    """A raw source document loaded into the system."""
    doc_id: str
    title: str
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A sliced piece of a Document produced by the Chunker."""
    chunk_id: str
    doc_id: str
    text: str
    span: CharSpan

    def gold_coverage(self, gold_doc_id: str, gold_span: CharSpan) -> float:
        """
        Returns 0.0 to 1.0 indicating how much of the ground-truth answer span
        is captured inside this specific chunk.
        """
        if self.doc_id != gold_doc_id:
            return 0.0
        return self.span.coverage_of(gold_span)


class BenchmarkItem(BaseModel):
    """
    A test case used in Offline CI Mode to evaluate and diagnose a pipeline.
    """
    question_id: str
    question: str
    gold_answer: str
    gold_doc_id: str
    gold_span: CharSpan
    question_type: str = "single_hop"  # single_hop, multi_hop, numeric, unanswerable


# ==========================================
# 4. OBSERVABILITY & STAGE TRACE SCHEMAS
# ==========================================

class ScoredChunkTrace(BaseModel):
    """
    Tracks a single candidate chunk as it moves through Retrieval -> Fusion -> Reranking.
    """
    chunk_id: str
    doc_id: str
    span: CharSpan
    text: str

    # Stage 1: Raw Retriever Scores & Ranks (1-indexed: Rank 1 is best)
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    fused_score: Optional[float] = None
    pre_rerank_rank: int

    # Stage 2: Reranker Score & Final Rank
    rerank_score: Optional[float] = None
    final_rank: int

    # Ground-truth overlap (populated automatically in Offline CI Mode)
    gold_coverage_ratio: float = 0.0

    @computed_field
    @property
    def rank_delta(self) -> int:
        """
        How many spots the reranker moved this chunk.
        Example: pre_rerank_rank = 12, final_rank = 2 -> rank_delta = +10 (promoted 10 spots).
        Example: pre_rerank_rank = 2, final_rank = 9  -> rank_delta = -7 (demoted 7 spots).
        """
        return self.pre_rerank_rank - self.final_rank


class PipelineRunTrace(BaseModel):
    """
    The complete flight-recorder log for a single question passing through the RAG pipeline.
    """
    trace_id: str
    question_id: str
    question: str
    config: PipelineConfig

    # Corpus chunking check (highest gold coverage achieved across ALL chunks in the index)
    max_index_gold_coverage: Optional[float] = None

    # Retrieval & Reranking telemetry
    candidates: List[ScoredChunkTrace]
    retrieval_latency_ms: float
    rerank_latency_ms: float = 0.0

    # Generation telemetry
    assembled_prompt: str
    prompt_tokens: int
    completion_tokens: int
    llm_answer: str
    abstained: bool = False
    generation_latency_ms: float

    # Evaluation & Counterfactual probes
    is_answer_correct: Optional[bool] = None
    oracle_answer: Optional[str] = None
    oracle_passed: Optional[bool] = None

    @computed_field
    @property
    def total_latency_ms(self) -> float:
        return round(
            self.retrieval_latency_ms + self.rerank_latency_ms + self.generation_latency_ms, 2
        )

    def get_final_context_chunks(self) -> List[ScoredChunkTrace]:
        """Returns the exact top-k chunks that were injected into the LLM prompt, ordered by rank."""
        final_chunks = [c for c in self.candidates if c.final_rank <= self.config.top_k_final]
        return sorted(final_chunks, key=lambda c: c.final_rank)


class DiagnosisResult(BaseModel):
    """
    The final output produced by RAG Doctor's Diagnostic Engine for a single trace.
    """
    trace_id: str
    question_id: str
    primary_root_cause: FailureLabel
    failure_probabilities: Dict[FailureLabel, float] = Field(default_factory=dict)
    explanation: str
    recommended_fix: str