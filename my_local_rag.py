from backend.app.schemas.contracts import ChunkStrategy, PipelineConfig, RetrieverType

LOCAL_RAG_CONFIG = PipelineConfig(
    chunk_strategy=ChunkStrategy.FIXED,
    chunk_size=64,
    chunk_overlap=0,
    retriever_type=RetrieverType.DENSE,
    top_n_candidates=5,
    top_k_final=1,
)

LAST_USER_QUERY = "Within how many seconds does automated replica promotion complete during a primary node outage?"
