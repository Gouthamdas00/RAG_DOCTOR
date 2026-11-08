from backend.app.schemas.contracts import (
    CharSpan,
    Chunk,
    PipelineConfig,
    ScoredChunkTrace,
)


def test_chunk_gold_coverage():
    # Gold answer sits at characters [100, 200] (length = 100)
    gold_span = CharSpan(start_char=100, end_char=200)

    # Case 1: Chunk completely contains the gold answer [50, 250] -> 100% coverage
    good_chunk = Chunk(
        chunk_id="c1",
        doc_id="doc1",
        text="Full answer here",
        span=CharSpan(start_char=50, end_char=250),
    )
    assert good_chunk.gold_coverage("doc1", gold_span) == 1.0

    # Case 2: Chunker sliced the answer in half! Chunk only covers [50, 140] -> 40% coverage
    split_chunk = Chunk(
        chunk_id="c2",
        doc_id="doc1",
        text="Half answer",
        span=CharSpan(start_char=50, end_char=140),
    )
    assert split_chunk.gold_coverage("doc1", gold_span) == 0.4


def test_reranker_rank_delta():
    # Chunk was at rank 12 in initial retrieval, promoted to rank 2 by reranker -> +10 delta
    promoted = ScoredChunkTrace(
        chunk_id="c1",
        doc_id="doc1",
        span=CharSpan(start_char=0, end_char=100),
        text="sample",
        pre_rerank_rank=12,
        final_rank=2,
    )
    assert promoted.rank_delta == 10

    # Chunk was at rank 2, demoted to rank 7 by reranker -> -5 delta
    demoted = ScoredChunkTrace(
        chunk_id="c2",
        doc_id="doc1",
        span=CharSpan(start_char=0, end_char=100),
        text="sample",
        pre_rerank_rank=2,
        final_rank=7,
    )
    assert demoted.rank_delta == -5


def test_config_hash_changes():
    cfg1 = PipelineConfig(chunk_size=512)
    cfg2 = PipelineConfig(chunk_size=256)
    assert cfg1.config_hash != cfg2.config_hash


if __name__ == "__main__":
    test_chunk_gold_coverage()
    test_reranker_rank_delta()
    test_config_hash_changes()
    print("All contract tests passed!")