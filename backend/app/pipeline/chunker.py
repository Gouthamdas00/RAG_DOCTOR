from typing import List, Optional
from backend.app.schemas.contracts import (
    BenchmarkItem,
    CharSpan,
    Chunk,
    ChunkStrategy,
    Document,
    PipelineConfig,
)


class DocumentChunker:
    """
    Slices Documents into Chunks while strictly preserving exact [start_char, end_char]
    offsets in the original document text so gold_coverage math is 100% exact.
    """

    def chunk_corpus(
        self, documents: List[Document], config: PipelineConfig
    ) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc, config)
            all_chunks.extend(doc_chunks)
        return all_chunks

    def chunk_document(
        self, doc: Document, config: PipelineConfig
    ) -> List[Chunk]:
        if config.chunk_strategy == ChunkStrategy.FIXED:
            return self._fixed_chunk(doc, config.chunk_size, config.chunk_overlap)
        elif config.chunk_strategy in (ChunkStrategy.RECURSIVE, ChunkStrategy.SEMANTIC):
            return self._recursive_chunk(doc, config.chunk_size, config.chunk_overlap)
        return self._fixed_chunk(doc, config.chunk_size, config.chunk_overlap)

    def _fixed_chunk(
        self, doc: Document, chunk_size: int, chunk_overlap: int
    ) -> List[Chunk]:
        text = doc.text
        n = len(text)
        if n == 0:
            return []

        step = max(1, chunk_size - chunk_overlap)
        chunks: List[Chunk] = []
        idx = 0
        start = 0

        while start < n:
            end = min(n, start + chunk_size)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_c{idx}",
                    doc_id=doc.doc_id,
                    text=text[start:end],
                    span=CharSpan(start_char=start, end_char=end),
                )
            )
            if end == n:
                break
            start += step
            idx += 1

        return chunks

    def _recursive_chunk(
        self, doc: Document, chunk_size: int, chunk_overlap: int
    ) -> List[Chunk]:
        """
        Boundary-aware sliding window chunker.
        Looks for natural break points (\n\n, \n, '. ', ' ') near the end of the target window
        so sentences and paragraphs aren't needlessly cut in the middle of a word.
        """
        text = doc.text
        n = len(text)
        if n == 0:
            return []

        separators = ["\n\n", "\n", ". ", " "]
        chunks: List[Chunk] = []
        idx = 0
        start = 0

        while start < n:
            target_end = min(n, start + chunk_size)
            actual_end = target_end

            if target_end < n:
                # Search the last 35% of the window for the cleanest natural boundary
                search_floor = start + max(1, int(chunk_size * 0.65))
                window_slice = text[search_floor:target_end]
                for sep in separators:
                    pos = window_slice.rfind(sep)
                    if pos != -1:
                        actual_end = search_floor + pos + len(sep)
                        break

            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_c{idx}",
                    doc_id=doc.doc_id,
                    text=text[start:actual_end],
                    span=CharSpan(start_char=start, end_char=actual_end),
                )
            )

            if actual_end >= n:
                break

            next_start = actual_end - chunk_overlap
            if next_start <= start:
                next_start = start + max(1, chunk_size - chunk_overlap)
            start = next_start
            idx += 1

        return chunks

    @staticmethod
    def compute_max_gold_coverage(
        chunks: List[Chunk], benchmark: Optional[BenchmarkItem]
    ) -> Optional[float]:
        """
        Finds the highest fraction (0.0 to 1.0) of the benchmark's gold answer span
        contained inside any single chunk in the index.
        """
        if benchmark is None or benchmark.question_type == "unanswerable":
            return None
        doc_chunks = [c for c in chunks if c.doc_id == benchmark.gold_doc_id]
        if not doc_chunks:
            return 0.0
        return max(
            c.gold_coverage(benchmark.gold_doc_id, benchmark.gold_span)
            for c in doc_chunks
        )
