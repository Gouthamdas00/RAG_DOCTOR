import os
import re
import time
import urllib.request
import json
from typing import List, Optional, Tuple

from backend.app.core.cache import DiskCache
from backend.app.schemas.contracts import (
    BenchmarkItem,
    PipelineConfig,
    ScoredChunkTrace,
)

ABSTAIN_PHRASE = "INSUFFICIENT_CONTEXT: I cannot answer this from the provided documents."


def _normalize_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class LLMGeneratorAndJudge:
    """
    Handles Prompt Assembly, Answer Generation, Counterfactual Oracle Probes,
    and Answer Correctness Judging.
    """

    def __init__(self, cache: Optional[DiskCache] = None, use_live_api: bool = False):
        self.cache = cache or DiskCache()
        self.use_live_api = use_live_api and bool(os.getenv("GEMINI_API_KEY"))

    def assemble_prompt(
        self, question: str, context_chunks: List[ScoredChunkTrace]
    ) -> str:
        context_blocks = []
        for idx, c in enumerate(context_chunks, start=1):
            context_blocks.append(f"[Chunk #{idx} | ID: {c.chunk_id}]\n{c.text}")
        joined_context = "\n\n".join(context_blocks)
        return (
            "You are a precise technical assistant. Answer the question using ONLY the provided context.\n"
            f"If the context does not contain the answer, reply exactly with: '{ABSTAIN_PHRASE}'\n\n"
            f"--- CONTEXT ---\n{joined_context}\n\n"
            f"--- QUESTION ---\n{question}\n\n"
            "--- ANSWER ---"
        )

    def generate(
        self,
        question: str,
        context_chunks: List[ScoredChunkTrace],
        config: PipelineConfig,
    ) -> Tuple[str, str, int, int, bool, float]:
        t0 = time.perf_counter()
        prompt = self.assemble_prompt(question, context_chunks)
        prompt_tokens = max(1, len(prompt) // 4)

        payload = {
            "mode": "live" if self.use_live_api else "local_sim_v2",
            "model": config.llm_model,
            "temp": config.temperature,
            "prompt": prompt,
            "top_k": config.top_k_final,
        }

        def _compute():
            if self.use_live_api:
                return self._call_gemini_api(prompt, config)
            return self._simulate_generation(question, context_chunks, config)

        answer = self.cache.get_or_compute("llm_gen", payload, _compute)
        completion_tokens = max(1, len(answer) // 4)
        abstained = ABSTAIN_PHRASE.lower() in answer.lower()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return prompt, answer, prompt_tokens, completion_tokens, abstained, latency_ms

    def run_oracle_probe(
        self, question: str, gold_passage_text: str, config: PipelineConfig
    ) -> str:
        oracle_chunk = ScoredChunkTrace(
            chunk_id="oracle_gold",
            doc_id="oracle_doc",
            span={"start_char": 0, "end_char": len(gold_passage_text)},
            text=gold_passage_text,
            pre_rerank_rank=1,
            final_rank=1,
            gold_coverage_ratio=1.0,
        )
        _, answer, _, _, _, _ = self.generate(question, [oracle_chunk], config)
        return answer

    def evaluate_correctness(
        self, llm_answer: str, benchmark: BenchmarkItem
    ) -> bool:
        if benchmark.question_type == "unanswerable":
            return ABSTAIN_PHRASE.lower() in llm_answer.lower()

        if ABSTAIN_PHRASE.lower() in llm_answer.lower():
            return False

        gold_tokens = set(_normalize_tokens(benchmark.gold_answer))
        ans_tokens = set(_normalize_tokens(llm_answer))
        if not gold_tokens:
            return False

        overlap_ratio = len(gold_tokens.intersection(ans_tokens)) / len(gold_tokens)
        return overlap_ratio >= 0.75

    def _simulate_generation(
        self,
        question: str,
        context_chunks: List[ScoredChunkTrace],
        config: PipelineConfig,
    ) -> str:
        if not context_chunks:
            return ABSTAIN_PHRASE

        if config.llm_model == "weak-tiny-llm":
            return "I could not parse the technical formatting in the passage."

        q_tokens = set(_normalize_tokens(question))
        best_chunk = None
        best_score = 0.0
        best_pos = 1

        for pos, chunk in enumerate(context_chunks, start=1):
            c_tokens = set(_normalize_tokens(chunk.text))
            overlap = len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))
            score = max(overlap, chunk.gold_coverage_ratio)
            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_pos = pos

        if best_chunk is None or best_score < 0.15:
            return ABSTAIN_PHRASE

        # Simulate Context Noise / Lost-in-the-Middle when too many distractors are stuffed into the prompt
        if len(context_chunks) >= 5 and best_pos >= 3:
            return "Based on the surrounding sections, the configuration depends on regional defaults."

        # Split by sentence boundaries OR newlines so headers don't obscure the answer sentence
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n+", best_chunk.text.strip())
            if s.strip()
        ]
        best_sentence = max(
            sentences,
            key=lambda s: len(q_tokens.intersection(set(_normalize_tokens(s)))),
        )
        return best_sentence

    def _call_gemini_api(self, prompt: str, config: PipelineConfig) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{config.llm_model}:generateContent?key={api_key}"
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": config.temperature},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
