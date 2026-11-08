"""Opt-in quality check against an installed loopback Ollama code model.

Set RAG_DOCTOR_TEST_MODEL_NAME to enable. This test creates a temporary,
synthetic RAG workspace and never reads the engineer's application files.
"""

import os
import tempfile
import unittest
from pathlib import Path

from backend.app.diagnose.agentic_investigator import investigate_case


MODEL_NAME = os.getenv("RAG_DOCTOR_TEST_MODEL_NAME", "").strip()
MODEL_ENDPOINT = os.getenv(
    "RAG_DOCTOR_TEST_MODEL_ENDPOINT",
    "http://127.0.0.1:11434/api/chat",
).strip()


@unittest.skipUnless(MODEL_NAME, "Set RAG_DOCTOR_TEST_MODEL_NAME to run the local-model quality check.")
class LiveLocalModelQualityTests(unittest.TestCase):
    def _require_model_confirmation(self, result, verifier_key):
        """Never count deterministic fallback as proof of local-model quality."""
        verifier = result.get(verifier_key) or {}
        self.assertTrue(
            verifier.get("finding_detected"),
            "The deterministic verifier did not detect the synthetic defect; "
            f"inspect its scan trace before evaluating local-model quality: {verifier.get('trace')}",
        )
        model_error = str(result.get("model_error") or "")
        if model_error.startswith("Could not reach the configured local model."):
            self.skipTest("The local-model endpoint is unreachable; fallback-only output is not a model-quality result.")
        self.assertFalse(model_error, f"The local model did not complete successfully: {model_error}")
        self.assertTrue(
            result.get("model_confirmed_deterministic_check"),
            "The model did not independently confirm the verifier finding from verified evidence: "
            f"verifier={verifier}; model_assessment={result.get('model_assessment_before_verification')}; "
            f"model_root_cause={result.get('root_cause')}; model_status={result.get('status')}; "
            f"verified_fix={result.get('fix')}",
        )

    def test_unlabeled_generator_that_drops_context_gets_a_grounded_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source_dir = workspace / "src"
            source_dir.mkdir()
            (source_dir / "pipeline.py").write_text(
                "def fuse(a, b):\n"
                "    return f\"Question: {a}\"\n"
                "\n"
                "def execute(q):\n"
                "    z = store.search(q)\n"
                "    p = fuse(q, z)\n"
                "    return llm.invoke(p)\n",
                encoding="utf-8",
            )

            result = investigate_case(
                workspace,
                {"diagnosis": {}},
                MODEL_ENDPOINT,
                MODEL_NAME,
            )

        self._require_model_confirmation(result, "static_verifier")
        self.assertEqual(result["root_cause"], "GENERATOR", result)
        self.assertEqual(result["status"], "probable", result)
        self.assertEqual(result["fix"]["file"], "src/pipeline.py", result["fix"])
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertIn("{b}", result["fix"]["replacement_code"])
        self.assertTrue(any(item["verified"] for item in result["evidence"]), result["evidence"])

    def test_unseen_nonprogressing_chunker_gets_a_grounded_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source_dir = workspace / "src"
            source_dir.mkdir()
            (source_dir / "chunker.py").write_text(
                "A = 128\n"
                "B = 128\n"
                "\n"
                "def split(t):\n"
                "    step = A - B\n"
                "    result = []\n"
                "    i = 0\n"
                "    while i < len(t):\n"
                "        end = min(len(t), i + A)\n"
                "        result.append(t[i:end])\n"
                "        i += step\n"
                "    return result\n",
                encoding="utf-8",
            )

            result = investigate_case(
                workspace,
                {"diagnosis": {}},
                MODEL_ENDPOINT,
                MODEL_NAME,
            )

        self._require_model_confirmation(result, "static_verifier")
        self.assertEqual(result["root_cause"], "CHUNKING", result)
        self.assertEqual(result["status"], "probable", result)
        self.assertEqual(result["fix"]["file"], "src/chunker.py", result["fix"])
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertIn("if A <= B:", result["fix"]["replacement_code"])
        self.assertTrue(any(item["verified"] for item in result["evidence"]), result["evidence"])

    def test_local_model_confirms_an_unsafe_benchmark_acceptance_path(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            test_dir = workspace / "tests"
            test_dir.mkdir()
            (test_dir / "scoring.py").write_text(
                "def evaluate_match(gold_answer, answer_text):\n"
                "    gold_clean = gold_answer.strip().lower()\n"
                "    answer_clean = answer_text.strip().lower()\n"
                "    if gold_clean and gold_clean in answer_clean:\n"
                "        return True\n"
                "    return False\n"
                "\n"
                "def run_case(question, gold_answer, answer_text):\n"
                "    return evaluate_match(gold_answer, answer_text)\n",
                encoding="utf-8",
            )

            result = investigate_case(
                workspace,
                {"diagnosis": {}},
                MODEL_ENDPOINT,
                MODEL_NAME,
            )

        self._require_model_confirmation(result, "static_verifier")
        self.assertEqual(result["root_cause"], "EVALUATION", result)
        self.assertEqual(result["status"], "probable", result)
        self.assertTrue(result["model_confirmed_deterministic_check"], result)
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertIn("question", result["fix"]["proposed_change"].lower())
        self.assertIn("tests/scoring.py", result["agent_trace"]["source_files_provided"])
        self.assertTrue(all(item["verified"] for item in result["evidence"]), result["evidence"])

    def test_hybrid_suppresses_model_tuning_hallucination_for_prompt_misbinding(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source_dir = workspace / "src"
            source_dir.mkdir()
            (source_dir / "pipeline.ts").write_text(
                "export function buildPrompt(question: string, passages: string[]): string {\n"
                "  const context = passages.join('\\n');\n"
                "  return `Context:\\n${question}\\n\\nQuestion: ${question}`;\n"
                "}\n"
                "\n"
                "export async function answerQuestion(question: string) {\n"
                "  const passages = await vectorStore.similaritySearch(question);\n"
                "  const prompt = buildPrompt(question, passages);\n"
                "  return await model.generateContent(prompt);\n"
                "}\n",
                encoding="utf-8",
            )

            result = investigate_case(
                workspace,
                {"diagnosis": {}},
                MODEL_ENDPOINT,
                MODEL_NAME,
            )

        self._require_model_confirmation(result, "static_verifier")
        self.assertTrue(result["static_verifier"]["finding_detected"], result["static_verifier"])
        self.assertEqual(result["root_cause"], "GENERATOR", result)
        self.assertEqual(result["status"], "probable", result)
        self.assertEqual(result["fix"]["file"], "src/pipeline.ts", result["fix"])
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertIn("${context}", result["fix"]["replacement_code"])
        self.assertNotIn("topK", result["fix"]["replacement_code"])
        self.assertTrue(any(item["verified"] for item in result["evidence"]), result["evidence"])
        self.assertIn("src/pipeline.ts", result["agent_trace"]["source_files_provided"])

    def test_hybrid_returns_grounded_fix_for_unseen_ranking_order_from_runtime_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source_dir = workspace / "src"
            source_dir.mkdir()
            (source_dir / "retriever.py").write_text(
                "# vector_index.search returns cosine similarity; larger scores mean closer matches.\n"
                "def rank_candidates(candidates):\n"
                "    return sorted(candidates, key=lambda item: item['similarity'])\n"
                "\n"
                "def retrieve(query):\n"
                "    candidates = vector_index.search(query)\n"
                "    return rank_candidates(candidates)[:5]\n",
                encoding="utf-8",
            )

            result = investigate_case(
                workspace,
                {
                    "question": "Higher similarity scores are better. The run shows a lower-scoring result ranked first. Identify the code defect and exact fix.",
                    "expected_answer": "The relevant document has similarity 0.91.",
                    "diagnosis": {
                        "source": "run_with_doctor",
                        "root_cause": "UNKNOWN",
                        "status": "needs_review",
                        "retrieval_observed": True,
                        "retrieval_count": 2,
                        "retrieved_chunks": [
                            {"chunk_id": "low", "doc_id": "distractor", "final_rank": 1, "pre_rerank_rank": 1, "dense_score": 0.12, "text": "unrelated passage"},
                            {"chunk_id": "high", "doc_id": "relevant", "final_rank": 2, "pre_rerank_rank": 2, "dense_score": 0.91, "text": "relevant fact"},
                        ],
                    },
                },
                MODEL_ENDPOINT,
                MODEL_NAME,
            )

        self._require_model_confirmation(result, "runtime_verifier")
        self.assertEqual(result["root_cause"], "RANKING_K", result)
        self.assertEqual(result["status"], "probable", result)
        self.assertTrue(result["runtime_verifier"]["finding_detected"], result["runtime_verifier"])
        self.assertTrue(result["runtime_verifier"]["trace"]["finding_source_excerpt_loaded"], result["runtime_verifier"])
        self.assertTrue(result["fix"] and result["fix"]["source_matches"], result["fix"])
        self.assertEqual(result["fix"]["syntax_status"], "valid", result["fix"])
        self.assertIn("reverse=True", result["fix"]["replacement_code"])
        self.assertTrue(all(item["verified"] for item in result["evidence"]), result["evidence"])


if __name__ == "__main__":
    unittest.main()
