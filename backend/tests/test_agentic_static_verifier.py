"""Offline checks for the deterministic side of the hybrid investigator."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.diagnose.agentic_investigator import _runtime_similarity_order_finding, _workspace_source_context, investigate_case


UNKNOWN_MODEL_RESULT = json.dumps({
    "final": {
        "status": "insufficient_evidence",
        "root_cause": "UNKNOWN",
        "summary": "The model could not identify a concrete issue.",
        "evidence": [],
        "fix": None,
        "verification": "Inspect the code manually.",
    }
})


class DeterministicStaticVerifierTests(unittest.TestCase):
    def _investigate(self, files):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for relative, source in files.items():
                target = workspace / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source, encoding="utf-8")
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=UNKNOWN_MODEL_RESULT,
            ):
                return investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "mock-local-model",
                )

    def _assert_grounded_fallback(self, result, cause):
        self.assertEqual(result["root_cause"], cause, result)
        self.assertEqual(result["status"], "probable", result)
        self.assertTrue(any(item["verified"] for item in result["evidence"]), result)
        self.assertTrue(result["fix"]["source_matches"], result)
        self.assertTrue(result["fix"]["replacement_code"], result)
        self.assertTrue(result["static_verifier"]["fallback_applied"], result)

    def test_generator_prompt_that_drops_context_gets_grounded_fix(self):
        result = self._investigate({
            "src/generator.py": (
                "def build_prompt(question, context):\n"
                "    return f\"Question: {question}\"\n"
            ),
        })
        self._assert_grounded_fallback(result, "GENERATOR")
        self.assertIn("context", result["fix"]["replacement_code"].lower())

    def test_zero_context_slice_is_reported_as_generator_fault(self):
        result = self._investigate({
            "src/generation.py": (
                "def build_context(chunks):\n"
                "    return '\\n'.join(chunk.text for chunk in chunks)[:0]\n"
            ),
        })
        self.assertEqual(result["root_cause"], "GENERATOR", result)
        self.assertEqual(result["status"], "probable")
        self.assertEqual(result["fix"]["line"], 2)
        self.assertIn("zero characters", result["summary"])

    def test_nonpositive_top_k_is_reported_without_claiming_optimal_value(self):
        result = self._investigate({
            "src/config.py": (
                "TOP_K_RETRIEVAL = 0\n"
                "def retrieve(store, query):\n"
                "    return store.search(query, top_k=TOP_K_RETRIEVAL)\n"
            ),
        })
        self.assertEqual(result["root_cause"], "RETRIEVAL", result)
        self.assertIn("minimum valid value", result["fix"]["proposed_change"])
        self.assertIn("tune", result["verification"])

    def test_model_receives_a_specific_static_candidate_to_review(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = workspace / "src" / "generator.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                "def build_prompt(question, context):\n"
                "    return f\"Question: {question}\"\n",
                encoding="utf-8",
            )
            first_request = {}

            def capture_request(_endpoint, _model, messages, **_kwargs):
                first_request.setdefault("messages", messages)
                return UNKNOWN_MODEL_RESULT

            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                side_effect=capture_request,
            ):
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "mock-local-model",
                )

        self.assertEqual(result["root_cause"], "GENERATOR", result)
        first_user_message = first_request["messages"][1]["content"]
        self.assertIn("STATIC VERIFIER CANDIDATE", first_user_message)
        self.assertIn('"root_cause":"GENERATOR"', first_user_message)
        self.assertIn('"replacement_code"', first_user_message)
        self.assertIn("code:src/generator.py:2", first_user_message)
        self.assertIn("return f\\\"Question: {question}\\\"", first_user_message)

    def test_unlabeled_generator_that_drops_retrieval_gets_grounded_fix(self):
        result = self._investigate({
            "src/pipeline.py": (
                "def fuse(a, b):\n"
                "    return f\"Question: {a}\"\n"
                "\n"
                "def execute(q):\n"
                "    z = store.search(q)\n"
                "    p = fuse(q, z)\n"
                "    return llm.invoke(p)\n"
            ),
        })
        self._assert_grounded_fallback(result, "GENERATOR")
        self.assertEqual(result["fix"]["file"], "src/pipeline.py")
        self.assertIn("{b}", result["fix"]["replacement_code"])
        self.assertIn("retrieval result is passed", result["summary"].lower())

    def test_missing_prompt_slot_restores_rendered_context_and_preserves_prompt(self):
        result = self._investigate({
            "src/generator.py": '''def build_prompt(question, retrieved_chunks):
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(chunk)
    context_str = "\\n".join(context_blocks)
    return (
        "Use only the retrieved passages."
        ""
        f"Question: {question}"
    )
''',
        })
        self._assert_grounded_fallback(result, "GENERATOR")
        self.assertEqual(result["fix"]["file"], "src/generator.py")
        self.assertEqual(result["fix"]["line"], 8)
        self.assertEqual(result["fix"]["current_code"].strip(), '""')
        self.assertIn("{context_str}", result["fix"]["replacement_code"])
        self.assertNotIn("{retrieved_chunks}", result["fix"]["replacement_code"])

    def test_context_empty_runtime_run_checks_static_generator_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = workspace / "src" / "generator.py"
            target.parent.mkdir(parents=True)
            target.write_text('''def build_prompt(question, retrieved_chunks):
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(chunk)
    context_str = "\\n".join(context_blocks)
    return (
        "Use only the retrieved passages."
        ""
        f"Question: {question}"
    )
''', encoding="utf-8")
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=UNKNOWN_MODEL_RESULT,
            ):
                result = investigate_case(
                    workspace,
                    {"question": "What does the filing say?", "diagnosis": {
                        "source": "run_with_doctor",
                        "root_cause": "RETRIEVAL",
                        "status": "needs_context_content",
                        "retrieval_observed": None,
                        "retrieval_count": None,
                        "retrieved_chunks": [],
                        "prompt_context_blocks": [],
                        "prompt": "System instructions.\\n\\nQuestion: What does the filing say?",
                        "llm_answer": "The answer is 42.",
                    }},
                    "http://127.0.0.1:11434/api/chat",
                    "mock-local-model",
                )

        self.assertEqual(result["root_cause"], "GENERATOR", result)
        self.assertTrue(result["static_verifier"]["checked"], result["static_verifier"])
        self.assertTrue(result["static_verifier"]["finding_detected"], result["static_verifier"])
        self.assertEqual(result["fix"]["line"], 8)
        self.assertIn("{context_str}", result["fix"]["replacement_code"])

    def test_opaque_nonprogressing_chunk_loop_gets_grounded_fix(self):
        result = self._investigate({
            "src/chunker.py": (
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
                "    return result\n"
            ),
        })
        self._assert_grounded_fallback(result, "CHUNKING")
        self.assertEqual(result["fix"]["file"], "src/chunker.py")
        self.assertIn("if A <= B:", result["fix"]["replacement_code"])
        self.assertIn("advances its cursor by 0", result["summary"])
        self.assertIn("`A` is 128", result["fix"]["proposed_change"])
        self.assertIn("`B` is 128", result["fix"]["proposed_change"])
        self.assertIn("intended overlap below `A`", result["fix"]["proposed_change"])

    def test_opaque_chunk_loop_with_positive_step_is_not_flagged(self):
        result = self._investigate({
            "src/chunker.py": (
                "A = 128\n"
                "B = 32\n"
                "\n"
                "def split(t):\n"
                "    step = A - B\n"
                "    result = []\n"
                "    i = 0\n"
                "    while i < len(t):\n"
                "        end = min(len(t), i + A)\n"
                "        result.append(t[i:end])\n"
                "        i += step\n"
                "    return result\n"
            ),
        })
        self.assertFalse(result["static_verifier"]["finding_detected"], result["static_verifier"])

    def test_chunk_loop_with_unobserved_parameter_step_is_not_flagged(self):
        result = self._investigate({
            "src/chunker.py": (
                "def split(t, a, b):\n"
                "    step = a - b\n"
                "    result = []\n"
                "    i = 0\n"
                "    while i < len(t):\n"
                "        end = min(len(t), i + a)\n"
                "        result.append(t[i:end])\n"
                "        i += step\n"
                "    return result\n"
            ),
        })
        self.assertFalse(result["static_verifier"]["finding_detected"], result["static_verifier"])

    def test_literal_zero_chunk_increment_gets_a_syntax_checked_guard(self):
        result = self._investigate({
            "src/chunker.py": (
                "def split(t):\n"
                "    result = []\n"
                "    i = 0\n"
                "    while i < len(t):\n"
                "        result.append(t[i:i + 16])\n"
                "        i += 0\n"
                "    return result\n"
            ),
        })
        self._assert_grounded_fallback(result, "CHUNKING")
        self.assertEqual(result["fix"]["syntax_status"], "valid")
        self.assertIn("if (0) <= 0:", result["fix"]["replacement_code"])

    def test_context_transformed_through_container_and_alias_is_not_a_drop(self):
        result = self._investigate({
            "src/generator.py": (
                "def build_prompt(question, context):\n"
                "    context_blocks = []\n"
                "    for passage in context:\n"
                "        context_blocks.append(f\"[{passage}]\")\n"
                "    context_text = \"\\n\".join(context_blocks)\n"
                "    return f\"Context: {context_text}\\nQuestion: {question}\"\n"
            ),
        })
        self.assertEqual(result["root_cause"], "UNKNOWN", result)
        self.assertFalse(result["static_verifier"]["finding_detected"], result["static_verifier"])

    def test_source_context_selects_relevant_functions_beyond_the_file_header(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source_dir = workspace / "src"
            source_dir.mkdir()
            source = "\n".join(f"constant_{index} = {index}" for index in range(140))
            source += (
                "\n\ndef query_pipeline(question):\n"
                "    retrieved = vector_store.search(question)\n"
                "    return generator.generate_answer(question, retrieved)\n"
            )
            (source_dir / "opaque.py").write_text(source, encoding="utf-8")

            _inventory, excerpts = _workspace_source_context(workspace, 4000)

        content = next(item["content"] for item in excerpts if item["path"] == "src/opaque.py")
        self.assertIn("142: def query_pipeline(question):", content)
        self.assertIn("144:     return generator.generate_answer(question, retrieved)", content)

    def test_source_context_selects_late_pipeline_code_in_typescript(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source_dir = workspace / "src"
            source_dir.mkdir()
            source = "\n".join(f"const unrelated_{index} = {index};" for index in range(140))
            source += (
                "\n\nexport async function queryRag(question: string) {\n"
                "  const retrieved = await vectorStore.similaritySearch(question);\n"
                "  const prompt = buildPrompt(question, retrieved);\n"
                "  return model.generateContent(prompt);\n"
                "}\n"
            )
            (source_dir / "pipeline.ts").write_text(source, encoding="utf-8")

            _inventory, excerpts = _workspace_source_context(workspace, 4000)

        content = next(item["content"] for item in excerpts if item["path"] == "src/pipeline.ts")
        self.assertIn("142: export async function queryRag(question: string)", content)
        self.assertIn("143:   const retrieved = await vectorStore.similaritySearch(question);", content)
        self.assertIn("145:   return model.generateContent(prompt);", content)

    def test_static_finding_survives_local_model_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = workspace / "src" / "generator.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                "def build_prompt(question, context):\n"
                "    return f\"Question: {question}\"\n",
                encoding="utf-8",
            )
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                side_effect=RuntimeError("The local model timed out."),
            ):
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "mock-local-model",
                )

        self.assertEqual(result["root_cause"], "GENERATOR", result)
        self.assertEqual(result["status"], "probable")
        self.assertTrue(result["static_verifier"]["fallback_applied"], result)
        self.assertIn("timed out", result["model_error"])
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertTrue(any(item["verified"] for item in result["evidence"]), result["evidence"])

    def test_chunk_step_uses_invalid_settings_from_separate_config_file(self):
        result = self._investigate({
            "src/config.py": "CHUNK_SIZE = 256\nCHUNK_OVERLAP = 256\n",
            "src/chunker.py": (
                "def split_text(text, chunk_size, chunk_overlap):\n"
                "    step = chunk_size - chunk_overlap\n"
                "    chunks = []\n"
                "    start = 0\n"
                "    while start < len(text):\n"
                "        end = min(len(text), start + chunk_size)\n"
                "        chunks.append(text[start:end])\n"
                "        start += step\n"
                "    return chunks\n"
            ),
        })
        self._assert_grounded_fallback(result, "CHUNKING")
        self.assertIn("chunk_overlap >= chunk_size", result["fix"]["replacement_code"])

    def test_reversed_chunk_step_with_clamped_cursor_gets_grounded_fix(self):
        result = self._investigate({
            "src/chunker.py": (
                "CHUNK_SIZE = 256\n"
                "CHUNK_OVERLAP = 32\n"
                "\n"
                "def split_text(text, chunk_size, chunk_overlap):\n"
                "    advance = chunk_overlap - chunk_size\n"
                "    chunks = []\n"
                "    position = 0\n"
                "    while position < len(text):\n"
                "        end = min(len(text), position + chunk_size)\n"
                "        chunks.append(text[position:end])\n"
                "        position = max(0, position + advance)\n"
                "    return chunks\n"
            ),
        })
        self._assert_grounded_fallback(result, "CHUNKING")
        self.assertIn("advance = chunk_size - chunk_overlap", result["fix"]["replacement_code"])

    def test_repeated_context_in_question_field_gets_grounded_fix(self):
        result = self._investigate({
            "src/generator.py": (
                "def build_prompt(question, retrieved_documents):\n"
                "    context = \"\\n\".join(retrieved_documents)\n"
                "    return f\"Context: {context}\\nQuestion: {context}\"\n"
            ),
        })
        self._assert_grounded_fallback(result, "GENERATOR")
        self.assertIn("Question: {question}", result["fix"]["replacement_code"])

    def test_typescript_prompt_misbinding_is_grounded_and_fixed(self):
        result = self._investigate({
            "src/pipeline.ts": (
                "export function buildPrompt(question: string, passages: string[]): string {\n"
                "  const context = passages.join('\\n');\n"
                "  return `Context:\\n${question}\\n\\nQuestion: ${question}`;\n"
                "}\n"
            ),
        })
        self._assert_grounded_fallback(result, "GENERATOR")
        self.assertEqual(result["fix"]["file"], "src/pipeline.ts")
        self.assertIn("${context}", result["fix"]["replacement_code"])
        self.assertNotIn("topK", result["fix"]["replacement_code"])

    def test_model_top_k_hallucination_is_overridden_by_verified_prompt_finding(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "pipeline.ts"
            source.parent.mkdir()
            current_retrieval = "  const passages = await vectorStore.similaritySearch(question);"
            source.write_text(
                "export function buildPrompt(question: string, passages: string[]): string {\n"
                "  const context = passages.join('\\n');\n"
                "  return `Context:\\n${question}\\n\\nQuestion: ${question}`;\n"
                "}\n"
                "\n"
                "export async function answerQuestion(question: string) {\n"
                f"{current_retrieval}\n"
                "  const prompt = buildPrompt(question, passages);\n"
                "  return await model.generateContent(prompt);\n"
                "}\n",
                encoding="utf-8",
            )
            model_hallucination = json.dumps({"final": {
                "status": "probable",
                "root_cause": "RETRIEVAL",
                "summary": "The vector store may retrieve too many passages by default.",
                "evidence": [{"source": "code:src/pipeline.ts:7", "quote": current_retrieval.strip()}],
                "fix": {
                    "file": "src/pipeline.ts",
                    "line": 7,
                    "end_line": 7,
                    "current_code": current_retrieval,
                    "replacement_code": "  const passages = await vectorStore.similaritySearch(question, { topK: 3 });",
                    "proposed_change": "Add a topK limit.",
                },
            }})
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=model_hallucination,
            ):
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "mock-local-model",
                )

        self.assertEqual(result["root_cause"], "GENERATOR", result)
        self.assertTrue(result["static_verifier"]["fallback_applied"], result["static_verifier"])
        self.assertTrue(result["model_did_not_confirm_static_check"])
        self.assertEqual(result["fix"]["line"], 3)
        self.assertIn("${context}", result["fix"]["replacement_code"])
        self.assertNotIn("topK", result["fix"]["replacement_code"])
        self.assertTrue(any(item["verified"] for item in result["evidence"]), result["evidence"])

    def test_retrieved_value_dropped_by_builder_is_traced_without_semantic_variable_names(self):
        result = self._investigate({
            "src/pipeline.py": (
                "def make_request(text, auxiliary):\n"
                "    return f\"Question: {text}\"\n"
                "\n"
                "def run_query(question):\n"
                "    evidence = vector_db.similarity_search(question, k=5)\n"
                "    prompt = make_request(question, evidence)\n"
                "    return llm.invoke(prompt)\n"
            ),
        })
        self._assert_grounded_fallback(result, "GENERATOR")
        self.assertIn("auxiliary", result["fix"]["replacement_code"])
        self.assertGreaterEqual(len([item for item in result["evidence"] if item["verified"]]), 2)

    def test_retrieval_to_prompt_flow_is_traced_across_python_modules(self):
        result = self._investigate({
            "src/__init__.py": "",
            "src/prompts.py": (
                "def make_request(text, auxiliary):\n"
                "    return f\"Question: {text}\"\n"
            ),
            "src/pipeline.py": (
                "from .prompts import make_request as build_request\n"
                "\n"
                "def run_query(question):\n"
                "    evidence = vector_db.similarity_search(question, k=5)\n"
                "    prompt = build_request(question, evidence)\n"
                "    return llm.invoke(prompt)\n"
            ),
        })
        self._assert_grounded_fallback(result, "GENERATOR")
        self.assertEqual(result["fix"]["file"], "src/prompts.py")
        self.assertIn("auxiliary", result["fix"]["replacement_code"])
        verified = [item for item in result["evidence"] if item["verified"]]
        self.assertGreaterEqual(len(verified), 3, result["evidence"])

    def test_numeric_evaluator_that_accepts_any_answer_number_gets_specific_warning(self):
        result = self._investigate({
            "tests/run_benchmark.py": (
                "def evaluate_match(gold_raw, answer_text):\n"
                "    gold_clean = gold_raw.replace(',', '').strip().lower()\n"
                "    ans_clean = answer_text.replace(',', '').strip().lower()\n"
                "    if gold_clean and gold_clean in ans_clean:\n"
                "        return True\n"
                "    gold_nums = extract_all_numbers(gold_raw)\n"
                "    answer_nums = extract_all_numbers(answer_text)\n"
                "    if gold_nums and answer_nums:\n"
                "        target = gold_nums[0]\n"
                "        for candidate in answer_nums:\n"
                "            if abs(target - candidate) <= max(0.15, abs(target * 0.02)):\n"
                "                return True\n"
                "    return False\n"
                "\n"
                "def run_case(question, gold_raw, answer_text):\n"
                "    return evaluate_match(gold_raw, answer_text)\n"
            ),
        })
        self.assertEqual(result["root_cause"], "EVALUATION", result)
        self.assertEqual(result["status"], "probable")
        self.assertTrue(result["static_verifier"]["fallback_applied"], result)
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertTrue(any(item["verified"] for item in result["evidence"]), result["evidence"])
        self.assertEqual(result["fix"]["file"], "tests/run_benchmark.py")
        self.assertEqual(result["fix"]["replacement_code"], "")
        self.assertIn("requested metric", result["fix"]["proposed_change"])
        self.assertIn("unit", result["fix"]["proposed_change"])
        self.assertIn("evaluate_match` to accept `(question, gold_raw, answer_text)`", result["fix"]["proposed_change"])
        self.assertIn("pass `question` as the first argument", result["fix"]["proposed_change"])
        self.assertTrue(any(
            item["verified"] and "evaluate_match(gold_raw, answer_text)" in item["quote"]
            for item in result["evidence"]
        ))

    def test_substring_evaluator_that_accepts_a_negated_gold_value_is_flagged(self):
        result = self._investigate({
            "tests/evaluation.py": (
                "def evaluate_match(gold_raw, answer_text):\n"
                "    gold_clean = gold_raw.strip().lower()\n"
                "    answer_clean = answer_text.strip().lower()\n"
                "    if gold_clean and gold_clean in answer_clean:\n"
                "        return True\n"
                "    return False\n"
            ),
        })
        self.assertEqual(result["root_cause"], "EVALUATION", result)
        self.assertEqual(result["status"], "probable")
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertEqual(result["fix"]["line"], 4)
        self.assertIn("negated", result["fix"]["proposed_change"])

    def test_runtime_rank_score_mismatch_gets_verified_ranking_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "retriever.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "def rank_candidates(candidates):\n"
                "    return sorted(candidates, key=lambda item: item['similarity'])\n",
                encoding="utf-8",
            )
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=UNKNOWN_MODEL_RESULT,
            ):
                result = investigate_case(
                    workspace,
                    {
                        "question": "Higher similarity scores are better, but the lower-scoring result is ranked first.",
                        "diagnosis": {
                            "source": "run_with_doctor",
                            "retrieval_observed": True,
                            "retrieval_count": 2,
                            "retrieved_chunks": [
                                {"chunk_id": "low", "doc_id": "distractor", "final_rank": 1, "dense_score": 0.12},
                                {"chunk_id": "high", "doc_id": "relevant", "final_rank": 2, "dense_score": 0.91},
                            ],
                        },
                    },
                    "http://127.0.0.1:11434/api/chat",
                    "mock-local-model",
                )

        self.assertEqual(result["root_cause"], "RANKING_K", result)
        self.assertEqual(result["status"], "probable")
        self.assertTrue(result["runtime_verifier"]["finding_detected"], result["runtime_verifier"])
        self.assertTrue(result["runtime_verifier"]["fallback_applied"], result["runtime_verifier"])
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertEqual(result["fix"]["syntax_status"], "valid")
        self.assertIn("reverse=True", result["fix"]["replacement_code"])
        self.assertTrue(any(
            item["source"] == "runtime:retrieval_summary" and item["verified"]
            for item in result["evidence"]
        ), result["evidence"])
        self.assertTrue(any(
            item["source"] == "code:src/retriever.py:2" and item["verified"]
            for item in result["evidence"]
        ), result["evidence"])

    def test_runtime_rank_verifier_does_not_flag_high_score_at_rank_one(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "retriever.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "def rank_candidates(candidates):\n"
                "    return sorted(candidates, key=lambda item: item['similarity'])\n",
                encoding="utf-8",
            )
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=UNKNOWN_MODEL_RESULT,
            ):
                result = investigate_case(
                    workspace,
                    {
                        "question": "Higher similarity scores are better, but the lower-scoring result is ranked first.",
                        "diagnosis": {
                            "source": "run_with_doctor",
                            "retrieval_observed": True,
                            "retrieval_count": 2,
                            "retrieved_chunks": [
                                {"chunk_id": "high", "doc_id": "relevant", "final_rank": 1, "dense_score": 0.91},
                                {"chunk_id": "low", "doc_id": "distractor", "final_rank": 2, "dense_score": 0.12},
                            ],
                        },
                    },
                    "http://127.0.0.1:11434/api/chat",
                    "mock-local-model",
                )

        self.assertFalse(result["runtime_verifier"]["finding_detected"], result["runtime_verifier"])
        self.assertEqual(result["root_cause"], "UNKNOWN", result)

    def test_runtime_rank_finding_replaces_explicit_reverse_false(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "retriever.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "def rank_candidates(candidates):\n"
                "    return sorted(candidates, key=lambda item: item['similarity'], reverse=False)\n",
                encoding="utf-8",
            )
            diagnostics = {}
            finding = _runtime_similarity_order_finding(workspace, {
                "question": "Higher similarity scores are better, but a lower-scoring item was ranked first.",
                "retrieved_chunks": [
                    {"rank": 1, "dense_score": 0.12},
                    {"rank": 2, "dense_score": 0.91},
                ],
                "retrieval_summary": "rank=1 dense_score=0.12; rank=2 dense_score=0.91",
            }, diagnostics)

        self.assertIsNotNone(finding)
        self.assertIn("reverse=True", finding["fix"]["replacement_code"])
        self.assertNotIn("reverse=False", finding["fix"]["replacement_code"])

    def test_runtime_rank_verifier_ignores_unrelated_similarity_sort(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "reporting.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "def format_similarity(items):\n"
                "    return sorted(items, key=lambda item: item['similarity'])\n",
                encoding="utf-8",
            )
            finding = _runtime_similarity_order_finding(workspace, {
                "question": "Higher similarity scores are better, but a lower-scoring item was ranked first.",
                "retrieved_chunks": [
                    {"rank": 1, "dense_score": 0.12},
                    {"rank": 2, "dense_score": 0.91},
                ],
                "retrieval_summary": "rank=1 dense_score=0.12; rank=2 dense_score=0.91",
            }, {})

        self.assertIsNone(finding)


if __name__ == "__main__":
    unittest.main()
