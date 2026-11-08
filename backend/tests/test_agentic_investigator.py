import json
import tempfile
import unittest
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request

from backend.app.diagnose.agentic_investigator import (
    LocalModelNoContentError,
    _NoRedirectHandler,
    _inside_workspace,
    _call_local_model,
    _initial_evidence,
    _list_files_tool,
    _local_chat_url,
    _local_model_opener,
    _model_evidence_packet,
    _search_tool,
    _validate_final,
    investigate_case,
)
from backend.app.diagnose.arithmetic_audit import audit_arithmetic


class ArithmeticAuditTests(unittest.TestCase):
    def test_finds_wrong_average_and_ratio_in_answer(self):
        answer = r"The average is $\frac{\$253 + \$282}{2} = \$266.5$. Ratio $\frac{\$6,489}{\$266.5} \approx 24.39$."
        findings = audit_arithmetic(answer)
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0]["computed"], "267.5")
        self.assertEqual(findings[0]["reported"], "266.5")
        self.assertEqual(findings[1]["reported"], "24.39")

    def test_accepts_correctly_rounded_equations(self):
        answer = r"Average $\frac{253 + 282}{2} = 267.5$. Ratio $\frac{6489}{267.5} \approx 24.26$."
        self.assertEqual(audit_arithmetic(answer), [])


class LocalAgentSafetyTests(unittest.TestCase):
    def test_only_loopback_http_endpoints_are_allowed(self):
        self.assertEqual(_local_chat_url("http://127.0.0.1:11434/api/chat"), "http://127.0.0.1:11434/api/chat")
        for endpoint in (
            "https://127.0.0.1:11434/api/chat",
            "http://example.com:11434/api/chat",
            "http://user:pass@127.0.0.1:11434/api/chat",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                _local_chat_url(endpoint)

    def test_workspace_tool_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with self.assertRaises(ValueError):
                _inside_workspace(workspace, "../outside.py")

    def test_file_inventory_is_bounded_to_inspectable_workspace_files(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "src").mkdir()
            (workspace / "src" / "opaque.py").write_text("limit = 1", encoding="utf-8")
            (workspace / "src" / "RagService.java").write_text(
                'class RagService { String prompt = "retrieval context"; }', encoding="utf-8"
            )
            (workspace / "src" / "pipeline.go").write_text(
                'package rag\nvar prompt = "retrieval context"\n', encoding="utf-8"
            )
            (workspace / "src" / "ranker.rs").write_text(
                'fn rank() { let prompt = "retrieval context"; }\n', encoding="utf-8"
            )
            (workspace / "logs").mkdir()
            (workspace / "logs" / "run.log").write_text('{"event":"retrieval","count":0}', encoding="utf-8")
            (workspace / ".hidden").mkdir()
            (workspace / ".hidden" / "ignored.py").write_text("secret = True", encoding="utf-8")
            (workspace / ".env").write_text("TOKEN=private", encoding="utf-8")
            result = _list_files_tool(workspace, {"path": "."})
            listed = {item["path"] for item in result["files"]}
            self.assertTrue({
                "logs/run.log", "src/opaque.py", "src/RagService.java", "src/pipeline.go", "src/ranker.rs",
            }.issubset(listed))
            self.assertFalse(result["truncated"])
            matches = _search_tool(workspace, {"query": "retrieval context"})["matches"]
            matched_paths = {item["path"] for item in matches}
            self.assertTrue({"src/RagService.java", "src/pipeline.go", "src/ranker.rs"}.issubset(matched_paths))
            self.assertNotIn(".env", matched_paths)

    def test_investigation_includes_only_allowlisted_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence, _ = _initial_evidence(workspace, {"diagnosis": {
                "llm_answer": "Answer",
                "retrieval_observed": True,
                "retrieval_count": 4,
                "metadata": {
                    "provider": "google",
                    "model": "gemini-local-app",
                    "input_tokens": 220,
                    "latency_ms": 81.5,
                    "streaming": True,
                    "capture_truncated": False,
                    "api_key": "must-not-be-forwarded",
                },
            }})
            self.assertEqual(evidence["baseline_diagnosis"]["retrieval_count"], 4)
            self.assertEqual(evidence["runtime_metadata"]["provider"], "google")
            self.assertEqual(evidence["runtime_metadata"]["input_tokens"], 220)
            self.assertTrue(evidence["runtime_metadata"]["streaming"])
            self.assertFalse(evidence["runtime_metadata"]["capture_truncated"])
            self.assertNotIn("api_key", evidence["runtime_metadata"])

    def test_model_evidence_is_valid_bounded_json_and_keeps_verified_code(self):
        evidence = {
            "runtime_capture_available": True,
            "case_id": "large-run",
            "question": "question " * 1500,
            "expected_answer": "expected " * 1000,
            "answer": "answer " * 3000,
            "prompt": "prompt " * 3000,
            "retrieval_summary": "rank and score " * 500,
            "baseline_diagnosis": {"root_cause": "UNKNOWN", "status": "needs_review"},
            "runtime_metadata": {"provider": "local", "latency_ms": 42},
            "retrieved_chunks": [{"chunk_id": f"c{index}", "rank": index + 1, "text": "passage " * 500} for index in range(10)],
            "prompt_context_blocks": [{"label": "context", "text": "context block " * 300} for _ in range(6)],
            "deterministic_static_check": {
                "root_cause": "GENERATOR",
                "summary": "The prompt builder drops retrieved context.",
                "evidence": [{"source": "code:src/generator.py:2", "quote": "return f'Question: {question}'"}],
                "fix": {"file": "src/generator.py", "line": 2, "current_code": "return f'Question: {question}'", "replacement_code": "return f'Context: {context}\\nQuestion: {question}'"},
            },
            "code_excerpts": [{
                "file": "src/generator.py",
                "reported_line": 2,
                "origin": "verified_finding",
                "excerpt": "1: def build_prompt(question, context):\\n2: return f'Question: {question}'\\n3: return prompt" + (" extra" * 1000),
            }],
        }

        packet = _model_evidence_packet(evidence)
        serialized = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
        reparsed = json.loads(serialized)

        self.assertLessEqual(len(serialized), 36000)
        self.assertEqual(reparsed["deterministic_static_check"]["root_cause"], "GENERATOR")
        self.assertIn("Question: {question}", reparsed["code_excerpts"][0]["excerpt"])
        self.assertIn("truncated_fields", reparsed)
        self.assertIn("prompt", reparsed["truncated_fields"])
        self.assertIn("answer", reparsed["truncated_fields"])

    def test_investigation_preserves_rank_scores_and_scrubs_run_content(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence, _ = _initial_evidence(workspace, {
                "question": "Check customer@example.com",
                "diagnosis": {
                    "llm_answer": "Contact customer@example.com",
                    "prompt": "Bearer sk-123456789012345678901234",
                    "evidence": ["Source log mentions customer@example.com and sk-123456789012345678901234"],
                    "retrieval_observed": True,
                    "retrieved_chunks": [
                        {
                            "chunk_id": "sk-123456789012345678901234",
                            "doc_id": "customer@example.com",
                            "final_rank": 1,
                            "pre_rerank_rank": 2,
                            "dense_score": 0.12,
                            "bm25_score": 1.25,
                            "rerank_score": 0.31,
                            "text": "Email customer@example.com",
                            "api_key": "must-not-be-forwarded",
                        },
                    ],
                },
            })

        chunk = evidence["retrieved_chunks"][0]
        self.assertEqual(chunk["rank"], 1)
        self.assertEqual(chunk["pre_rerank_rank"], 2)
        self.assertEqual(chunk["dense_score"], 0.12)
        self.assertEqual(chunk["bm25_score"], 1.25)
        self.assertEqual(chunk["rerank_score"], 0.31)
        serialized = json.dumps(evidence)
        self.assertNotIn("customer@example.com", serialized)
        self.assertNotIn("sk-123456789012345678901234", serialized)
        self.assertNotIn("must-not-be-forwarded", serialized)
        self.assertIn("[REDACTED_EMAIL]", serialized)
        self.assertIn("[REDACTED_API_KEY]", serialized)

    def test_unlabeled_latest_run_uses_reported_code_and_log_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "src").mkdir()
            (workspace / "logs").mkdir()
            (workspace / "src" / "pipeline.py").write_text("results = retrieve(question)\n", encoding="utf-8")
            (workspace / "logs" / "rag.log").write_text('{"retrieval_count":0}\n', encoding="utf-8")
            evidence, _ = _initial_evidence(workspace, {"diagnosis": {
                "question": "Where did retrieval fail?",
                "llm_answer": "No answer.",
                "source_location": {"file": "src/pipeline.py", "line": 1},
                "log_location": {"file": "logs/rag.log", "line": 1},
            }})
            self.assertEqual({item["file"] for item in evidence["code_excerpts"]}, {"src/pipeline.py", "logs/rag.log"})

    def test_static_audit_does_not_mark_uncaptured_workspace_healthy(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence, code_sources = _initial_evidence(workspace, {"diagnosis": {}})
            self.assertFalse(evidence["runtime_capture_available"])
            result = _validate_final(workspace, {"final": {
                "status": "confirmed",
                "root_cause": "NO_FAILURE",
                "summary": "No defect was found in the inspected files.",
                "evidence": [],
                "fix": None,
                "verification": "Capture and evaluate a live run.",
            }}, evidence, code_sources)
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertEqual(result["root_cause"], "UNKNOWN")

    def test_static_workspace_audit_can_find_a_code_defect_without_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "retriever.py"
            source.parent.mkdir()
            source.write_text("top_k = 0\nresults = search(query, top_k=top_k)\n", encoding="utf-8")
            responses = [
                json.dumps({"actions": [{"tool": "list_files", "path": "src"}]}),
                json.dumps({"actions": [{"tool": "read_file", "path": "src/retriever.py", "start_line": 1, "end_line": 2}]}),
                json.dumps({"final": {
                    "status": "probable",
                    "root_cause": "RETRIEVAL",
                    "summary": "The configured candidate count is zero.",
                    "evidence": [{"source": "code:src/retriever.py:1", "quote": "top_k = 0"}],
                    "fix": {
                        "file": "src/retriever.py",
                        "line": 1,
                        "end_line": 1,
                        "current_code": "top_k = 0",
                        "replacement_code": "top_k = 5",
                        "proposed_change": "Use a positive candidate count, then verify it on a labeled suite.",
                    },
                    "verification": "Replay the same local cases and compare retrieval support and latency.",
                }}),
            ]
            with patch("backend.app.diagnose.agentic_investigator._call_local_model", side_effect=responses) as model_call:
                result = investigate_case(workspace, {"diagnosis": {
                    "workspace_code_references": [{"file": "src/retriever.py", "line": 1, "label": "retriever"}],
                }}, "http://127.0.0.1:11434/api/chat", "local-test-model")
            self.assertIn("static audit", model_call.call_args_list[0].args[2][1]["content"])
            self.assertIn("src/retriever.py", model_call.call_args_list[0].args[2][1]["content"])
            self.assertEqual(result["root_cause"], "RETRIEVAL")
            self.assertEqual(result["fix"]["file"], "src/retriever.py")
            self.assertTrue(result["fix"]["source_matches"])
            self.assertEqual(result["fix"]["replacement_code"], "top_k = 1")
            self.assertTrue(result["evidence"][0]["verified"])

    def test_verified_static_finding_source_is_given_to_local_model(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "tests" / "run_benchmark.py"
            source.parent.mkdir()
            source_text = (
                "def evaluate_match(gold_raw, answer_text):\n"
                "    gold_clean = gold_raw.strip().lower()\n"
                "    answer_clean = answer_text.strip().lower()\n"
                "    if gold_clean and gold_clean in answer_clean:\n"
                "        return True\n"
                "    return False\n"
            )
            source_text += "\n".join(f"# unrelated helper {index}" for index in range(50))
            source_text += (
                "\n\ndef run_case(question, gold_raw, answer_text):\n"
                "    return evaluate_match(gold_raw, answer_text)\n"
            )
            source.write_text(source_text, encoding="utf-8")
            model_result = json.dumps({"final": {
                "status": "insufficient_evidence",
                "root_cause": "UNKNOWN",
                "summary": "The local model did not isolate a supported defect.",
                "evidence": [],
                "fix": None,
            }})
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=model_result,
            ) as model_call:
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "local-test-model",
                )

        model_context = model_call.call_args.args[2][1]["content"]
        self.assertIn('"file": "tests/run_benchmark.py"', model_context)
        self.assertIn("gold_clean and gold_clean in answer_clean", model_context)
        self.assertIn("def run_case(question, gold_raw, answer_text):", model_context)
        self.assertIn("return evaluate_match(gold_raw, answer_text)", model_context)
        self.assertIn("tests/run_benchmark.py", result["agent_trace"]["source_files_provided"])
        self.assertTrue(result["static_verifier"]["finding_detected"], result["static_verifier"])
        self.assertEqual(result["root_cause"], "EVALUATION", result)
        self.assertTrue(result["fix"]["source_matches"], result["fix"])

    def test_model_gets_one_focused_review_when_it_misses_verified_finding(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "tests" / "run_benchmark.py"
            source.parent.mkdir()
            source.write_text(
                "def evaluate_match(gold_raw, answer_text):\n"
                "    gold_clean = gold_raw.strip().lower()\n"
                "    answer_clean = answer_text.strip().lower()\n"
                "    if gold_clean and gold_clean in answer_clean:\n"
                "        return True\n"
                "    return False\n"
                "\n"
                "def run_case(question, gold_raw, answer_text):\n"
                "    return evaluate_match(gold_raw, answer_text)\n",
                encoding="utf-8",
            )
            unknown = json.dumps({"final": {
                "status": "insufficient_evidence",
                "root_cause": "UNKNOWN",
                "summary": "No defect was found.",
                "evidence": [],
                "fix": None,
            }})
            supported = json.dumps({"final": {
                "status": "probable",
                "root_cause": "EVALUATION",
                "summary": "The evaluator accepts a substring match without linking it to the question's requested claim.",
                "evidence": [{
                    "source": "code:tests/run_benchmark.py:4",
                    "quote": "if gold_clean and gold_clean in answer_clean:",
                }],
                "fix": {
                    "file": "tests/run_benchmark.py",
                    "line": 4,
                    "end_line": 5,
                    "current_code": "    if gold_clean and gold_clean in answer_clean:\n        return True",
                    "replacement_code": "",
                    "proposed_change": "Compare the answer's claim for the requested metric, including polarity and unit, with the labeled value.",
                    "why": "Substring presence does not prove that the answer asserts the requested fact.",
                },
                "verification": "Reject a negated gold value and a gold value attached to the wrong metric.",
            }})
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                side_effect=[unknown, supported],
            ) as model_call:
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "local-test-model",
                )

        self.assertEqual(model_call.call_count, 2)
        retry_prompt = model_call.call_args_list[1].args[2][-1]["content"]
        self.assertIn("Independently review the exact source excerpts", retry_prompt)
        self.assertIn("tests/run_benchmark.py", retry_prompt)
        self.assertTrue(result["agent_trace"]["verifier_review_retry_used"], result["agent_trace"])
        self.assertFalse(result["static_verifier"]["fallback_applied"], result["static_verifier"])
        self.assertEqual(result["root_cause"], "EVALUATION", result)
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertTrue(any(item["verified"] for item in result["evidence"]), result["evidence"])

    def test_inconclusive_static_model_result_gets_one_focused_source_review(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "prompt.ts"
            source.parent.mkdir()
            current = "  return `Context: ${context} Question: ${context}`;"
            replacement = "  return `Context: ${context} Question: ${question}`;"
            source.write_text(
                "export function makePrompt(question: string, context: string): string {\n"
                f"{current}\n"
                "}\n",
                encoding="utf-8",
            )
            inconclusive = json.dumps({"final": {
                "status": "insufficient_evidence",
                "root_cause": "UNKNOWN",
                "summary": "No runtime capture available for analysis. Static audit found no issue.",
                "evidence": [],
                "fix": None,
            }})
            grounded = json.dumps({"final": {
                "status": "probable",
                "root_cause": "GENERATOR",
                "summary": "The question section repeats context instead of using the question argument.",
                "evidence": [{"source": "code:src/prompt.ts:2", "quote": current}],
                "fix": {
                    "file": "src/prompt.ts",
                    "line": 2,
                    "end_line": 2,
                    "current_code": current,
                    "replacement_code": replacement,
                    "proposed_change": "Interpolate question in the question section.",
                    "why": "The question and retrieved context are distinct inputs.",
                },
                "verification": "Assert the assembled prompt contains context and question in their respective sections.",
            }})
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                side_effect=[inconclusive, grounded],
            ) as model_call:
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "local-test-model",
                )

        self.assertEqual(model_call.call_count, 2)
        self.assertIn("Review the supplied source excerpts once more", model_call.call_args_list[1].args[2][-1]["content"])
        self.assertTrue(result["agent_trace"]["static_review_retry_used"])
        self.assertEqual(result["root_cause"], "GENERATOR")
        self.assertTrue(result["fix"]["source_matches"])
        self.assertTrue(result["evidence"][0]["verified"])

    def test_verified_chunking_finding_overrides_unknown_local_model_result(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "chunker.py"
            source.parent.mkdir()
            source.write_text(
                "CHUNK_SIZE = 256\n"
                "CHUNK_OVERLAP = 256\n"
                "\n"
                "def split_text(text, chunk_size, chunk_overlap):\n"
                "    step = chunk_size - chunk_overlap\n"
                "    chunks = []\n"
                "    start = 0\n"
                "    while start < len(text):\n"
                "        end = min(len(text), start + chunk_size)\n"
                "        chunks.append(text[start:end])\n"
                "        start += step\n"
                "    return chunks\n",
                encoding="utf-8",
            )
            unknown_model_result = json.dumps({"final": {
                "status": "insufficient_evidence",
                "root_cause": "UNKNOWN",
                "summary": "No runtime capture available for analysis. Static audit performed but no concrete defects found in code configuration.",
                "evidence": [],
                "fix": None,
            }})
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=unknown_model_result,
            ):
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "local-test-model",
                )

        self.assertEqual(result["root_cause"], "CHUNKING", result)
        self.assertTrue(result["static_verifier"]["finding_detected"], result["static_verifier"])
        self.assertTrue(result["static_verifier"]["fallback_applied"], result["static_verifier"])
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertIn("chunk_overlap", result["fix"]["replacement_code"])
        self.assertEqual(result["model_assessment_before_verification"]["root_cause"], "UNKNOWN")
        self.assertIn("no concrete defects found", result["model_assessment_before_verification"]["summary"])

    def test_verified_generator_finding_overrides_unknown_local_model_result(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "generator.py"
            source.parent.mkdir()
            source.write_text(
                'def build_prompt(question, context):\n'
                '    return f"Question: {question}"\n',
                encoding="utf-8",
            )
            unknown_model_result = json.dumps({"final": {
                "status": "insufficient_evidence",
                "root_cause": "UNKNOWN",
                "summary": "No runtime capture available for analysis. Static audit performed but no concrete defects found in code configuration.",
                "evidence": [],
                "fix": None,
            }})
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=unknown_model_result,
            ):
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "local-test-model",
                )

        self.assertEqual(result["root_cause"], "GENERATOR", result)
        self.assertTrue(result["static_verifier"]["finding_detected"], result["static_verifier"])
        self.assertTrue(result["static_verifier"]["fallback_applied"], result["static_verifier"])
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertIn("context", result["fix"]["replacement_code"].lower())
        self.assertEqual(result["agent_trace"]["source_files_provided"], ["src/generator.py"])
        self.assertEqual(result["model_assessment_before_verification"]["root_cause"], "UNKNOWN")

    def test_static_verifier_rejects_a_different_model_fix_for_the_same_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "generator.py"
            source.parent.mkdir()
            current_code = '    return f"Question: {question}"'
            source.write_text(
                'def build_prompt(question, context):\n'
                f'{current_code}\n',
                encoding="utf-8",
            )
            model_result = json.dumps({"final": {
                "status": "probable",
                "root_cause": "GENERATOR",
                "summary": "The context argument is not included in the prompt.",
                "evidence": [{
                    "source": "code:src/generator.py:2",
                    "quote": current_code.strip(),
                }],
                "fix": {
                    "file": "src/generator.py",
                    "line": 2,
                    "end_line": 2,
                    "current_code": current_code,
                    "replacement_code": current_code,
                    "proposed_change": "Keep the question-only prompt.",
                },
                "verification": "Check that the prompt contains the retrieved context.",
            }})
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=model_result,
            ):
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "local-test-model",
                )

        self.assertEqual(result["root_cause"], "GENERATOR", result)
        self.assertTrue(result["static_verifier"]["fallback_applied"], result["static_verifier"])
        self.assertTrue(result["model_confirmed_deterministic_check"])
        self.assertFalse(result.get("model_did_not_confirm_static_check", False))
        self.assertTrue(result["model_fix_replaced_by_verified_fix"])
        self.assertIn("context", result["fix"]["replacement_code"].lower())

    def test_model_fix_for_typescript_is_cited_and_matches_static_finding(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "pipeline.ts"
            source.parent.mkdir()
            current_code = "  return `Context:\\n${question}\\n\\nQuestion: ${question}`;"
            source.write_text(
                "export function buildPrompt(question: string, passages: string[]): string {\n"
                "  const context = passages.join('\\n');\n"
                f"{current_code}\n"
                "}\n"
                "\n"
                "export async function answerQuestion(question: string) {\n"
                "  const passages = await vectorStore.similaritySearch(question);\n"
                "  const prompt = buildPrompt(question, passages);\n"
                "  return await model.generateContent(prompt);\n"
                "}\n",
                encoding="utf-8",
            )
            replacement = "  return `Context:\\n${context}\\n\\nQuestion: ${question}`;"
            model_result = json.dumps({"final": {
                "status": "probable",
                "root_cause": "GENERATOR",
                "summary": "The prompt interpolates the question where the assembled retrieved context belongs.",
                "evidence": [{"source": "code:src/pipeline.ts:3", "quote": current_code.strip()}],
                "fix": {
                    "file": "src/pipeline.ts",
                    "line": 3,
                    "end_line": 3,
                    "current_code": current_code,
                    "replacement_code": replacement,
                    "proposed_change": "Interpolate the joined retrieved passages into the context section.",
                },
                "verification": "Assert that a distinctive retrieved passage appears in the prompt passed to generateContent.",
            }})
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                return_value=model_result,
            ) as model_call:
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "local-test-model",
                )
            initial_user_message = model_call.call_args.args[2][1]["content"]

        self.assertEqual(result["root_cause"], "GENERATOR", result)
        self.assertTrue(result["static_verifier"]["finding_detected"], result["static_verifier"])
        self.assertFalse(result["static_verifier"]["fallback_applied"], result["static_verifier"])
        self.assertTrue(result["fix"]["source_matches"], result["fix"])
        self.assertEqual(result["fix"]["replacement_code"], replacement)
        self.assertTrue(result["evidence"][0]["verified"], result["evidence"])
        self.assertIn("src/pipeline.ts", result["agent_trace"]["source_files_provided"])
        self.assertIn("src/pipeline.ts", initial_user_message)
        self.assertIn("${question}", initial_user_message)

    def test_copy_ready_replacement_is_suppressed_when_current_block_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "retriever.py").write_text("top_k = 5\n", encoding="utf-8")
            evidence = {"runtime_capture_available": False, "deterministic_arithmetic_audit": []}
            result = _validate_final(workspace, {"final": {
                "status": "probable",
                "root_cause": "RETRIEVAL",
                "summary": "Candidate configuration may be wrong.",
                "evidence": [],
                "fix": {
                    "file": "retriever.py",
                    "line": 1,
                    "current_code": "top_k = 10",
                    "replacement_code": "top_k = 20",
                    "proposed_change": "Increase the candidate limit.",
                },
                "verification": "Replay the same cases.",
            }}, evidence, {})
            self.assertFalse(result["fix"]["source_matches"])
            self.assertEqual(result["fix"]["replacement_code"], "")

    def test_python_replacement_must_parse_before_it_is_shown_as_copy_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "retriever.py").write_text("top_k = 5\n", encoding="utf-8")
            result = _validate_final(workspace, {"final": {
                "status": "probable",
                "root_cause": "RETRIEVAL",
                "summary": "The candidate configuration may be wrong.",
                "evidence": [],
                "fix": {
                    "file": "retriever.py",
                    "line": 1,
                    "current_code": "top_k = 5",
                    "replacement_code": "def broken(",
                    "proposed_change": "Update the retrieval setting.",
                },
                "verification": "Replay the same cases.",
            }}, {"runtime_capture_available": False, "deterministic_arithmetic_audit": []}, {})
            self.assertTrue(result["fix"]["source_matches"])
            self.assertEqual(result["fix"]["syntax_status"], "invalid")
            self.assertEqual(result["fix"]["replacement_code"], "")

    def test_workspace_search_skips_file_symlinks_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            outside = Path(directory) / "outside.py"
            outside.write_text("private_marker = 'outside'", encoding="utf-8")
            link = root / "linked.py"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("File symlinks are not available on this system.")
            result = _search_tool(root, {"query": "private_marker"})
            self.assertEqual(result["matches"], [])

    def test_local_model_request_does_not_follow_redirects(self):
        handler = _NoRedirectHandler()
        request = Request("http://127.0.0.1:11434/api/chat")
        with self.assertRaises(HTTPError) as caught:
            handler.redirect_request(request, None, 302, "Found", {}, "https://example.com/collect")
        self.assertEqual(caught.exception.code, 302)

    def test_local_model_opener_ignores_environment_proxies(self):
        with patch.dict("os.environ", {
            "HTTP_PROXY": "http://proxy.invalid:8080",
            "HTTPS_PROXY": "http://proxy.invalid:8080",
            "http_proxy": "http://proxy.invalid:8080",
            "https_proxy": "http://proxy.invalid:8080",
        }), patch("urllib.request.getproxies", side_effect=AssertionError("proxy environment was read")) as getproxies:
            opener = _local_model_opener()

        getproxies.assert_not_called()
        self.assertIsNotNone(opener)

    def test_missing_local_model_has_a_specific_setup_error(self):
        class MissingModelOpener:
            def open(self, _request, timeout):
                raise HTTPError("http://127.0.0.1:11434/api/chat", 404, "Not Found", {}, None)

        with patch(
            "backend.app.diagnose.agentic_investigator.urllib.request.build_opener",
            return_value=MissingModelOpener(),
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 404.*test-model.*api/chat"):
                _call_local_model("http://127.0.0.1:11434/api/chat", "test-model", [], timeout=1)

    def test_local_model_http_error_exposes_bounded_server_detail(self):
        class BrokenModelOpener:
            def open(self, _request, timeout):
                body = BytesIO(b'{"error":"input token count exceeds context length"}')
                raise HTTPError("http://127.0.0.1:11434/api/chat", 500, "Server Error", {}, body)

        with patch(
            "backend.app.diagnose.agentic_investigator.urllib.request.build_opener",
            return_value=BrokenModelOpener(),
        ):
            with self.assertRaisesRegex(RuntimeError, "exceeds context length"):
                _call_local_model("http://127.0.0.1:11434/api/chat", "test-model", [], timeout=1)

    def test_local_model_reasoning_flag_is_forwarded_but_thinking_text_is_not_returned(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps({
                    "message": {
                        "thinking": "private internal reasoning",
                        "content": '{"final":{"status":"insufficient_evidence"}}',
                    },
                }).encode("utf-8")

        class RecordingOpener:
            def __init__(self):
                self.payload = None

            def open(self, request, timeout):
                self.payload = json.loads(request.data.decode("utf-8"))
                return Response()

        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                opener = RecordingOpener()
                with patch(
                    "backend.app.diagnose.agentic_investigator._local_model_opener",
                    return_value=opener,
                ):
                    content = _call_local_model(
                        "http://127.0.0.1:11434/api/chat",
                        "test-model",
                        [],
                        timeout=1,
                        think=enabled,
                    )
                self.assertIs(opener.payload["think"], enabled)
                self.assertEqual(content, '{"final":{"status":"insufficient_evidence"}}')
                self.assertNotIn("private internal reasoning", content)

    def test_investigator_retries_once_without_thinking_when_final_content_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            response = json.dumps({"final": {
                "status": "insufficient_evidence",
                "root_cause": "UNKNOWN",
                "summary": "No captured run or concrete code defect was available.",
                "evidence": [],
                "fix": None,
            }})
            with patch(
                "backend.app.diagnose.agentic_investigator._call_local_model",
                side_effect=[LocalModelNoContentError("No message.content"), response],
            ) as model_call:
                result = investigate_case(
                    workspace,
                    {"diagnosis": {}},
                    "http://127.0.0.1:11434/api/chat",
                    "local-test-model",
                    thinking=True,
                )

        self.assertEqual([call.kwargs["think"] for call in model_call.call_args_list], [True, False])
        self.assertTrue(result["agent_trace"]["thinking_enabled"])
        self.assertTrue(result["agent_trace"]["thinking_fallback_used"])
        self.assertNotIn("No message.content", json.dumps(result))

    def test_model_claim_is_probable_unless_verifier_confirms_it(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "app.py"
            source.write_text("answer = generate(context)\n", encoding="utf-8")
            evidence = {
                "answer": "The model answered 42.",
                "prompt": "",
                "question": "What is the answer?",
                "expected_answer": "42",
                "retrieved_chunks": [],
                "deterministic_arithmetic_audit": [],
            }
            code_sources = {"app.py": "1: answer = generate(context)"}
            result = _validate_final(workspace, {
                "final": {
                    "status": "confirmed",
                    "root_cause": "GENERATOR",
                    "summary": "The model missed the answer.",
                    "evidence": [{"source": "answer", "quote": "answered 42"}],
                    "fix": {"file": "app.py", "line": 1, "proposed_change": "Add a verified calculation."},
                    "verification": "Rerun the case.",
                }
            }, evidence, code_sources)
            self.assertEqual(result["status"], "probable")
            self.assertTrue(result["evidence"][0]["verified"])
            self.assertEqual(result["fix"]["current_code"], "answer = generate(context)")

    def test_model_result_summary_verification_and_citations_are_scrubbed(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "app.py"
            source.write_text(
                "# Contact customer@example.com with sk-123456789012345678901234\n",
                encoding="utf-8",
            )
            code_line = source.read_text(encoding="utf-8").strip()
            result = _validate_final(workspace, {"final": {
                "status": "probable",
                "root_cause": "GENERATOR",
                "summary": "The source contains customer@example.com and sk-123456789012345678901234.",
                "evidence": [{"source": "code:app.py:1", "quote": code_line}],
                "fix": None,
                "verification": "Check customer@example.com and sk-123456789012345678901234.",
            }}, {
                "runtime_capture_available": False,
                "question": "",
                "expected_answer": "",
                "answer": "",
                "prompt": "",
                "retrieved_chunks": [],
                "deterministic_arithmetic_audit": [],
            }, {})

        serialized = json.dumps(result)
        self.assertNotIn("customer@example.com", serialized)
        self.assertNotIn("sk-123456789012345678901234", serialized)
        self.assertIn("[REDACTED_EMAIL]", serialized)
        self.assertIn("[REDACTED_API_KEY]", serialized)
        self.assertTrue(result["evidence"][0]["verified"], result["evidence"])

    def test_only_known_safe_top_k_experiments_are_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence = {
                "answer": "Not found.",
                "prompt": "",
                "question": "Find the filing value.",
                "expected_answer": "10",
                "retrieved_chunks": [],
                "baseline_diagnosis": {"runtime_parameters": {"top_k_final": 5}},
                "deterministic_arithmetic_audit": [],
            }
            proposed = {"final": {
                "status": "probable",
                "root_cause": "RETRIEVAL",
                "summary": "The candidate limit may be too small.",
                "evidence": [{"source": "answer", "quote": "Not found."}],
                "fix": None,
                "experiment": {"parameter": "top_k", "baseline": 99, "candidate": 10, "rationale": "Compare one cutoff."},
                "verification": "Run the same labeled cases.",
            }}
            result = _validate_final(workspace, proposed, evidence, {})
            self.assertEqual(result["experiment"], {
                "parameter": "top_k",
                "baseline": 5,
                "candidate": 10,
                "rationale": "Compare one cutoff.",
            })
            proposed["final"]["experiment"] = {"parameter": "temperature", "candidate": 0}
            self.assertIsNone(_validate_final(workspace, proposed, evidence, {})["experiment"])

    def test_static_model_cannot_invent_unmeasured_top_k_value(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "pipeline.ts"
            source.parent.mkdir()
            current = "  const passages = await vectorStore.similaritySearch(question);"
            source.write_text(current + "\n", encoding="utf-8")
            result = _validate_final(workspace, {"final": {
                "status": "probable",
                "root_cause": "RETRIEVAL",
                "summary": "The API may retrieve too many results by default.",
                "evidence": [{"source": "code:src/pipeline.ts:1", "quote": current.strip()}],
                "fix": {
                    "file": "src/pipeline.ts",
                    "line": 1,
                    "end_line": 1,
                    "current_code": current,
                    "replacement_code": "  const passages = await vectorStore.similaritySearch(question, { topK: 3 });",
                    "proposed_change": "Add a topK limit.",
                },
            }}, {"runtime_capture_available": False, "deterministic_arithmetic_audit": []}, {})

        self.assertEqual(result["root_cause"], "UNKNOWN")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["fix"])
        self.assertTrue(result["unverified_tuning_suppressed"])
        self.assertIn("no captured run establishes the baseline", result["summary"])

    def test_investigator_uses_read_only_tool_loop_then_returns_grounded_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "opaque.py"
            source.parent.mkdir()
            source.write_text("allowance = 1\nretrieved = index.fetch(question, allowance=allowance)\n", encoding="utf-8")
            responses = [
                json.dumps({"actions": [{"tool": "list_files", "path": "src"}]}),
                json.dumps({"actions": [{"tool": "read_file", "path": "src/opaque.py", "start_line": 1, "end_line": 2}]}),
                json.dumps({"final": {
                    "status": "confirmed",
                    "root_cause": "RETRIEVAL",
                    "summary": "The opaque retrieval allowance is configured to one candidate.",
                    "evidence": [{"source": "code:src/opaque.py:1", "quote": "allowance = 1"}],
                    "fix": {"file": "src/opaque.py", "line": 1, "proposed_change": "Compare allowance=1 with a larger candidate pool."},
                    "verification": "Replay the same labeled retrieval cases and compare relevant-document recall and latency.",
                }}),
            ]
            with patch("backend.app.diagnose.agentic_investigator._call_local_model", side_effect=responses) as model_call:
                result = investigate_case(workspace, {
                    "case_id": "case-1",
                    "question": "Find the fact",
                    "expected_answer": "known fact",
                    "diagnosis": {"llm_answer": "not found"},
                }, "http://127.0.0.1:11434/api/chat", "local-test-model")
            self.assertEqual(result["status"], "probable")
            self.assertEqual(result["root_cause"], "RETRIEVAL")
            self.assertEqual(result["fix"]["file"], "src/opaque.py")
            self.assertEqual(result["agent_turns"], 3)
            self.assertFalse(result["model_changed_files"])
            self.assertEqual(result["agent_trace"]["read_only_tool_calls"], 2)
            self.assertEqual(result["agent_trace"]["tools_used"], ["list_files", "read_file"])
            self.assertEqual(result["agent_trace"]["source_files_opened"], ["src/opaque.py"])
            self.assertTrue(all(call.kwargs["think"] is True for call in model_call.call_args_list))

    def test_agentic_investigation_round_trips_through_local_ollama_compatible_http(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "src" / "retriever.py"
            source.parent.mkdir()
            source.write_text("top_k = 0\nresults = search(query, top_k=top_k)\n", encoding="utf-8")
            responses = [
                json.dumps({"actions": [{"tool": "list_files", "path": "src"}]}),
                json.dumps({"actions": [{"tool": "read_file", "path": "src/retriever.py", "start_line": 1, "end_line": 2}]}),
                json.dumps({"final": {
                    "status": "probable",
                    "root_cause": "RETRIEVAL",
                    "summary": "The candidate count is zero.",
                    "evidence": [{"source": "code:src/retriever.py:1", "quote": "top_k = 0"}],
                    "fix": {
                        "file": "src/retriever.py",
                        "line": 1,
                        "end_line": 1,
                        "current_code": "top_k = 0",
                        "replacement_code": "top_k = 5",
                        "proposed_change": "Use a positive candidate count.",
                    },
                    "verification": "Replay the same labeled cases and compare support and latency.",
                }}),
            ]
            models = []

            class OllamaCompatibleHandler(BaseHTTPRequestHandler):
                def do_POST(self):
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    models.append(payload.get("model"))
                    content = responses.pop(0)
                    body = json.dumps({"message": {"content": content}}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *_args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaCompatibleHandler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                try:
                    result = investigate_case(
                        workspace,
                        {"diagnosis": {}},
                        f"http://127.0.0.1:{server.server_port}/api/chat",
                        "test-local-model",
                    )
                except RuntimeError as exc:
                    reason = getattr(getattr(exc, "__cause__", None), "reason", None)
                    if isinstance(reason, OSError) and (reason.errno in {1, 13} or getattr(reason, "winerror", None) == 10013):
                        self.skipTest("The sandbox blocks loopback client sockets.")
                    raise
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertEqual(models, ["test-local-model"] * 3)
            self.assertEqual(result["root_cause"], "RETRIEVAL")
            self.assertTrue(result["fix"]["source_matches"])
            self.assertEqual(result["fix"]["replacement_code"], "top_k = 1")

    def test_api_exposes_opt_in_agentic_investigation(self):
        from backend.app.api.server import AgentInvestigationRequest, agentic_investigate

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            expected = {
                "status": "probable",
                "root_cause": "RETRIEVAL",
                "summary": "Candidate pool may be too small.",
                "evidence": [],
                "deterministic_checks": [],
                "fix": None,
                "verification": "Compare top-k values.",
                "model_changed_files": False,
                "storage": "VOLATILE_RAM_ONLY",
                "model": "test-model",
                "agent_turns": 1,
            }
            request = AgentInvestigationRequest(
                workspace_path=str(workspace),
                case_id="sample",
                question="Find the fact",
                expected_answer="known fact",
                diagnosis={"llm_answer": "not found"},
                local_model_endpoint="http://127.0.0.1:11434/api/chat",
                local_model_name="test-model",
                local_model_thinking=False,
            )
            with patch("backend.app.api.server._resolve_workspace", return_value=workspace), patch(
                "backend.app.api.server.investigate_case", return_value=expected
            ) as investigate:
                response = agentic_investigate(request)
            self.assertEqual(response["investigation"]["root_cause"], "RETRIEVAL")
            self.assertEqual(response["network_policy"], "LOOPBACK_MODEL_ONLY")
            investigate.assert_called_once()
            self.assertIs(investigate.call_args.kwargs["thinking"], False)

    def test_evaluate_route_marks_explicit_wrong_answer_arithmetic(self):
        from backend.app.api.server import (
            EvaluateCapturedRunRequest,
            VOLATILE_EXTERNAL_RUNS,
            evaluate_captured_run,
        )

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run = {
                "llm_answer": r"Average $\frac{253 + 282}{2} = 266.5$. Ratio $\frac{6489}{266.5} \approx 24.39$.",
                "retrieved_chunks": [],
            }
            baseline = {
                "status": "needs_review",
                "root_cause": None,
                "evidence_level": "limited",
                "explanation": "Unresolved.",
                "recommended_fix": "Review the result.",
                "evidence": [],
            }
            VOLATILE_EXTERNAL_RUNS[str(workspace)] = run
            try:
                with patch("backend.app.api.server._resolve_workspace", return_value=workspace), patch(
                    "backend.app.api.server.diagnose_live_run", return_value=baseline.copy()
                ), patch("backend.app.api.server._code_references_for_diagnosis", return_value=[]):
                    result = evaluate_captured_run(EvaluateCapturedRunRequest(
                        workspace_path=str(workspace),
                        expected_answer="24.26",
                        case_id="financebench_id_02987",
                        question_type="numeric-derived",
                    ))
            finally:
                VOLATILE_EXTERNAL_RUNS.pop(str(workspace), None)
            diagnosis = result["diagnosis"]
            self.assertEqual(diagnosis["finding"], "ARITHMETIC_MISMATCH")
            self.assertEqual(diagnosis["root_cause"], "GENERATOR")
            self.assertEqual(len(diagnosis["arithmetic_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
