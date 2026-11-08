"""Checks for separating prompt-extracted context from explicit retrieval traces."""

import tempfile
import unittest
from pathlib import Path

from backend.app.diagnose.agentic_investigator import _initial_evidence, _validate_final
from backend.app.diagnose.live_diagnoser import diagnose_live_run
from backend.app.api.server import LiveCaptureRequest, _sanitize_run


class PromptContextDiagnosisTests(unittest.TestCase):
    def test_answer_miss_in_heuristically_extracted_context_points_to_generation(self):
        result = diagnose_live_run({
            "question": "What was reported revenue?",
            "llm_answer": "Revenue was 79 million dollars.",
            "expected_answer": "82 million dollars",
            "retrieval_observed": None,
            "retrieved_chunks": [],
            "prompt_context_blocks": [{
                "label": "Context section",
                "prompt_position": 1,
                "extraction_method": "prompt_context_heading",
                "confidence": "moderate",
                "text": "The annual report states revenue was 82 million dollars.",
            }],
            "metadata": {"context_extraction_method": "prompt_context_heading"},
        })

        self.assertEqual(result["root_cause"], "GENERATOR")
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertTrue(result["prompt_context_content_included"])
        self.assertIn("heuristic", " ".join(result["evidence"]).lower())
        self.assertIn("no verified retriever identity or rank", " ".join(result["evidence"]).lower())

    def test_context_only_without_expected_fact_does_not_claim_retrieval_failure(self):
        result = diagnose_live_run({
            "question": "What was reported revenue?",
            "llm_answer": "I could not find the value.",
            "expected_answer": "82 million dollars",
            "retrieval_observed": None,
            "retrieved_chunks": [],
            "prompt_context_blocks": [{"text": "The filing discusses staffing and office locations."}],
        })

        self.assertIsNone(result["root_cause"])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["finding"], "GOLD_SOURCE_NOT_LABELED")

    def test_zero_retrieval_signal_conflicts_with_context_in_prompt(self):
        result = diagnose_live_run({
            "question": "What was reported revenue?",
            "llm_answer": "82 million dollars",
            "expected_answer": "82 million dollars",
            "retrieval_observed": True,
            "retrieval_count": 0,
            "retrieved_chunks": [],
            "prompt_context_blocks": [{"text": "The annual report states revenue was 82 million dollars."}],
        })

        self.assertEqual(result["root_cause"], "NO_FAILURE")
        self.assertIn("these signals conflict", " ".join(result["evidence"]))

    def test_agent_evidence_can_cite_prompt_context_without_claiming_it_is_a_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            diagnosis = {
                "source": "run_with_doctor",
                "llm_answer": "The value was 79 million.",
                "retrieval_observed": None,
                "retrieved_chunks": [],
                "prompt_context_blocks": [{
                    "label": "Context section",
                    "prompt_position": 1,
                    "extraction_method": "prompt_context_heading",
                    "confidence": "moderate",
                    "text": "The report states the value was 82 million dollars.",
                }],
            }
            evidence, _ = _initial_evidence(Path(directory), {"question": "What was the value?", "diagnosis": diagnosis})
            result = _validate_final(Path(directory), {"final": {
                "status": "probable",
                "root_cause": "GENERATOR",
                "summary": "The captured prompt contains the reference value, but the response differs.",
                "evidence": [{
                    "source": "prompt_context_block:0",
                    "quote": "The report states the value was 82 million dollars.",
                }],
                "fix": None,
                "verification": "Replay the same labeled case and inspect the captured answer.",
            }}, evidence, {})

        self.assertTrue(evidence["runtime_capture_available"])
        self.assertEqual(evidence["retrieved_chunks"], [])
        self.assertTrue(result["evidence"][0]["verified"])

    def test_api_sanitizer_preserves_context_only_when_content_capture_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            block = {
                "label": "Context section",
                "prompt_position": 1,
                "extraction_method": "prompt_context_heading",
                "confidence": "moderate",
                "text": "The report states revenue was 82 million dollars.",
            }
            opted_in = _sanitize_run(LiveCaptureRequest(
                workspace_path=str(workspace),
                prompt_context_blocks=[block],
                metadata={"context_extraction_method": "prompt_context_heading", "prompt_context_block_count": 1},
                include_content=True,
            ), workspace)
            redacted = _sanitize_run(LiveCaptureRequest(
                workspace_path=str(workspace),
                prompt_context_blocks=[block],
                metadata={"context_extraction_method": "prompt_context_heading", "prompt_context_block_count": 1},
                include_content=False,
            ), workspace)

        self.assertIn("82 million dollars", opted_in["prompt_context_blocks"][0]["text"])
        self.assertEqual(opted_in["metadata"]["context_extraction_method"], "prompt_context_heading")
        self.assertEqual(redacted["prompt_context_blocks"][0]["text"], "")
        self.assertEqual(redacted["prompt_context_count"], 1)

    def test_empty_observer_chunk_list_does_not_claim_zero_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            observer_capture = _sanitize_run(LiveCaptureRequest(
                workspace_path=str(workspace),
                retrieved_chunks=[],
                include_content=True,
            ), workspace)
            explicit_empty_retrieval = _sanitize_run(LiveCaptureRequest(
                workspace_path=str(workspace),
                retrieved_chunks=[],
                retrieval_observed=True,
                retrieval_count=0,
                include_content=True,
            ), workspace)

        self.assertIsNone(observer_capture["retrieval_observed"])
        self.assertIsNone(observer_capture["retrieval_count"])
        self.assertTrue(explicit_empty_retrieval["retrieval_observed"])
        self.assertEqual(explicit_empty_retrieval["retrieval_count"], 0)

    def test_empty_provider_prompt_is_reported_as_missing_retrieval_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            captured = _sanitize_run(LiveCaptureRequest(
                workspace_path=str(workspace),
                question="What was in the filing?",
                prompt="System instructions.\\n\\nRetrieved passages:\\n\\nQuestion: What was in the filing?",
                llm_answer="The filing says 42.",
                retrieved_chunks=[],
                include_content=True,
            ), workspace)
        result = diagnose_live_run(captured)

        self.assertIsNone(result["root_cause"])
        self.assertEqual(result["status"], "needs_context_content")
        self.assertNotIn("candidate count was zero", result["explanation"].lower())


if __name__ == "__main__":
    unittest.main()
