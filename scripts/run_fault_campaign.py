"""Generate and run a local RAG fault-detection campaign against RAG Doctor.

Each case gets a fresh temporary workspace and a distinct source mutation. The
local model call is stubbed for the 700-run deterministic coverage sweep; use
--qwen-sample N for an additional small live-model sample.
"""

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.diagnose.agentic_investigator import investigate_case


UNKNOWN = json.dumps({"final": {
    "status": "insufficient_evidence", "root_cause": "UNKNOWN",
    "summary": "No supported deterministic defect was found.", "evidence": [],
    "fix": None, "verification": "Add runtime evidence or inspect the relevant code path.",
}})


def scenario(family: str, n: int):
    size = 128 + n % 97
    if family == "GENERATOR_CONTEXT_DROPPED":
        q, docs, ctx = f"question_{n}", f"retrieved_documents_{n}", f"context_text_{n}"
        source = (f"def build_prompt({q}, {docs}):\n"
                 f"    {ctx} = '\\n'.join(str(item) for item in {docs})\n"
                 f"    return (\"Use evidence only.\\n\" + f\"Question: {{{q}}}\\n\" + \"\")\n")
        path, cause = "src/generation/generator.py", "GENERATOR"
        diagnosis = {"llm_answer": "Unsupported answer.", "prompt": f"Question: test-{n}\nRetrieved passages:\n", "source": "runtime"}
    elif family == "GENERATOR_PROMPT_MISBINDING":
        q, docs = f"user_query_{n}", f"retrieved_context_{n}"
        source = (f"def build_prompt({q}, {docs}):\n"
                 f"    return f\"Context: {{{docs}}}\\nQuestion: {{{docs}}}\"\n")
        path, cause, diagnosis = "src/generation/generator.py", "GENERATOR", {}
    elif family == "CHUNKING_INVALID_OVERLAP":
        source = (f"CHUNK_SIZE_{n} = {size}\nCHUNK_OVERLAP_{n} = {size + n % 13}\n"
                 "def split(text):\n    start = 0\n    out = []\n"
                 f"    step_{n} = CHUNK_SIZE_{n} - CHUNK_OVERLAP_{n}\n"
                 "    while start < len(text):\n"
                 f"        out.append(text[start:start + CHUNK_SIZE_{n}])\n"
                 f"        start += step_{n}\n    return out\n")
        path, cause, diagnosis = "src/ingestion/chunker.py", "CHUNKING", {}
    elif family == "CHUNKING_NONPROGRESSING_LOOP":
        source = (f"def split_{n}(text):\n    cursor_{n} = 0\n    out = []\n"
                 f"    while cursor_{n} < len(text):\n"
                 f"        out.append(text[cursor_{n}:cursor_{n}+{size}])\n"
                 f"        cursor_{n} += 0\n    return out\n")
        path, cause, diagnosis = "src/ingestion/chunker.py", "CHUNKING", {}
    elif family == "RANKING_ASCENDING_SIMILARITY":
        source = (f"def retrieve_{n}(items):\n"
                 f"    ranked_{n} = sorted(items, key=lambda item: item['similarity_{n}'])\n"
                 f"    return ranked_{n}[:{1 + n % 8}]\n")
        path, cause = "src/retrieval/vector_store.py", "RANKING_K"
        diagnosis = {
            "llm_answer": "Answer based on retrieved sources.",
            "retrieved_chunks": [
                {"chunk_id": f"c{n}a", "doc_id": "doc-a", "text": "evidence one", "rank": 1, "dense_score": 0.2},
                {"chunk_id": f"c{n}b", "doc_id": "doc-b", "text": "evidence two", "rank": 2, "dense_score": 0.9},
            ],
            "source": "runtime",
        }
        diagnosis["question"] = "When higher similarity scores are better, which candidate should rank first?"
    elif family == "RETRIEVAL_TOPK_ZERO":
        source = f"TOP_K_RETRIEVAL = 0\ndef retrieve_{n}(store, query):\n    return store.search(query, top_k=TOP_K_RETRIEVAL)\n"
        path, cause, diagnosis = "src/config.py", "RETRIEVAL", {}
    elif family == "RERANKER_SCORE_ASCENDING":
        source = (f"def rerank_{n}(scored):\n"
                 f"    scored.sort(key=lambda row: row['rerank_score_{n}'])\n"
                 f"    return scored[:{1 + n % 5}]\n")
        path, cause = "src/retrieval/reranker.py", "RERANKER"
        diagnosis = {"question": "Higher reranker scores are better; which candidate should rank first?",
                     "source": "run_with_doctor", "retrieval_observed": True, "retrieval_count": 2,
                     "retrieved_chunks": [
                         {"chunk_id": f"low{n}", "doc_id": "d1", "text": "low score", "final_rank": 1, "rerank_score": 0.1},
                         {"chunk_id": f"high{n}", "doc_id": "d2", "text": "high score", "final_rank": 2, "rerank_score": 0.9},
                     ]}
    elif family == "CONTEXT_TRUNCATION":
        source = (f"def build_context_{n}(chunks):\n"
                 f"    return '\\n'.join(chunk.text for chunk in chunks)[:0]\n")
        path, cause, diagnosis = "src/generation/generator.py", "GENERATOR", {}
    elif family == "EVALUATOR_FALSE_PASS":
        source = (f"def evaluate_match_{n}(gold_answer, generated_answer):\n"
                 f"    gold_clean = gold_answer.strip().lower()\n"
                 f"    answer_clean = generated_answer.strip().lower()\n"
                 f"    if gold_clean and gold_clean in answer_clean:\n        return True\n"
                 f"    gold_nums = extract_all_numbers(gold_answer)\n"
                 f"    answer_nums = extract_all_numbers(generated_answer)\n"
                 f"    if gold_nums and answer_nums:\n        target = gold_nums[0]\n"
                 f"        for answer_value in answer_nums:\n"
                 f"            if abs(target - answer_value) <= max(0.15, abs(target * 0.02)):\n                return True\n"
                 f"    return False\n")
        path, cause, diagnosis = "tests/run_benchmark.py", "EVALUATION", {}
    elif family == "HEALTHY_CONTROL":
        source = (f"def build_prompt_{n}(question, context):\n"
                 f"    return f\"Context: {{context}}\\nQuestion: {{question}}\"\n")
        path, cause, diagnosis = "src/generation/generator.py", "UNKNOWN", {}
    else:
        raise ValueError(family)
    return {"case_id": f"{family}-{n:03d}", "family": family, "expected": cause,
            "path": path, "source": source, "diagnosis": diagnosis}


FAMILIES = [
    "GENERATOR_CONTEXT_DROPPED", "GENERATOR_PROMPT_MISBINDING",
    "CHUNKING_INVALID_OVERLAP", "CHUNKING_NONPROGRESSING_LOOP",
    "RANKING_ASCENDING_SIMILARITY", "RETRIEVAL_TOPK_ZERO",
    "RERANKER_SCORE_ASCENDING", "CONTEXT_TRUNCATION", "EVALUATOR_FALSE_PASS",
    "HEALTHY_CONTROL",
]


def run_case(case, workspace_root, live_model=None):
    with tempfile.TemporaryDirectory(prefix="ragdoctor-fault-") as tmp:
        root = Path(tmp)
        source_path = root / case["path"]
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(case["source"], encoding="utf-8")
        payload = {"case_id": case["case_id"], "question": case.get("diagnosis", {}).get("question", "Diagnose this controlled fault."),
                   "expected_answer": "expected evidence", "diagnosis": case["diagnosis"]}
        started = time.perf_counter()
        if live_model:
            result = investigate_case(root, payload, "http://127.0.0.1:11434/api/chat", live_model, thinking=False)
        else:
            with patch("backend.app.diagnose.agentic_investigator._call_local_model", return_value=UNKNOWN):
                result = investigate_case(root, payload, "http://127.0.0.1:11434/api/chat", "campaign-stub", thinking=False)
        elapsed = time.perf_counter() - started
        detected = result.get("root_cause", "UNKNOWN")
        matched = detected == case["expected"]
        return {
            "case_id": case["case_id"], "family": case["family"], "expected": case["expected"],
            "detected": detected, "status": result.get("status"), "correct": matched,
            "fix_file": (result.get("fix") or {}).get("file"), "fix_line": (result.get("fix") or {}).get("line"),
            "source_matches": (result.get("fix") or {}).get("source_matches"),
            "syntax_status": (result.get("fix") or {}).get("syntax_status"),
            "elapsed_seconds": round(elapsed, 4), "model": live_model or "deterministic-verifier+stub",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=700)
    parser.add_argument("--qwen-sample", type=int, default=0)
    parser.add_argument("--output", default="data/fault_campaign_results.csv")
    args = parser.parse_args()
    if args.runs < 700:
        raise SystemExit("The campaign requires at least 700 runs.")
    fault_families = FAMILIES[:-1]
    cases = [scenario(fault_families[i % len(fault_families)], i) for i in range(args.runs)]
    control_count = max(100, args.runs // 7)
    cases.extend(scenario("HEALTHY_CONTROL", args.runs + i) for i in range(control_count))
    qwen_indexes = set()
    if args.qwen_sample:
        per_family = max(1, args.qwen_sample // len(FAMILIES))
        for family in FAMILIES:
            qwen_indexes.update(i for i, case in enumerate(cases) if case["family"] == family and i < 9 * per_family)
        qwen_indexes = set(sorted(qwen_indexes)[:args.qwen_sample])
    results = []
    for i, case in enumerate(cases, 1):
        result = run_case(case, None, "qwen3:4b" if i - 1 in qwen_indexes else None)
        results.append(result)
        print(f"[{i:03d}/{len(cases)}] {result['family']} expected={result['expected']} detected={result['detected']} {'PASS' if result['correct'] else 'MISS'} {result['elapsed_seconds']:.3f}s", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    by_family = {}
    for family in FAMILIES:
        group = [r for r in results if r["family"] == family]
        by_family[family] = {"runs": len(group), "correct": sum(r["correct"] for r in group),
                             "accuracy": round(sum(r["correct"] for r in group) / len(group), 4) if group else None}
    fault_results = [r for r in results if r["family"] != "HEALTHY_CONTROL"]
    control_results = [r for r in results if r["family"] == "HEALTHY_CONTROL"]
    summary = {"runs": len(results), "fault_runs": len(fault_results), "faults_detected_correctly": sum(r["correct"] for r in fault_results),
               "fault_detection_rate": round(sum(r["correct"] for r in fault_results) / len(fault_results), 4),
               "healthy_controls": len(control_results), "healthy_false_positive_rate": round(sum(r["detected"] != "UNKNOWN" for r in control_results) / len(control_results), 4),
               "qwen_runs": sum(r["model"] == "qwen3:4b" for r in results), "by_family": by_family,
               "results_file": str(output.resolve())}
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
