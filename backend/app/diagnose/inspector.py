import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from backend.app.core.security import SecurityGuard
from backend.app.schemas.contracts import FailureLabel


class WorkspaceASTInspector:
    """
    Reads a developer's local RAG code and dataset files WITHOUT needing existing RAG logs.
    Uses Python's AST (Abstract Syntax Tree) to locate the exact file and line number where
    RAG parameters (chunk_size, chunk_overlap, top_k, retriever) are defined, and cross-checks
    them against the actual dataset structure.
    """

    TARGET_KNOBS = {
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "top_k_final",
        "top_n_candidates",
        "temperature",
    }

    EXCLUDED_DIRS = {
        "venv",
        ".venv",
        "__pycache__",
        "node_modules",
        "tests",
        "vscode-extension",
        ".git",
        ".cache",
    }

    def scan_workspace(self, workspace_dir: str) -> Dict[str, Any]:
        root = Path(workspace_dir).resolve()
        code_findings: List[Dict[str, Any]] = []
        logic_warnings: List[Dict[str, Any]] = []
        detected_knobs: Dict[str, Any] = {}
        dataset_files: List[Dict[str, Any]] = []
        evaluation_cases: List[Dict[str, Any]] = []

        code_files = self._iter_files(root, {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}, 400)
        network_locations: List[Dict[str, Any]] = []
        pipeline_locations: List[Dict[str, Any]] = []
        for code_file in code_files:
            try:
                source = code_file.read_text(encoding="utf-8-sig", errors="ignore").lstrip("\ufeff")
                lines = source.splitlines()
                rel_path = code_file.relative_to(root)
                if code_file.suffix.lower() == ".py":
                    tree = ast.parse(source, filename=str(code_file))
                    self._extract_ast_knobs(tree, rel_path, lines, code_findings, detected_knobs)
                    logic_warnings.extend(self._find_chunk_step_warnings(tree, rel_path, lines))
                else:
                    self._extract_text_knobs(rel_path, lines, code_findings, detected_knobs)
                network_locations.extend(self._find_network_calls(rel_path, lines))
                pipeline_locations.extend(self._find_pipeline_calls(rel_path, lines))
            except (OSError, UnicodeError, SyntaxError, ValueError):
                continue

        for doc_file in self._iter_files(root, {".txt", ".md", ".csv", ".json", ".jsonl", ".ndjson"}, 200):
            if doc_file.name.lower() == "requirements.txt":
                continue
            try:
                raw_text = doc_file.read_text(encoding="utf-8-sig", errors="ignore")
                if len(raw_text.strip()) < 40:
                    continue
                if doc_file.suffix.lower() == ".csv":
                    rows = list(csv.reader(raw_text.splitlines()[:10001]))
                    try:
                        evaluation_rows = list(csv.DictReader(raw_text.splitlines()[:501]))
                        evaluation_cases.extend(self._extract_evaluation_cases(evaluation_rows, str(doc_file.relative_to(root))))
                    except csv.Error:
                        pass
                    dataset_files.append({
                        "file": str(doc_file.relative_to(root)),
                        "format": "csv",
                        "char_length": len(raw_text),
                        "num_rows_sampled": max(0, len(rows) - 1),
                        "column_count": len(rows[0]) if rows else 0,
                        "columns": [SecurityGuard.sanitize_text(str(name)[:80]) for name in (rows[0] if rows else [])[:40]],
                    })
                    continue
                if doc_file.suffix.lower() in {".jsonl", ".ndjson"}:
                    records = []
                    for line in raw_text.splitlines()[:500]:
                        try:
                            row = json.loads(line)
                            if isinstance(row, dict):
                                records.append(row)
                        except json.JSONDecodeError:
                            continue
                    evaluation_cases.extend(self._extract_evaluation_cases(records, str(doc_file.relative_to(root))))
                    dataset_files.append({
                        "file": str(doc_file.relative_to(root)),
                        "format": "jsonl",
                        "char_length": len(raw_text),
                        "record_count": len(records),
                    })
                    continue
                if doc_file.suffix.lower() == ".json":
                    parsed = json.loads(raw_text)
                    keys = list(parsed.keys())[:40] if isinstance(parsed, dict) else []
                    evaluation_cases.extend(self._extract_evaluation_cases(parsed, str(doc_file.relative_to(root))))
                    dataset_files.append({
                        "file": str(doc_file.relative_to(root)),
                        "format": "json",
                        "char_length": len(raw_text),
                        "record_count": len(parsed) if isinstance(parsed, list) else 1,
                        "top_level_keys": [SecurityGuard.sanitize_text(str(key)[:80]) for key in keys],
                    })
                    continue
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if len(p.strip()) > 20]
                max_para_len = max((len(p) for p in paragraphs), default=0)
                avg_para_len = int(sum(len(p) for p in paragraphs) / max(1, len(paragraphs)))
                dataset_files.append({
                    "file": str(doc_file.relative_to(root)),
                    "format": "text",
                    "char_length": len(raw_text),
                    "num_paragraphs": len(paragraphs),
                    "avg_paragraph_chars": avg_para_len,
                    "max_paragraph_chars": max_para_len,
                })
            except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
                continue

        return {
            "workspace_root": str(root),
            "detected_knobs": detected_knobs,
            "code_locations": code_findings,
            "dataset_summary": dataset_files,
            "evaluation_cases": evaluation_cases[:200],
            "evaluation_case_count": len(evaluation_cases),
            "static_warnings": logic_warnings + self._correlate_code_and_data(code_findings, dataset_files),
            "potential_data_transfer_locations": network_locations[:100],
            "pipeline_locations": pipeline_locations[:200],
            "scan_limits": {"code_files": len(code_files), "max_file_bytes": 1000000},
        }

    @staticmethod
    def _extract_evaluation_cases(value: Any, relative_file: str) -> List[Dict[str, Any]]:
        """Find locally labeled question/answer cases without assuming app naming conventions."""
        if isinstance(value, dict):
            for key in ("cases", "examples", "records", "data", "items"):
                nested = value.get(key)
                if isinstance(nested, list):
                    value = nested
                    break
        if not isinstance(value, list):
            return []

        cases: List[Dict[str, Any]] = []
        question_keys = ("question", "query", "user_query", "input_question")
        answer_keys = ("gold_answer", "expected_answer", "reference_answer", "ground_truth", "expected_output")
        document_keys = ("doc_name", "gold_doc_id", "relevant_document", "document_id", "gold_document_id")
        id_keys = ("financebench_id", "case_id", "question_id", "id")
        for index, row in enumerate(value[:500]):
            if not isinstance(row, dict):
                continue
            normalized = {str(key).lower(): item for key, item in row.items()}
            question = next((normalized.get(key) for key in question_keys if isinstance(normalized.get(key), str) and normalized[key].strip()), None)
            expected = next((normalized.get(key) for key in answer_keys if isinstance(normalized.get(key), str) and normalized[key].strip()), None)
            if not question or not expected:
                continue
            document = next((normalized.get(key) for key in document_keys if isinstance(normalized.get(key), str) and normalized[key].strip()), None)
            case_id = next((normalized.get(key) for key in id_keys if isinstance(normalized.get(key), (str, int))), index + 1)
            cases.append({
                "case_id": str(case_id)[:120],
                "file": relative_file,
                "question": question[:4000],
                "expected_answer": expected[:4000],
                "gold_document_id": str(document)[:300] if document else None,
                "question_type": str(normalized.get("question_type", ""))[:100] or None,
            })
        return cases

    def _iter_files(self, root: Path, suffixes: set, limit: int) -> List[Path]:
        files: List[Path] = []
        visited = 0
        for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in self.EXCLUDED_DIRS | {"dist", "build"}]
            for name in names:
                visited += 1
                if visited > 30000:
                    return files
                candidate = Path(current) / name
                if candidate.suffix.lower() not in suffixes:
                    continue
                try:
                    if candidate.stat().st_size > 1000000:
                        continue
                    candidate.resolve().relative_to(root)
                    files.append(candidate)
                except (OSError, ValueError):
                    continue
                if len(files) >= limit:
                    return files
        return files

    def _extract_text_knobs(self, rel_path: Path, lines: List[str], findings: List[Dict[str, Any]], detected: Dict[str, Any]) -> None:
        pattern = re.compile(r"\b(chunk_size|chunk_overlap|top_k_final|top_n_candidates|top_k|temperature)\b\s*[:=]\s*(\d+(?:\.\d+)?)")
        for line_no, line in enumerate(lines, start=1):
            for match in pattern.finditer(line):
                raw = match.group(2)
                value = float(raw) if "." in raw else int(raw)
                entry = {"parameter": match.group(1), "value": value, "file": str(rel_path), "line": line_no, "call": "text_setting", "code_snippet": SecurityGuard.sanitize_text(line.strip()[:500])}
                findings.append(entry)
                detected[entry["parameter"]] = entry

    @staticmethod
    def _find_network_calls(rel_path: Path, lines: List[str]) -> List[Dict[str, Any]]:
        pattern = re.compile(r"(?i)(requests\.(?:post|put|patch)|httpx\.(?:post|put|patch)|axios\.(?:post|put|patch)|fetch\s*\(|openai|anthropic|generativelanguage|api\.google|bedrock|vertexai)")
        found = []
        for line_no, line in enumerate(lines, start=1):
            if pattern.search(line):
                found.append({"file": str(rel_path), "line": line_no, "code_snippet": SecurityGuard.sanitize_text(line.strip()[:500]), "severity": "INFO", "note": "Static match only: this code may transfer data when executed; RAG Doctor has not observed a runtime transfer."})
        return found

    @staticmethod
    def _find_pipeline_calls(rel_path: Path, lines: List[str]) -> List[Dict[str, Any]]:
        patterns = {
            "chunking": re.compile(r"(?i)(chunk_size|chunk_overlap|textsplitter|split_documents|chunk_document|chunk_text)"),
            "retrieval": re.compile(r"(?i)(similarity_search\s*\(|as_retriever\s*\(|(?:vector_?store|retriever)\.(?:query|search|invoke|get_relevant_documents)\s*\(|retrieve_documents\s*\()") ,
            "reranking": re.compile(r"(?i)(\.rerank\s*\(|rerank_documents\s*\(|rank_documents\s*\()"),
            "prompt": re.compile(r"(?i)(prompttemplate|chatprompttemplate|system_prompt|system_message|assemble_prompt|self\._build_prompt\s*\()"),
            "generation": re.compile(r"(?i)(chat\.completions\.create\s*\(|generate_content\s*\(|generateContent\s*\(|\.generate(?:_[\w]+)?\s*\(|\.invoke\s*\(|llm\s*\(|model\.generate\w*\s*\()") ,
        }
        found = []
        for line_no, line in enumerate(lines, start=1):
            for component, pattern in patterns.items():
                matched = bool(pattern.search(line))
                if component == "generation" and not matched and re.search(r"(?i)requests\.(?:post|put)\s*\(", line):
                    nearby = "\n".join(lines[max(0, line_no - 12):line_no + 1])
                    matched = bool(re.search(r"(?i)(generativelanguage\.googleapis\.com|api\.openai\.com|api\.anthropic\.com|11434|/api/generate|/api/chat)", nearby))
                if matched:
                    found.append({
                        "component": component,
                        "file": str(rel_path),
                        "line": line_no,
                        "code_snippet": SecurityGuard.sanitize_text(line.strip()[:500]),
                    })
                    break
        return found

    def _extract_ast_knobs(
        self,
        tree: ast.AST,
        rel_path: Path,
        lines: List[str],
        findings: List[Dict[str, Any]],
        detected_knobs: Dict[str, Any],
    ) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = getattr(node.func, "id", getattr(node.func, "attr", "call"))
                for kw in node.keywords:
                    if (
                        kw.arg in self.TARGET_KNOBS
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, (int, float))
                        and not isinstance(kw.value.value, bool)
                    ):
                        line_no = getattr(kw, "lineno", getattr(node, "lineno", 1))
                        snippet = (
                            SecurityGuard.sanitize_text(lines[line_no - 1].strip()[:500])
                            if 0 < line_no <= len(lines)
                            else ""
                        )
                        entry = {
                            "parameter": kw.arg,
                            "value": kw.value.value,
                            "file": str(rel_path),
                            "line": line_no,
                            "call": str(func_name),
                            "code_snippet": snippet,
                        }
                        findings.append(entry)
                        detected_knobs[kw.arg] = entry

            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value_node = node.value
                if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, (int, float)) or isinstance(value_node.value, bool):
                    continue
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    normalized_name = target.id.lower()
                    param_name = "top_k" if normalized_name.startswith("top_k_") else normalized_name
                    if param_name not in self.TARGET_KNOBS:
                        continue
                    line_no = node.lineno
                    snippet = (
                        SecurityGuard.sanitize_text(lines[line_no - 1].strip()[:500])
                        if 0 < line_no <= len(lines)
                        else ""
                    )
                    entry = {
                        "parameter": param_name,
                        "value": value_node.value,
                        "file": str(rel_path),
                        "line": line_no,
                        "call": "annotated_assignment" if isinstance(node, ast.AnnAssign) else "assignment",
                        "code_snippet": snippet,
                    }
                    findings.append(entry)
                    detected_knobs[param_name] = entry

    @staticmethod
    def _find_chunk_step_warnings(
        tree: ast.AST,
        rel_path: Path,
        lines: List[str],
    ) -> List[Dict[str, Any]]:
        """Find common chunkers that configure overlap but advance by full size."""
        warnings: List[Dict[str, Any]] = []
        for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            has_overlap_setting = any(
                isinstance(node, ast.Attribute) and node.attr.lower() in {"chunk_overlap", "overlap"}
                or isinstance(node, ast.Name) and node.id.lower() in {"chunk_overlap", "overlap"}
                for node in ast.walk(class_node)
            )
            if not has_overlap_setting:
                continue
            for function in (node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                for statement in ast.walk(function):
                    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                        continue
                    value = statement.value
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    target_names = [target.id for target in targets if isinstance(target, ast.Name)]
                    if not target_names or not (
                        isinstance(value, ast.Name) and value.id.lower() == "chunk_size"
                        or isinstance(value, ast.Attribute) and value.attr.lower() == "chunk_size"
                    ):
                        continue
                    step_name = target_names[0]
                    advances_cursor = any(
                        isinstance(node, ast.AugAssign)
                        and isinstance(node.op, ast.Add)
                        and isinstance(node.target, ast.Name)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == step_name
                        for node in ast.walk(function)
                    )
                    if not advances_cursor:
                        continue
                    line_no = statement.lineno
                    snippet = lines[line_no - 1].strip()[:500] if 0 < line_no <= len(lines) else ""
                    warnings.append({
                        "severity": "MEDIUM",
                        "root_cause": FailureLabel.CHUNKING.value,
                        "file": str(rel_path),
                        "line": line_no,
                        "code_snippet": SecurityGuard.sanitize_text(snippet),
                        "problem": f"`{step_name}` advances by the full chunk size at {rel_path}:{line_no}, so the configured overlap is not applied between chunks.",
                        "fix_suggestion": "Advance by `max(1, chunk_size - chunk_overlap)` (or the equivalent instance fields), then confirm the labeled answer span is preserved in an indexed chunk.",
                    })
        return warnings

    def _correlate_code_and_data(
        self, code_findings: List[Dict[str, Any]], dataset_files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        warnings: List[Dict[str, Any]] = []
        paragraph_avgs = [d["avg_paragraph_chars"] for d in dataset_files if isinstance(d.get("avg_paragraph_chars"), (int, float))]
        avg_para = int(sum(paragraph_avgs) / max(1, len(paragraph_avgs)))
        paragraph_reference = max(180, avg_para)

        for item in code_findings:
            param = item["parameter"]
            val = item["value"]
            if param == "chunk_size" and isinstance(val, (int, float)):
                if val < paragraph_reference:
                    warnings.append(
                        {
                            "severity": "CRITICAL",
                            "root_cause": FailureLabel.CHUNKING.value,
                            "file": item["file"],
                            "line": item["line"],
                            "code_snippet": item["code_snippet"],
                            "problem": (
                                f"`chunk_size={val}` at {item['file']}:{item['line']} is smaller than the dataset's "
                                f"average paragraph length ({paragraph_reference} chars), which may split related facts across chunks."
                            ),
                            "fix_suggestion": f"Consider `chunk_size={max(240, paragraph_reference + 40)}` with `chunk_overlap=40`; then compare retrieval coverage on a labeled query set.",
                        }
                    )
            if param == "chunk_overlap" and val == 0:
                warnings.append(
                    {
                        "severity": "HIGH",
                        "root_cause": FailureLabel.CHUNKING.value,
                        "file": item["file"],
                        "line": item["line"],
                        "code_snippet": item["code_snippet"],
                        "problem": f"`chunk_overlap=0` at {item['file']}:{item['line']} provides zero boundary protection when sentences cross chunk splits.",
                        "fix_suggestion": "Set `chunk_overlap=40` (approx 15-20% of chunk_size).",
                    }
                )
            if param in ("top_k", "top_k_final") and isinstance(val, int) and val == 1:
                warnings.append(
                    {
                        "severity": "MEDIUM",
                        "root_cause": FailureLabel.RANKING_K.value,
                        "file": item["file"],
                        "line": item["line"],
                        "code_snippet": item["code_snippet"],
                        "problem": f"`{param}=1` at {item['file']}:{item['line']} drops any relevant chunk ranked #2 or #3 before prompt assembly.",
                        "fix_suggestion": f"Increase `{param}` from `1` to `3`.",
                    }
                )
            if param in {"top_k", "top_k_final", "top_n_candidates"} and isinstance(val, (int, float)) and val <= 0:
                warnings.append(
                    {
                        "severity": "CRITICAL",
                        "root_cause": FailureLabel.RETRIEVAL.value,
                        "file": item["file"],
                        "line": item["line"],
                        "code_snippet": item["code_snippet"],
                        "problem": f"`{param}={val}` at {item['file']}:{item['line']} is non-positive and can prevent retrieval from returning candidates.",
                        "fix_suggestion": f"Set `{param}` to a positive value. Use `1` only as the minimum valid value, then tune on labeled queries.",
                    }
                )
        return warnings
