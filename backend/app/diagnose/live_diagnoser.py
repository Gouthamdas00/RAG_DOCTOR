"""Evidence-first diagnosis for locally captured RAG runs.

This module deliberately avoids claiming a root cause when a live integration has
not supplied enough trace evidence to distinguish pipeline stages.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from backend.app.schemas.contracts import FailureLabel


def _tokens(value: Any) -> set:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _ordered_tokens(value: Any) -> List[str]:
    text = str(value or "").lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)
    normalized = []
    for token in tokens:
        if token[0].isdigit():
            try:
                token = format(Decimal(token).normalize(), "f")
            except InvalidOperation:
                pass
        normalized.append(token)
    return normalized


def _answer_present(expected: str, answer: str) -> bool:
    """Conservative phrase match; partial token overlap is not factual evidence."""
    expected_tokens = _ordered_tokens(expected)
    answer_tokens = _ordered_tokens(answer)
    if not expected_tokens or len(expected_tokens) > len(answer_tokens):
        return False
    width = len(expected_tokens)
    return any(answer_tokens[index:index + width] == expected_tokens for index in range(len(answer_tokens) - width + 1))


def _answer_exact(expected: str, answer: str) -> bool:
    expected_tokens = _ordered_tokens(expected)
    return bool(expected_tokens) and expected_tokens == _ordered_tokens(answer)


def _is_derived_question(question: str, question_type: Optional[str]) -> bool:
    """Detect common prompts whose gold answer is computed rather than quoted."""
    if str(question_type or "").strip().lower() in {"calculation", "derived", "numeric-derived"}:
        return True
    return bool(re.search(
        r"(?i)\b(?:calculate|compute|ratio|average|mean|as a percentage|as a %|round .* decimal|defined as|what drove|growth rate)\b",
        question,
    ))


_QUESTION_STOPWORDS = {
    "the", "what", "was", "were", "is", "are", "does", "did", "has", "have", "for", "from", "with", "based",
    "on", "in", "of", "to", "by", "and", "or", "as", "at", "an", "a", "this", "that", "their", "its",
    "which", "how", "much", "many", "year", "end", "fy", "fiscal", "give", "answer", "question", "using",
    "shown", "details", "primarily", "assume", "public", "equities", "analyst", "company", "3m", "usd", "million",
    "millions", "billion", "billions", "please", "state", "explain", "relying", "provided", "available",
}


def _question_topic_terms(value: str) -> set:
    text = str(value or "").lower()
    text = re.sub(r"property[\s,]+plant[\s,]+(?:and\s+)?equipment", " ppe ", text)
    text = text.replace("pp&e", " ppe ").replace("ppne", " ppe ")
    text = re.sub(r"capital expenditures?", " capex ", text)
    terms = set(re.findall(r"[a-z]{2,}", text))
    return terms - _QUESTION_STOPWORDS


def _topic_coverage(question: str, context: str) -> Dict[str, Any]:
    query_terms = _question_topic_terms(question)
    context_terms = _question_topic_terms(context)
    matched = sorted(query_terms & context_terms)
    missing = sorted(query_terms - context_terms)
    return {
        "query_terms": sorted(query_terms),
        "matched_terms": matched,
        "missing_terms": missing,
        "coverage": round(len(matched) / max(1, len(query_terms)), 2) if query_terms else None,
    }


def _as_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else default
    except (TypeError, ValueError, OverflowError):
        return default


def _as_coverage(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if 0 <= parsed <= 1 else None
    except (TypeError, ValueError, OverflowError):
        return None


def diagnose_live_run(run: Dict[str, Any], static_warnings: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Return a root cause only when the available evidence supports one."""
    chunks = run.get("retrieved_chunks") or []
    chunk_texts = [str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in chunks]
    prompt_context_blocks = [block for block in (run.get("prompt_context_blocks") or [])[:10] if isinstance(block, dict)]
    prompt_context_texts = [str(block.get("text", "")) for block in prompt_context_blocks]
    joined_context = "\n".join(chunk_texts + prompt_context_texts)
    question = str(run.get("question") or "")
    answer = str(run.get("llm_answer") or "")
    expected = str(run.get("expected_answer") or "")
    oracle_passed = run.get("oracle_passed")
    coverage = _as_coverage(run.get("max_index_gold_coverage"))
    top_k = _as_int(run.get("top_k_final"))
    candidates = [c for c in chunks if isinstance(c, dict)]
    retrieval_observed = run.get("retrieval_observed")
    retrieval_count = run.get("retrieval_count")
    if not isinstance(retrieval_count, int) or isinstance(retrieval_count, bool) or retrieval_count < 0:
        retrieval_count = len(chunks)

    result: Dict[str, Any] = {
        "status": "diagnosed",
        "root_cause": None,
        "evidence_level": "limited",
        "explanation": "A run was captured, but the available evidence is not enough to identify one pipeline stage reliably.",
        "recommended_fix": "Capture retrieved chunk ranks/scores and, when possible, compare the run with a known expected answer or isolated-context oracle result.",
        "evidence": [],
        "source_location": None,
        "question": question,
        "llm_answer": answer,
        "expected_answer": expected or None,
        "prompt": str(run.get("prompt") or ""),
        "retrieved_chunks": chunks[:10],
        "prompt_context_blocks": prompt_context_blocks,
        "prompt_context_count": run.get("prompt_context_count", len(prompt_context_blocks)),
        "prompt_context_content_included": any(text.strip() for text in prompt_context_texts),
        "retrieval_observed": retrieval_observed,
        "retrieval_count": retrieval_count,
        "metadata": run.get("metadata") or {},
        "content_excluded": bool(run.get("content_excluded", False)),
        "retrieved_text_included": any(text.strip() for text in chunk_texts),
        "prompt_response_excluded": bool(run.get("prompt_response_excluded", not (question or answer or run.get("prompt")))),
        "finding": None,
        "finding_label": None,
    }

    if retrieval_observed is False and not any(text.strip() for text in prompt_context_texts):
        result.update(
            status="needs_retrieval_trace",
            explanation="The captured run has no retrieval field. RAG Doctor cannot tell whether retrieval returned nothing or telemetry omitted the result.",
            recommended_fix="Enable retrieval-span telemetry with candidate count, ranks, and scores. RAG Doctor will not infer an empty retrieval from a missing field.",
        )
        result["evidence"].append("No retrieval field was present in the source log record.")
        return result

    if retrieval_observed is False and any(text.strip() for text in prompt_context_texts):
        result["evidence"].append("No explicit retrieval field was present, but a context span was extracted from the assembled prompt; this does not prove the span came from a retriever.")

    if retrieval_observed is True and retrieval_count == 0 and not any(text.strip() for text in prompt_context_texts):
        result.update(
            status="diagnosed",
            root_cause=FailureLabel.RETRIEVAL.value,
            evidence_level="strong",
            explanation="Runtime telemetry explicitly reports zero retrieved candidates for this run.",
            recommended_fix="Inspect the query embedding, collection/index selection, metadata filters, and similarity threshold. Compare the query against a known relevant document.",
        )
        result["evidence"].append("The retrieval event was observed and its candidate count was zero.")
        return _attach_static_location(result, static_warnings)

    if retrieval_observed is True and retrieval_count == 0 and any(text.strip() for text in prompt_context_texts):
        result["evidence"].append("Retrieval telemetry reports zero candidates, but a context span was extracted from the assembled prompt; these signals conflict and the prompt span does not prove a retriever result.")

    if not any(text.strip() for text in chunk_texts + prompt_context_texts):
        result.update(
            status="needs_context_content",
            root_cause=None,
            evidence_level="limited",
            explanation=(
                f"Runtime metadata reports {retrieval_count} retrieved item(s), but no passage text or relevance labels are available. "
                "RAG Doctor cannot determine whether those items contain the required evidence."
                if retrieval_observed is True or chunks
                else "No retrieval result was captured, so RAG Doctor cannot determine whether retrieval returned nothing or was omitted from telemetry."
            ),
            recommended_fix="Capture the gold document ID/rank or locally opt in to passage-text inspection for this run. Otherwise, treat this as an observability gap, not a diagnosed retrieval failure.",
        )
        result["evidence"].append("No non-empty retrieved passage text was available for evidence matching.")
        return result

    if any(text.strip() for text in prompt_context_texts):
        method = str((run.get("metadata") or {}).get("context_extraction_method") or "prompt markers")
        result["evidence"].append(f"Assembled-prompt context was extracted using {method}; these spans are heuristic and have no verified retriever identity or rank.")

    if expected:
        if expected.strip().lower() in {"unanswerable", "not answerable", "no answer"}:
            abstained = bool(re.search(r"(?i)(cannot answer|can't answer|insufficient context|not in the provided|do not know|don't know)", answer))
            if abstained:
                result.update(status="needs_review", root_cause=None, evidence_level="limited", explanation="The response contains an abstention phrase, but RAG Doctor cannot confirm it did not also provide an unsupported answer.", recommended_fix="Review the full response or provide a structured abstention label from the evaluation harness.")
            else:
                result.update(status="needs_review", root_cause=None, evidence_level="limited", explanation="The query is labeled unanswerable, but no reliable abstention signal was supplied. A phrase scan alone cannot prove the model answered or abstained.", recommended_fix="Provide a structured abstained=true/false label from the application or evaluation harness.")
            result["evidence"].append("Expected-answer label marks this query as unanswerable.")
            return result
        answer_ok = _answer_present(expected, answer)
        expected_in_context = any(_answer_present(expected, chunk) for chunk in chunk_texts + prompt_context_texts)
        if answer_ok and _answer_exact(expected, answer):
            result.update(
                status="no_obvious_issue",
                root_cause=FailureLabel.NO_FAILURE.value,
                finding="REFERENCE_MATCH",
                finding_label="Reference answer matched",
                evidence_level="moderate",
                explanation="The normalized response exactly matches the supplied reference. This does not assess completeness or source faithfulness.",
                recommended_fix="No pipeline fix is indicated by this reference check; review source faithfulness separately if required.",
            )
            result["evidence"].append("The normalized response token sequence exactly matched the supplied expected answer.")
            return result
        if answer_ok:
            result.update(
                status="needs_review",
                root_cause=None,
                finding="PARTIAL_REFERENCE_MATCH",
                finding_label="Partial reference match; review units and extra claims",
                evidence_level="limited",
                explanation="The expected-answer phrase appears in the response, but that alone cannot verify polarity, units, completeness, or whether extra claims are wrong.",
                recommended_fix="Use a domain-specific validator or human-reviewed evaluation label before treating this run as successful.",
            )
            result["evidence"].append("Reference phrase match only; no semantic correctness claim was made.")
            return result

        if coverage is not None and float(coverage) < 0.75:
            result.update(
                root_cause=FailureLabel.CHUNKING.value,
                evidence_level="strong",
                explanation=f"The best indexed chunk covers only {float(coverage):.0%} of the expected answer span.",
                recommended_fix="Increase chunk_size or chunk_overlap, or use a boundary-aware chunker. Confirm the gold span stays intact in at least one indexed chunk.",
            )
            result["evidence"].append(f"Index-level expected-answer span coverage: {float(coverage):.0%}.")
            return _attach_static_location(result, static_warnings)

        if not expected_in_context:
            if run.get("expected_answer_in_source") is False:
                result.update(
                    status="needs_reference",
                    root_cause=None,
                    evidence_level="limited",
                    explanation="The expected answer is missing from both the retrieved chunks and the supplied source document. The reference may be incorrect, or the relevant data may not be present in the indexed source.",
                    recommended_fix="Verify the source document and expected answer first; then inspect ingestion and index freshness before changing retrieval settings.",
                )
                result["evidence"].append("Expected answer was not found in the supplied source document.")
                return result
            if run.get("gold_document_retrieved") is False:
                result.update(
                    root_cause=FailureLabel.RETRIEVAL.value,
                    finding="GOLD_DOCUMENT_NOT_RETRIEVED",
                    finding_label="Labeled source document missing from retrieved results",
                    evidence_level="strong" if run.get("expected_answer_in_source") is True else "moderate",
                    explanation="The labeled source document was not present in the captured retrieved chunks, and the reference answer was not found in their text.",
                    recommended_fix="Inspect the retrieval call and its filters. Compare the same query with a larger candidate pool, then verify whether the labeled source enters the final context before changing defaults.",
                )
                result["evidence"].append("The labeled gold document was explicitly absent from the captured retrieved chunks.")
                if run.get("expected_answer_in_source") is True:
                    result["evidence"].append("Expected answer was found in the supplied source document.")
                return _attach_static_location(result, static_warnings)

            derived = _is_derived_question(question, run.get("question_type"))
            result["derived_question"] = derived
            if derived or run.get("gold_document_retrieved") is True or run.get("gold_document_retrieved") is None:
                topic = _topic_coverage(question, joined_context)
                result["question_topic_evidence"] = topic
                rank = run.get("gold_document_rank")
                topic_coverage = topic.get("coverage")
                if derived:
                    finding = "DERIVED_RESULT_UNVERIFIED"
                    cue = next((word for word in ("ratio", "average", "mean", "percentage", "calculate", "compute", "growth rate") if word in question.lower()), "calculation")
                    finding_label = f"Derived {cue} answer needs input/formula validation"
                    explanation = "This question asks for a calculation or interpretation. The final reference value is not quoted verbatim, so the trace alone cannot tell whether an input fact was missing or the model calculated/interpreted it incorrectly."
                    recommendation = "Use the captured context to identify each required input and verify the formula, period, and units against the reference. If an input is absent, inspect retrieval; if all inputs are present in the assembled prompt, investigate generation or calculation."
                elif run.get("gold_document_retrieved") is True and topic_coverage is not None and topic_coverage < 0.5:
                    finding = "TARGET_TERMS_MISSING_FROM_CONTEXT"
                    missing_preview = ", ".join(topic["missing_terms"][:3]) or "target topic"
                    finding_label = f"Gold filing found; retrieved chunks miss: {missing_preview}"
                    explanation = "The labeled filing was retrieved, but fewer than half of the distinctive question terms appear in the captured chunks. This points to a possible within-document section/chunk selection gap; it does not prove the requested fact is absent."
                    recommendation = "Inspect the retrieved chunk IDs/ranks and compare their sections with the question's target topic. Test a larger candidate pool or section-aware retrieval for this case; keep the change only if the needed balance-sheet/table passage appears."
                elif run.get("gold_document_retrieved") is True:
                    finding = "TOPIC_PRESENT_REFERENCE_NOT_VERIFIED"
                    matched_preview = ", ".join(topic["matched_terms"][:3]) or "some topic terms"
                    finding_label = f"Topic terms found ({matched_preview}); reference mismatch unresolved"
                    explanation = "The labeled filing was retrieved and some distinctive question terms occur in its captured chunks, but the expected answer was not matched. The captured evidence cannot yet determine whether the exact table row was retrieved, the reference differs, or generation missed it."
                    recommendation = "Compare the exact period, statement row, value, and units in the retrieved passages with the reference. If the row is present and the answer is wrong, inspect generation; if the row is absent, investigate within-document retrieval."
                else:
                    finding = "GOLD_SOURCE_NOT_LABELED"
                    finding_label = "No labeled source-document signal"
                    explanation = "The reference answer is not quoted verbatim and this test case supplies no usable gold-document retrieval signal. RAG Doctor cannot separate a retrieval gap from a paraphrase, calculation, or reference mismatch."
                    recommendation = "Add a gold document or relevant chunk ID to the local evaluation case, then compare that source against the captured chunk list. Until then, review the passages and answer manually."
                result.update(
                    status="needs_review",
                    root_cause=None,
                    evidence_level="limited",
                    finding=finding,
                    finding_label=finding_label,
                    explanation=explanation,
                    recommended_fix=recommendation,
                )
                result["evidence"].append("Reference phrase was absent from captured context; this alone is not proof of a retrieval miss.")
                if run.get("gold_document_retrieved") is True:
                    result["evidence"].append(f"The labeled source document was present among the retrieved chunks{f' at rank {rank}' if rank else ''}.")
                    coverage_display = f"{topic_coverage:.0%}" if isinstance(topic_coverage, (int, float)) else "unavailable"
                    result["evidence"].append(f"Distinctive question-term coverage in captured chunk text: {coverage_display}. Matched: {', '.join(topic['matched_terms']) or '(none)'}; missing: {', '.join(topic['missing_terms']) or '(none)'}.")
                if derived:
                    result["evidence"].append("Question wording indicates a derived metric or interpretation.")
                return result
            return _attach_static_location(result, static_warnings)

        gold_candidates = [c for c in candidates if _answer_present(expected, c.get("text", ""))]
        if gold_candidates and top_k:
            best_candidate = min(gold_candidates, key=lambda c: _as_int(c.get("final_rank", c.get("rank")), 999999))
            best_rank = _as_int(best_candidate.get("final_rank", best_candidate.get("rank")), 999999)
            if best_rank > top_k:
                pre_rank = best_candidate.get("pre_rerank_rank")
                if best_candidate.get("rerank_score") is not None and pre_rank is not None and _as_int(pre_rank, 999999) <= top_k:
                    result.update(
                        root_cause=FailureLabel.RERANKER.value,
                        evidence_level="strong",
                        explanation=f"The expected fact was inside top_k before reranking (rank {pre_rank}) but was pushed to rank {best_rank}.",
                        recommended_fix="Review the reranker model and its query/document input formatting; compare pre-rerank and post-rerank ordering on this query.",
                    )
                    result["evidence"].append(f"Gold candidate moved from rank {pre_rank} to {best_rank} after reranking.")
                    return _attach_static_location(result, static_warnings)
                result.update(
                    root_cause=FailureLabel.RANKING_K.value,
                    evidence_level="strong",
                    explanation=f"The expected fact was retrieved at rank {best_rank}, beyond the final top_k={top_k} prompt cutoff.",
                    recommended_fix=f"Increase top_k_final from {top_k} or improve candidate ordering; measure the extra prompt-token cost on your evaluation set.",
                )
                result["evidence"].append(f"Best expected-answer candidate rank {best_rank} is outside top_k={top_k}.")
                return _attach_static_location(result, static_warnings)

        if oracle_passed is True:
            result.update(
                root_cause=FailureLabel.CONTEXT_NOISE.value,
                evidence_level="strong",
                explanation="The expected fact reached the prompt, but the full-context answer failed while the isolated-context oracle passed.",
                recommended_fix="Reduce irrelevant context, rerank before prompt assembly, or move the strongest evidence to the beginning of the prompt.",
            )
            result["evidence"].append("Isolated-context oracle passed while the captured full-context response missed the expected answer.")
            return _attach_static_location(result, static_warnings)
        if oracle_passed is False:
            result.update(
                root_cause=FailureLabel.GENERATOR.value,
                evidence_level="strong",
                explanation="The expected fact was retrieved, and the isolated-context oracle also failed to answer correctly.",
                recommended_fix="Review the system prompt and answer format constraints, then compare with a stronger or locally available model at temperature 0.",
            )
            result["evidence"].append("Isolated-context oracle failed.")
            return _attach_static_location(result, static_warnings)

        result.update(
            root_cause=FailureLabel.GENERATOR.value,
            evidence_level="moderate",
            explanation="The expected fact appears in captured context that reached the model, but is missing from the answer. Without an oracle result, context overload and generator failure cannot be separated confidently. Prompt-extracted context does not reveal retriever IDs or ranks.",
            recommended_fix="Capture an oracle result using the expected passage alone. Inspect prompt instructions and context length; use explicit retrieval telemetry before changing top_k or ranking settings.",
        )
        result["evidence"].append("Expected answer appears in captured context but not in the response.")
        return _attach_static_location(result, static_warnings)

    if not question:
        result.update(
            status="needs_query_or_reference",
            root_cause=None,
            evidence_level="limited",
            explanation="Retrieved passage text is available, but the question and a labeled reference were excluded. RAG Doctor can show the retrieved chunks, but cannot judge their relevance or verify the answer without query/reference evidence.",
            recommended_fix="Review the retrieved passages and source IDs, or provide a non-sensitive gold document ID/rank from an evaluation case. Prompt and response text can remain excluded.",
        )
        result["evidence"].append("Retrieved passages were captured; the query and answer were not included in this privacy mode.")
        return result

    q = _tokens(question)
    c = _tokens(joined_context)
    overlap = len(q & c) / max(1, len(q))
    result["query_context_overlap"] = round(overlap, 3)
    if overlap < 0.12:
        result.update(
            root_cause=FailureLabel.RETRIEVAL.value,
            evidence_level="limited",
            explanation=f"Only {overlap:.0%} of query terms appear in the retrieved context. This is a retrieval warning, not proof that the response is factually wrong.",
            recommended_fix="Inspect retrieved candidates and metadata filters; add a labeled evaluation case before changing retrieval settings.",
        )
        result["evidence"].append(f"Query/context lexical overlap: {overlap:.0%}.")
    else:
        result.update(
            status="needs_reference",
            explanation="The run has context and some query/context overlap, but no expected answer or oracle result was supplied. RAG Doctor cannot verify factual correctness from this evidence alone.",
            recommended_fix="Add a known expected answer or provide an isolated-context oracle result for this query.",
        )
        result["evidence"].append(f"Query/context lexical overlap: {overlap:.0%}; factual correctness remains unverified.")
    return _attach_static_location(result, static_warnings)


def _attach_static_location(result: Dict[str, Any], warnings: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not warnings:
        return result
    root = result.get("root_cause")
    related = next((w for w in warnings if w.get("root_cause") == root), None)
    if related:
        result["source_location"] = {"file": related.get("file"), "line": related.get("line")}
        result["evidence"].append(f"Related code finding at {related.get('file')}:{related.get('line')}.")
        result["recommended_fix"] = f"{related.get('fix_suggestion')} Review whether this setting explains the captured run."
    return result
