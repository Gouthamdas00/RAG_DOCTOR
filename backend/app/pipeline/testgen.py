import re
from typing import List
from backend.app.schemas.contracts import BenchmarkItem, CharSpan, Document


class SyntheticBenchmarkGenerator:
    """
    Generates a labelled evaluation suite (question, gold_answer, gold_span, question_type)
    directly from raw Documents, including lexical-overlap calculation and unanswerable probes.
    """

    def generate_from_corpus(self, documents: List[Document]) -> List[BenchmarkItem]:
        benchmarks: List[BenchmarkItem] = []
        q_counter = 1

        for doc in documents:
            sentences = [
                m.group(0).strip()
                for m in re.finditer(r"[^.!?\n]+[.!?]", doc.text)
                if len(m.group(0).strip()) > 35
            ]
            for sent in sentences:
                # Look for factual sentences containing numbers/units or policy rules
                num_match = re.search(
                    r"\b(\d+(?:\s*(?:seconds|minutes|hours|days|requests per minute|times|Mi|GB|%))?)\b",
                    sent,
                )
                if not num_match:
                    continue

                gold_answer = num_match.group(1).strip()
                start_idx = doc.text.index(sent)
                end_idx = start_idx + len(sent)

                # Formulate a paraphrased question by masking the number/target
                masked = sent.replace(gold_answer, "what specific value or threshold")
                masked = re.sub(r"^Section \d+:\s*", "", masked).strip(". ")
                question = f"According to {doc.title}, {masked}?"

                q_type = "numeric" if any(ch.isdigit() for ch in gold_answer) else "single_hop"
                benchmarks.append(
                    BenchmarkItem(
                        question_id=f"syn_q{q_counter}",
                        question=question,
                        gold_answer=gold_answer,
                        gold_doc_id=doc.doc_id,
                        gold_span=CharSpan(start_char=start_idx, end_char=end_idx),
                        question_type=q_type,
                    )
                )
                q_counter += 1

        # Always inject an unanswerable adversarial probe to test ABSTENTION
        if documents:
            benchmarks.append(
                BenchmarkItem(
                    question_id=f"syn_q{q_counter}_unans",
                    question="What is the quantum blockchain billing rate for Mars satellite regions?",
                    gold_answer="UNANSWERABLE",
                    gold_doc_id=documents[0].doc_id,
                    gold_span=CharSpan(start_char=0, end_char=0),
                    question_type="unanswerable",
                )
            )

        return benchmarks
