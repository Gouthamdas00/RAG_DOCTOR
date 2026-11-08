"""Small, deterministic checks for explicit arithmetic written in model answers.

This is a verifier signal, not a general-purpose math parser. It only audits
simple numeric equations that are explicitly present in the response.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional


_NUMBER = r"\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
_AVERAGE = re.compile(
    r"\\frac\s*\{\s*\\?\$?\s*([+-]?[\d,]+(?:\.\d+)?)\s*([+\-])\s*"
    r"\\?\$?\s*([+-]?[\d,]+(?:\.\d+)?)\s*\}\s*\{\s*"
    r"\\?\$?\s*([+-]?[\d,]+(?:\.\d+)?)\s*\}\s*(=|≈|~=|\\approx)\s*"
    r"\\?\$?\s*([+-]?[\d,]+(?:\.\d+)?)"
)
_FRACTION = re.compile(
    r"\\frac\s*\{\s*\\?\$?\s*([+-]?[\d,]+(?:\.\d+)?)\s*\}\s*"
    r"\{\s*\\?\$?\s*([+-]?[\d,]+(?:\.\d+)?)\s*\}\s*(=|≈|~=|\\approx)\s*"
    r"\\?\$?\s*([+-]?[\d,]+(?:\.\d+)?)"
)
_SIMPLE_EQUATION = re.compile(
    rf"{_NUMBER}\s*([+\-×*/÷])\s*{_NUMBER}\s*(=|≈|~=|\\approx)\s*{_NUMBER}"
)


def _decimal(value: str) -> Decimal:
    return Decimal(value.replace(",", "").replace("$", "").strip())


def _rounded_close(computed: Decimal, reported: Decimal) -> bool:
    """Allow only the rounding implied by the displayed result's decimals."""
    places = max(0, -reported.as_tuple().exponent)
    tolerance = Decimal("0.5") * (Decimal(10) ** -places)
    return abs(computed - reported) <= tolerance


def audit_arithmetic(answer: str) -> List[Dict[str, Any]]:
    """Return explicit response equations whose displayed result is inconsistent."""
    text = str(answer or "")
    findings: List[Dict[str, Any]] = []
    occupied: List[tuple] = []

    def add(match: re.Match, kind: str, expression: str, operands: List[str], op: str, reported_raw: str) -> None:
        try:
            values = [_decimal(value) for value in operands]
            reported = _decimal(reported_raw)
            if kind == "average":
                computed = (values[0] + values[1]) / values[2]
            elif kind == "fraction":
                if values[1] == 0:
                    return
                computed = values[0] / values[1]
            elif op == "+":
                computed = values[0] + values[1]
            elif op == "-":
                computed = values[0] - values[1]
            elif op in {"×", "*"}:
                computed = values[0] * values[1]
            elif op in {"/", "÷"}:
                if values[1] == 0:
                    return
                computed = values[0] / values[1]
            else:
                return
        except (InvalidOperation, ZeroDivisionError, ValueError):
            return
        occupied.append(match.span())
        if _rounded_close(computed, reported):
            return
        findings.append({
            "kind": "ARITHMETIC_MISMATCH",
            "expression": expression[:240],
            "computed": format(computed.normalize(), "f"),
            "reported": format(reported.normalize(), "f"),
            "evidence": "The response contains an explicit arithmetic expression whose displayed result does not match its operands, even after display rounding.",
        })

    for match in _AVERAGE.finditer(text):
        left, operator, right, divisor, _, reported = match.groups()
        add(match, "average", match.group(0), [left, right, divisor], operator, reported)

    for match in _FRACTION.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        numerator, denominator, _, reported = match.groups()
        add(match, "fraction", match.group(0), [numerator, denominator], "/", reported)

    for match in _SIMPLE_EQUATION.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        left, operator, right, _, reported = match.groups()
        add(match, "simple", match.group(0), [left, right], operator, reported)

    return findings
