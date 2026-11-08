import re
from pathlib import Path
from typing import Any, Dict

# Compiled enterprise regex patterns for scrubbing secrets & PII in volatile memory
SECRET_PATTERNS = [
    (re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b(AIza[0-9A-Za-z\-_]{35})\b"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"\b(ghp_[a-zA-Z0-9]{36})\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"(?i)(bearer\s+[a-zA-Z0-9_\-\.]+)"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"(?i)(?:api[_-]?key|client[_-]?secret|password|access[_-]?token)\s*[:=]\s*['\"]?[^\s,;'\"]{8,}"), "[REDACTED_SECRET]"),
]

PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b"), "[REDACTED_PHONE]"),
]


class SecurityGuard:
    """
    Enforces industrial security guarantees:
      1. Volatile-Only PII & Secret Scrubbing (never logs credentials or sensitive user data).
      2. Strict Workspace Path Jailing (prevents directory traversal outside the project root).
      3. Prompt Injection Detection on retrieved context chunks.
    """

    @staticmethod
    def sanitize_text(text: str) -> str:
        if not text:
            return text
        cleaned = text
        for pattern, replacement in SECRET_PATTERNS + PII_PATTERNS:
            cleaned = pattern.sub(replacement, cleaned)
        return cleaned

    @staticmethod
    def validate_workspace_path(target_path: str, workspace_root: str) -> Path:
        root = Path(workspace_root).resolve()
        target = Path(target_path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise PermissionError(f"Security Violation: Path '{target}' is outside workspace root '{root}'.")
        return target

    @staticmethod
    def scan_for_prompt_injection(text: str) -> Dict[str, Any]:
        injection_Regex = re.compile(
            r"(?i)(ignore\s+all\s+previous\s+instructions|system\s+override|you\s+are\s+now\s+dan|disregard\s+above)"
        )
        match = injection_Regex.search(text or "")
        return {
            "injection_detected": bool(match),
            "matched_phrase": match.group(0) if match else None,
        }
