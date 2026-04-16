"""
guardrails/guardrails.py
Input validation, SQL safety checks, and prompt-injection detection.
Every user input passes through run_guardrails() before reaching the LLM.
"""

import re, os
from typing import Tuple

# ── Config ────────────────────────────────────────────────────────────────────
MAX_LEN = int(os.getenv("MAX_INPUT_LENGTH", 2000))

BLOCKED_SQL = [
    kw.strip().upper()
    for kw in os.getenv(
        "BLOCKED_SQL_KEYWORDS",
        "DROP,DELETE,TRUNCATE,ALTER,CREATE,INSERT,UPDATE,EXEC,EXECUTE,GRANT,REVOKE"
    ).split(",")
]

# Prompt-injection patterns to block
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+instructions",
    r"disregard\s+your\s+(training|guidelines|rules)",
    r"you\s+are\s+now\s+(a\s+)?different",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak",
    r"<\s*system\s*>",
    r"act\s+as\s+if\s+you\s+(have\s+no|are)",
]

# PII patterns to redact from output
PII_PATTERNS = {
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "aadhar":      r"\b\d{4}\s\d{4}\s\d{4}\b",
    "pan":         r"\b[A-Z]{5}\d{4}[A-Z]\b",
}


# ── Main entry point ──────────────────────────────────────────────────────────
def run_guardrails(user_input: str) -> Tuple[bool, str]:
    """
    Validates user input. Returns (passed: bool, reason: str).
    Checks: length, empty, prompt-injection patterns.
    """
    if not user_input or not user_input.strip():
        return False, "Input cannot be empty."
    if len(user_input) > MAX_LEN:
        return False, f"Input too long (max {MAX_LEN} characters)."
    for pat in INJECTION_PATTERNS:
        if re.search(pat, user_input, re.IGNORECASE):
            return False, "Input contains potentially unsafe instructions."
    return True, "OK"


# ── SQL safety ────────────────────────────────────────────────────────────────
def validate_sql(sql: str) -> Tuple[bool, str]:
    """
    Validates a generated SQL query is read-only.
    Returns (safe: bool, reason: str).
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query."
    sql_up = sql.upper()
    for kw in BLOCKED_SQL:
        if re.search(r'\b' + re.escape(kw) + r'\b', sql_up):
            return False, f"Forbidden keyword: {kw}"
    stmts = [s.strip() for s in sql.split(";") if s.strip()]
    if len(stmts) > 1:
        return False, "Multiple statements not allowed."
    first = sql_up.strip().split()[0] if sql_up.strip() else ""
    if first not in ("SELECT", "WITH"):
        return False, "Only SELECT queries are permitted."
    if "--" in sql or "/*" in sql:
        return False, "SQL comments not permitted."
    return True, "OK"


# ── Output sanitiser ─────────────────────────────────────────────────────────
def sanitize_output(text: str) -> str:
    """Redacts PII patterns from any LLM-generated output."""
    for pii_type, pat in PII_PATTERNS.items():
        text = re.sub(pat, f"[REDACTED {pii_type.upper()}]", text)
    return text
