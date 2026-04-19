"""
services/memory.py
Short-term and long-term memory helpers for the assistant.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from db.models import LongTermMemory, ShortTermMemory


def remember_turn(
    db: Session,
    user_id: int,
    session_token: str,
    user_message: str,
    assistant_message: str,
    route: str | None,
) -> None:
    entries = [
        ShortTermMemory(
            user_id=user_id,
            session_token=session_token,
            role="user",
            content=user_message,
            route=route,
        ),
        ShortTermMemory(
            user_id=user_id,
            session_token=session_token,
            role="assistant",
            content=assistant_message,
            route=route,
        ),
    ]
    db.add_all(entries)
    db.commit()


def recent_short_term_memory(
    db: Session,
    user_id: int,
    session_token: str,
    limit: int = 8,
) -> list[ShortTermMemory]:
    rows = (
        db.query(ShortTermMemory)
        .filter(
            ShortTermMemory.user_id == user_id,
            ShortTermMemory.session_token == session_token,
        )
        .order_by(ShortTermMemory.created_at.desc(), ShortTermMemory.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def long_term_memory(db: Session, user_id: int, limit: int = 5) -> list[LongTermMemory]:
    return (
        db.query(LongTermMemory)
        .filter(LongTermMemory.user_id == user_id)
        .order_by(LongTermMemory.updated_at.desc(), LongTermMemory.id.desc())
        .limit(limit)
        .all()
    )


def extract_reflection_note(user_message: str, assistant_message: str) -> dict[str, str] | None:
    text = user_message.strip()
    lower = text.lower()
    patterns: list[tuple[str, str]] = [
        ("preference", "prefer "),
        ("preference", "always "),
        ("profile", "my name is "),
        ("profile", "i am "),
        ("business", "our shop "),
        ("memory", "remember that "),
    ]
    for category, trigger in patterns:
        if trigger in lower:
            return {
                "category": category,
                "summary": text[:280],
                "source_excerpt": assistant_message[:280],
            }
    return None


def upsert_long_term_memory(db: Session, user_id: int, insight: dict[str, str]) -> None:
    existing = (
        db.query(LongTermMemory)
        .filter(
            LongTermMemory.user_id == user_id,
            LongTermMemory.category == insight["category"],
            LongTermMemory.summary == insight["summary"],
        )
        .first()
    )
    if existing:
        existing.source_excerpt = insight.get("source_excerpt")
    else:
        db.add(LongTermMemory(user_id=user_id, **insight))
    db.commit()


def build_memory_context(db: Session, user_id: int | None, session_token: str | None) -> dict[str, Any]:
    if not user_id or not session_token:
        return {"short_term": [], "long_term": []}

    short_term = recent_short_term_memory(db, user_id, session_token)
    long_term = long_term_memory(db, user_id)
    return {
        "short_term": [
            {"role": row.role, "content": row.content, "route": row.route}
            for row in short_term
        ],
        "long_term": [
            {"category": row.category, "summary": row.summary}
            for row in long_term
        ],
    }
