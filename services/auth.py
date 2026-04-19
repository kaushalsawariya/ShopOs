"""
services/auth.py
Authentication helpers shared by FastAPI and Streamlit.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime

from sqlalchemy.orm import Session

from db.models import AuthSession, User


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}:{digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, digest_hex = stored_hash.split(":", 1)
    expected = bytes.fromhex(digest_hex)
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        100_000,
    )
    return hmac.compare_digest(candidate, expected)


def create_user(db: Session, full_name: str, email: str, password: str) -> User:
    normalized_email = email.strip().lower()
    existing = db.query(User).filter(User.email == normalized_email).first()
    if existing:
        raise ValueError("An account with this email already exists.")
    user = User(
        full_name=full_name.strip(),
        email=normalized_email,
        password_hash=_hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    normalized_email = email.strip().lower()
    user = db.query(User).filter(User.email == normalized_email).first()
    if not user or not _verify_password(password, user.password_hash):
        return None
    return user


def create_session(db: Session, user: User) -> AuthSession:
    session = AuthSession(
        user_id=user.id,
        token=secrets.token_urlsafe(32),
        last_seen_at=datetime.utcnow(),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def revoke_session(db: Session, token: str) -> None:
    session = db.query(AuthSession).filter(AuthSession.token == token).first()
    if session and not session.revoked:
        session.revoked = True
        session.last_seen_at = datetime.utcnow()
        db.commit()


def get_user_by_token(db: Session, token: str) -> User | None:
    session = (
        db.query(AuthSession)
        .filter(AuthSession.token == token, AuthSession.revoked.is_(False))
        .first()
    )
    if not session:
        return None
    session.last_seen_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session.user
