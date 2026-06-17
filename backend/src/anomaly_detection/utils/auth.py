"""Authentication utilities for secure password hashing and verification."""

from __future__ import annotations

import hashlib
import secrets


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2 with SHA-256 and a random salt."""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    ).hex()
    return f"{salt}:{pwd_hash}"


def verify_password(password: str, stored_password: str) -> bool:
    """Verify a password against its stored hash."""
    try:
        salt, stored_hash = stored_password.split(":")
        ref_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,
        ).hex()
        return secrets.compare_digest(stored_hash, ref_hash)
    except Exception:
        return False
