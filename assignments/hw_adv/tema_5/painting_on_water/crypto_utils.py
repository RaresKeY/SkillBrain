import os
import hmac
import base64
import hashlib


def get_secret_key(key_name: str) -> bytes:
    """Get base64-encoded secret key from env and decode to bytes."""
    key_b64 = os.environ.get(key_name)
    if not key_b64:
        raise RuntimeError(f"{key_name} not set in environment")
    return base64.urlsafe_b64decode(key_b64.encode())


def hmac_sha256_hex(plaintext:str, secret_key: bytes) -> str:
    """Return HMAC-SHA256 hex digest of plaintext with secret key."""
    return hmac.new(secret_key, plaintext.encode(), hashlib.sha256).hexdigest()


def verify_hmac_sha256_hex(plaintext: str, secret_key: bytes, hashed: str) -> bool:
    """Check if plaintext + secret key matches given HMAC digest."""
    new_hash = hmac.new(secret_key, plaintext.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(new_hash, hashed)


def hmac_sha256_bytes(plaintext: str, secret_key: bytes) -> bytes:
    """Return HMAC-SHA256 raw bytes digest of plaintext with secret key."""
    return hmac.new(secret_key, plaintext.encode(), hashlib.sha256).digest()


def verify_hmac_sha256_bytes(plaintext: str, secret_key: bytes, hashed: bytes) -> bool:
    """Check if plaintext + secret key matches given HMAC bytes digest."""
    new_hash = hmac.new(secret_key, plaintext.encode(), hashlib.sha256).digest()
    return hmac.compare_digest(new_hash, hashed)


# Aliases
hmac_hex = hmac_sha256_hex
verify_hmac_hex = verify_hmac_sha256_hex
hmac_bytes = hmac_sha256_bytes
verify_hmac_bytes = verify_hmac_sha256_bytes