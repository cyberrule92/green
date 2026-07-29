"""Authenticated symmetric encryption for secrets at rest.

Stdlib-only, because the project has no `cryptography` dependency: an HMAC-SHA256
keystream in counter mode for confidentiality, with encrypt-then-MAC for
integrity. Ciphertext is urlsafe-base64 of ``nonce || ct || tag``.

Originally lived in `workflows.py` to protect stored HTTP credentials. It moved
here when the workflow engine was removed, because the Hugging Face token used by
`model_onboarding` is still encrypted with it (`HF_TOKEN_ENC`) and a token sitting
in plaintext `.env` is readable by anything that can read the file.

The ``wf-secret-v1`` domain-separation tag is deliberately unchanged: altering it
would invalidate every value already encrypted under the old module.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import secrets as _secrets

_SECRET_NONCE_LEN = 16
_SECRET_TAG_LEN = 32


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(_hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(out[:length])


def secret_encrypt(plaintext: str, key: bytes) -> str:
    """Encrypt+authenticate a UTF-8 string; returns a urlsafe-base64 token."""
    data = plaintext.encode("utf-8")
    nonce = _secrets.token_bytes(_SECRET_NONCE_LEN)
    ct = bytes(a ^ b for a, b in zip(data, _keystream(key, nonce, len(data))))
    tag = _hmac.new(key, b"wf-secret-v1" + nonce + ct, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(nonce + ct + tag).decode("ascii")


def secret_decrypt(token: str, key: bytes) -> str:
    """Inverse of :func:`secret_encrypt`. Raises ``ValueError`` on tamper/format."""
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
    except Exception as exc:  # noqa: BLE001 - malformed token
        raise ValueError("malformed secret token") from exc
    if len(raw) < _SECRET_NONCE_LEN + _SECRET_TAG_LEN:
        raise ValueError("secret token too short")
    nonce, tag = raw[:_SECRET_NONCE_LEN], raw[-_SECRET_TAG_LEN:]
    ct = raw[_SECRET_NONCE_LEN:-_SECRET_TAG_LEN]
    expected = _hmac.new(key, b"wf-secret-v1" + nonce + ct, hashlib.sha256).digest()
    if not _hmac.compare_digest(tag, expected):
        raise ValueError("secret token failed authentication")
    return bytes(a ^ b for a, b in zip(ct, _keystream(key, nonce, len(ct)))).decode("utf-8")
