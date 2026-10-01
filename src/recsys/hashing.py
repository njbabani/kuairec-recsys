"""Deterministic, platform-independent hashing for assigning users to groups.

Used for the tune/test user split and, later, A/B arms. Python's built-in ``hash`` is salted per
process and polars' hash may change between releases, so assignments use SHA-256 instead.
"""

import hashlib

HASH_BYTES = 8
HASH_SPACE = 2 ** (8 * HASH_BYTES)


def stable_fraction(key: str) -> float:
    """Map ``key`` to a number in [0, 1) that never changes across runs, machines or versions."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:HASH_BYTES], "big") / HASH_SPACE
