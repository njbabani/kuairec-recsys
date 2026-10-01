"""Download the KuaiRec archive with streaming, retries, MD5 verification and atomic writes.

Run as a DVC stage: ``python -m recsys.data.download``.
"""

import hashlib
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import httpx
from tqdm import tqdm

from recsys.config import load_params

logger = logging.getLogger(__name__)

CHUNK_SIZE_BYTES = 1 << 20
TIMEOUT_SECONDS = 60.0
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2.0
FIRST_SERVER_ERROR_STATUS = 500


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file does not match its expected MD5."""


def md5_of(path: Path, chunk_size: int = CHUNK_SIZE_BYTES) -> str:
    """Return the hex MD5 of a file, read in chunks so large archives never sit in memory."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_retryable(exc: httpx.HTTPError) -> bool:
    """Network failures and 5xx responses are transient; a 4xx means the request is wrong."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= FIRST_SERVER_ERROR_STATUS
    return isinstance(exc, httpx.TransportError)


def _stream_to(client: httpx.Client, url: str, target: Path) -> None:
    with client.stream("GET", url) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with (
            target.open("wb") as handle,
            tqdm(total=total, unit="B", unit_scale=True, desc=target.stem, disable=None) as bar,
        ):
            for chunk in response.iter_bytes(CHUNK_SIZE_BYTES):
                handle.write(chunk)
                bar.update(len(chunk))


def _fetch_with_retries(
    client: httpx.Client,
    url: str,
    target: Path,
    max_attempts: int,
    sleep: Callable[[float], None],
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            _stream_to(client, url, target)
            return
        except httpx.HTTPError as exc:
            if attempt == max_attempts or not _is_retryable(exc):
                raise
            delay = BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
            logger.warning(
                "Attempt %d/%d failed (%s); retrying in %.0fs", attempt, max_attempts, exc, delay
            )
            sleep(delay)


def download_file(
    url: str,
    dest: Path,
    expected_md5: str,
    client: httpx.Client | None = None,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download ``url`` to ``dest`` and verify its MD5.

    A verified existing copy is reused without touching the network. Transient failures are
    retried with exponential backoff. Data is streamed to a ``.part`` file and only moved into
    place once the checksum matches, so ``dest`` never holds a truncated or corrupted archive.
    """
    expected = expected_md5.lower()
    if dest.is_file() and md5_of(dest) == expected:
        logger.info("Verified copy already present at %s; skipping download", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    http_client = (
        nullcontext(client)
        if client is not None
        else httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True)
    )

    try:
        with http_client as http:
            logger.info("Downloading %s -> %s", url, dest)
            _fetch_with_retries(http, url, partial, max_attempts, sleep)
        actual = md5_of(partial)
        if actual != expected:
            raise ChecksumMismatchError(
                f"MD5 mismatch for {url}: expected {expected}, got {actual}"
            )
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)

    logger.info("Download verified (md5=%s)", expected)
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    params = load_params()
    download_file(str(params.data.url), params.data.archive_path, params.data.md5)


if __name__ == "__main__":
    main()
