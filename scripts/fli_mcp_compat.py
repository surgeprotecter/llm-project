# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "flights[mcp] @ https://github.com/punitarani/fli/archive/af5d0ed82a8c1316ca142dcd0362574d9194135a.tar.gz",
# ]
# ///
"""Launch the proven fli transport without depending on a global tool upgrade.

Portable invocation: uv run --locked --script scripts/fli_mcp_compat.py
The pinned upstream source replaces the gated shopping RPC with the search
page transport and raises SearchParseError for missing payloads. The installed
MCP tools, flight serialization and round-trip assembly remain unchanged.
"""

from __future__ import annotations

import json
from importlib.metadata import distribution

UPSTREAM_REVISION = "af5d0ed82a8c1316ca142dcd0362574d9194135a"


def verify_runtime() -> None:
    package = distribution("flights")
    source = json.loads(package.read_text("direct_url.json") or "{}")
    if package.version != "0.10.0" or UPSTREAM_REVISION not in source.get("url", ""):
        raise RuntimeError(
            "Use uv run --locked --script scripts/fli_mcp_compat.py to install the pinned fli transport"
        )


def _locale_variant(url: str) -> str:
    """Return the same search URL with an English/US locale."""
    if "hl=" not in url or "gl=" not in url:
        return url
    parts = []
    for piece in url.split("&"):
        if piece.startswith("hl="):
            parts.append("hl=en")
        elif piece.startswith("gl="):
            parts.append("gl=US")
        else:
            parts.append(piece)
    return "&".join(parts)


def install_resilient_fetch() -> None:
    """Harden the search-page fetch against Google's intermittent shells.

    Live runs showed roughly a quarter of search-page fetches returning a
    JavaScript shell with no ``ds:1`` payload while the very same URL works
    moments later — the pinned upstream transport retries only three times
    with 0.5s/1.5s backoff, which is not enough when a burst of parallel
    round-trip expansions trips Google's throttling. This wrapper retries
    longer, rotates to an English/US locale on alternate rounds and leaves
    the parsing/serialization layers untouched.
    """

    import random
    import time

    from fli.search import _tfs, dates, flights

    attempts = 5
    backoff_seconds = (0.8, 1.6, 3.0, 5.0)

    def resilient_fetch(client, url: str):
        alternate = _locale_variant(url)
        for attempt in range(attempts):
            target = alternate if attempt % 2 else url
            payload = None
            try:
                response = client.get(target, impersonate="chrome", allow_redirects=True)
                response.raise_for_status()
                payload = _tfs.extract_payload(response.text)
            except Exception:  # noqa: BLE001 — network flake, retried below
                payload = None
            if payload is not None:
                return payload
            if attempt + 1 < attempts:
                delay = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                time.sleep(delay + random.uniform(0.0, 0.5))
        return None

    # ``flights`` and ``dates`` imported ``fetch_payload`` by name, so patch
    # their module globals as well as the shared helper.
    _tfs.fetch_payload = resilient_fetch
    flights.fetch_payload = resilient_fetch
    dates.fetch_payload = resilient_fetch


def main() -> None:
    verify_runtime()
    install_resilient_fetch()
    from fli.mcp._entry import run

    run()


if __name__ == "__main__":
    main()
