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


def main() -> None:
    verify_runtime()
    from fli.mcp._entry import run

    run()


if __name__ == "__main__":
    main()
