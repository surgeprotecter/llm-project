# Flight service diagnosis — 2026-11-16

Run the portable MCP from the project root:

```sh
uv run --locked --script scripts/fli_mcp_compat.py
```

Merge the `google_flights` entry from `mcp_servers.fli-pinned.example.json` into the
active configuration. For callers outside the project root, use an absolute
script path. The script metadata pins official upstream revision
`af5d0ed82a8c1316ca142dcd0362574d9194135a` (package version 0.10.0), and the adjacent
`.lock` pins its dependency resolution. Existing MCP tool names and flat flight
leg fields are preserved. This does not rely on changes to global site-packages.
The local global tool was also upgraded to that revision during diagnosis.

Upstream documents that the legacy `GetShoppingResults` RPC can return HTTP 200
without a payload because it requires a browser-generated header. The pinned
revision reads the public flight search page's inline shopping payload and
raises a parse error when it cannot decode it:
[official source](https://github.com/punitarani/fli/tree/af5d0ed82a8c1316ca142dcd0362574d9194135a).

Exactly three flight MCP searches were run, with zero model calls:

| Search | Result |
| --- | --- |
| 0.9.0 HKG–DAD, 2026-12-10 through 2026-12-13 | Initial and two return-expansion RPCs returned HTTP 200 and non-null first WRB payloads. The diagnostic HTTP cap stopped further default expansion; this run is not a completed service failure or a successful round trip. |
| Pinned upstream HKG–CAI, 2026-12-23 through 2027-01-05, `top_n=1`, one adult | Two complete round-trip combinations, CNY 5,589 each. Two HTTP requests. Real serialized results saved. |
| Pinned project launcher CAI–ASW, 2026-12-27, one adult | HTTP 200, 1,944,860 response bytes, five inline callbacks, but `extract_payload` returned `None`. The diagnostic cap prevented retries. No nonempty CAI–ASW result was verified. |

There were six actual flight HTTP requests including expansion. URLs are logged
without query values; logs contain status, content type, lengths and structural
counts, never cookies, request bodies or raw response bodies.

Evidence: `tmp/fli-http-diagnostics.jsonl`,
`tmp/fli-upstream-egypt-roundtrip-result.json`, and
`tmp/fli-upstream-egypt-domestic-result.json`.

The earlier raw RPC bodies were not retained, so their two WRB frames cannot be
retroactively decoded. The original 0.9.0 probe did reach return-expansion calls,
which means its first frame produced outbound flights. A first-metadata-frame
bug is therefore unproven; no speculative selector monkey patch was added.
The diagnostic harness now counts all decoded WRB frames and both shopping
branches, including decoder successes and exception classes, for future probes.
No additional network queries were run after the third search.

The project adapter now also treats an explicit embedded `success: false` as a
failed invocation even if the MCP envelope's `isError` is false. Genuine
`success: true, count: 0` results retain their meaning. This prevents parse errors
from appearing as successful flight searches.

## 2026-11-24 — intermittent `ds:1` misses

Real runs started failing with `Search failed: Search page carried no ds:1
payload` for identical routes that had worked minutes earlier (PVG,SHA →
NRT,HND, 2 adults, 2026-12-02 → 2026-12-05). Manual probes reproduced it in
roughly a quarter of back-to-back fetches, with no consent/captcha interstitial
and normal page bytes, so this is Google intermittently serving the JS shell
under bursty traffic rather than a changed page shape.

The pinned transport retried only 3 times (0.5s/1.5s). `scripts/fli_mcp_compat.py`
now calls `install_resilient_fetch()` before starting the MCP server, which
replaces `fetch_payload` in `fli.search._tfs`, `fli.search.flights` and
`fli.search.dates` with a version that:

- retries up to 5 times with 0.8s/1.6s/3s/5s backoff plus jitter;
- alternates the original URL with an `hl=en&gl=US` variant (currency and
  other params untouched), so a locale-specific shell can fall back.

Verification: simulated misses (fake client answering a payload-less shell for
the first 2 and first 4 calls) now succeed on attempt 3 and 5; an always-shell
client gives up after 5 attempts (~11s) instead of 3 (~2s). Live checks after
the patch: PVG→BKK 2026-12-10 (20 flights, 1.0s), PEK→HND 2026-12-12 → 2026-12-18
(13 flights, 2.2s), and an agent turn on flifix-1202 stored 24 round-trip
options (cheapest Korean Air KE892/KE711/KE714/KE881, CNY 5,610). `tests/unit/
test_fli_compat.py` covers the locale-variant helper.
