# Changelog

## 0.2.0 (2026-11-20)

### Added
- **Built-in response skills** (`skills.py`): the proxy adds a constant instruction block
  to every routed request, on the OpenAI routes and Claude Code's main loop. Default
  `caveman` (terse prose, normal code) cut output tokens by about half in a controlled A/B
  test. Configure under `skills:` in `policy.yaml` or with `RCE_SENTINEL_SKILLS`.
- **Answer cache shared across models** (`memory/answers.py`): exact repeats on the
  OpenAI-compatible proxy are answered from signed SQLite rows with no model call, as JSON
  or a stream. `rce-sentinel cache stats`, `cache clear --which answers|verdicts|all`,
  header `x-rce-cache: off` to bypass.
- **Proxy token** (`RCE_SENTINEL_PROXY_TOKEN`): required on every route but `/health`
  when set, never forwarded upstream; binding beyond loopback without one is refused.
- **Security posture check** (`rce-sentinel doctor --posture`): key and database modes,
  git exposure of the state directory, token strength, rows with bad signatures.
- **Pre-write screen** for model text that later reaches another model (facts, packs).
- 16 hard multi-file build prompts in `demo/bench/dev.yaml`; `bench report --only GLOB`
  scores a slice of the held-out prompts.

### Changed
- **Context pack v3**: only the user's own decisions bind. A model's choices that are in
  the files are "keep consistent"; invented process rules, the plan and the next step are
  advisory; the pack ends with a scope line. Live handoff test (Haiku, Sonnet, Haiku):
  decisions kept went from 62% (v2) to 100%, with 4.0K handoff tokens against 16.2K for
  the full transcript. One run.
- **Seat handoff seam repair** (`orchestrator/seat.py`): when a model resumes a continued
  reply, `_trim_seam()` removes a re-opened fence that has a language (e.g. ` ```json`
  inside an open ```` ``` ````), and trims a repeated tail of 24+ chars containing letters.
  Fixes truncated Opus output splitting into two chunks at a fence boundary.
- The database is created owner-only (`0600`).
- Handoff events carry an idempotency key; a replayed event is logged once.

### Benchmark Results

**Live routing performance (dev@1, 51 test items, single run):**
- Router (calibrated): **90% right, $0.84 regret, $1.11 spend**
- Always Sonnet: 78% right, $1.01 regret, $1.51 spend
- Always Haiku: 76% right, $3.12 regret, $0.15 spend
- Always Opus: 35% right, $2.41 regret, $4.45 spend

Hard-build subset (17 test items): router and Sonnet both at 82% right; router $2.51
regret vs Sonnet $2.97. Router saves 0.17 / 0.46 on hard prompts. Opus fails 94% on hard
items (mostly 16K token truncation). Full methodology: `rce-sentinel bench report --set seat --models seat/claude-haiku-5-5,seat/claude-sonnet-5-5,seat/claude-opus-5-5`.
