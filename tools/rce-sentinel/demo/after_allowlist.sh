#!/usr/bin/env bash
# Real end-to-end check once openrouter.ai is reachable. Spends real money, capped below.
#
#   ./demo/after_allowlist.sh [path/to/.env]
#
# Env: SESSION_USD (default 3) caps each switch-test run, BENCH_USD (default 5) caps this
# benchmark pass. The benchmark resumes, so re-run to grade more items.
# Everything is logged to .rce-sentinel/after_allowlist-<time>.log for inspection.
set -uo pipefail

cd "$(dirname "$0")/.."
ENV_FILE="${1:-${RCE_SENTINEL_ENV_FILE:-}}"
SESSION_USD="${SESSION_USD:-3}"
BENCH_USD="${BENCH_USD:-5}"
PORT="${PORT:-8787}"
export RCE_SENTINEL_PROVIDER=auto
unset RCE_SENTINEL_OFFLINE
[ -n "$ENV_FILE" ] && export RCE_SENTINEL_ENV_FILE="$ENV_FILE"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
RS() { "$PY" -m rce_sentinel "$@"; }
mkdir -p .rce-sentinel
LOG=".rce-sentinel/after_allowlist-$(date +%Y%m%d-%H%M%S).log"
FAILED=()

step() {
  local name="$1"; shift
  printf '\n=== %s\n' "$name" | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "--- ok: $name" | tee -a "$LOG"
  else
    echo "--- FAILED: $name" | tee -a "$LOG"
    FAILED+=("$name")
  fi
}

reachable() {
  local code
  code=$("$PY" - <<'EOF'
import httpx, rce_sentinel  # noqa: F401  (rce_sentinel turns on the OS trust store)
try:
    r = httpx.get("https://openrouter.ai/api/v1/models", timeout=10)
    print(f"{r.status_code} {r.headers.get('server', '')}")
except httpx.HTTPError as exc:
    print(f"error {type(exc).__name__}")
EOF
)
  echo "openrouter.ai: $code"
  [[ "$code" == 200* ]]
}

step "openrouter.ai reachable" reachable || true
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "openrouter.ai is still blocked; nothing else will work. See $LOG"
  exit 1
fi

step "doctor: real call to every OpenRouter model" RS doctor --provider openrouter
step "route: Jev tags a vague prompt" RS route "hmm the thing on the page looks off sometimes"
step "route: hard build, split per target" RS route --split \
  "Build a scientific calculator app with the best possible and cleanest UI for iOS and Android. It needs to be in React Native, Swift, and Flutter."
step "route: commit message stays cheap" RS route "write a commit message for the staged change"
step "switch test: fixed models, all handoff modes" RS session --scenario calculator \
  --max-usd "$SESSION_USD"
step "switch test: auto routing with split, pack mode" RS session --scenario calculator \
  --models auto --split --modes pack --max-usd "$SESSION_USD"
step "benchmark: run (resumable, capped)" RS bench run --max-usd "$BENCH_USD"
step "benchmark: report" RS bench report
step "calibration after real evidence" RS calibrate

proxy_check() {
  RS serve --port "$PORT" > "$LOG.serve" 2>&1 &
  local pid=$! ok=1
  for _ in $(seq 1 50); do curl -s "localhost:$PORT/health" > /dev/null && break; sleep 0.2; done
  curl -s -D - "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"model":"auto","messages":[{"role":"user","content":"explain the TCP handshake in two sentences"}]}' \
    | tee /dev/stderr | grep -qi '^x-rce-model' || ok=0
  curl -sN "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"write a commit message for: rename foo to bar"}]}' \
    | tee /dev/stderr | grep -q 'data: \[DONE\]' || ok=0
  kill "$pid"
  [ "$ok" -eq 1 ]
}
step "proxy: real routed reply and real stream" proxy_check

printf '\n'
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All steps passed. Inspect the outputs in $LOG before trusting them."
else
  echo "Failed steps: ${FAILED[*]}. Details in $LOG"
  exit 1
fi
