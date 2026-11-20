#!/usr/bin/env bash
# Real check without OpenRouter: Claude through your Claude Code login, plus local models.
#
#   ./demo/seat_run.sh
#
# Env: BENCH_USD (default 5) caps the benchmark pass (API-equivalent dollars; resumable).
# OLLAMA_BASE_URL / APFEL_BASE_URL point at the local servers if not on default ports.
# Everything is logged to .rce-sentinel/seat_run-<time>.log for inspection.
set -uo pipefail

cd "$(dirname "$0")/.."
BENCH_USD="${BENCH_USD:-5}"
PORT="${PORT:-8787}"
export RCE_SENTINEL_PROVIDER=auto
unset RCE_SENTINEL_OFFLINE

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
RS() { "$PY" -m rce_sentinel "$@"; }
mkdir -p .rce-sentinel
LOG=".rce-sentinel/seat_run-$(date +%Y%m%d-%H%M%S).log"
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

seat_ready() {
  "$PY" - <<'PY'
import sys
from pathlib import Path
from rce_sentinel.config import load_settings
from rce_sentinel.handoff.session import find_model
from rce_sentinel.orchestrator.llm import ModelRegistry
from rce_sentinel.orchestrator.seat import claude_bin

s = load_settings(Path("."))
reg = ModelRegistry(s.models, "auto")
print("claude CLI:", claude_bin() or "not found")
ok = True
for m in ("seat/claude-haiku-5-5", "qwen2.5:3b", "apple-foundationmodel"):
    spec = find_model(reg, m)
    up = bool(spec and reg.is_available(spec))
    print(f"{m}: {'up' if up else 'down'}")
    ok &= up or m != "seat/claude-haiku-5-5"
sys.exit(0 if ok else 1)
PY
}

step "seat and local models reachable" seat_ready
[ ${#FAILED[@]} -gt 0 ] && { echo "Claude CLI not usable; see $LOG"; exit 1; }

step "benchmark: run (resumable, capped)" RS bench run --set seat --max-usd "$BENCH_USD"
step "benchmark: report" RS bench report --set seat
step "calibration after real evidence" RS calibrate

claude_proxy_check() {
  local bin pid ok=1 dir
  bin=$("$PY" -c "from rce_sentinel.orchestrator.seat import claude_bin; print(claude_bin())")
  RS serve --port "$PORT" > "$LOG.serve" 2>&1 &
  pid=$!
  for _ in $(seq 1 50); do curl -s "localhost:$PORT/health" > /dev/null && break; sleep 0.2; done
  dir=$(mktemp -d); echo "hello world" > "$dir/notes.txt"
  (cd "$dir" && ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT" "$bin" -p \
    "Read notes.txt and reply with its first word only." --model claude-opus-5-5 \
    --tools Read --setting-sources "" --strict-mcp-config --no-session-persistence) || ok=0
  kill "$pid"
  RS usage --days 1 || ok=0
  [ "$ok" -eq 1 ]
}
step "Claude Code through the proxy: routed, cached, logged" claude_proxy_check

printf '\n'
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All steps passed. Inspect the outputs in $LOG before trusting them."
else
  echo "Failed steps: ${FAILED[*]}. Details in $LOG"
  exit 1
fi
