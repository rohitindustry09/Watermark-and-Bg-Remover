#!/usr/bin/env bash
# Everyday commands for the GPU server: start/stop the inference server, learn the watermark pattern, run a small test, run everything.
#   bash scripts/server-run.sh help
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh"

PORT="${INFERENCE_PORT:-8787}"
URL="http://127.0.0.1:$PORT"
LOG="${SERVER_LOG:-$HOME/server.log}"
PATTERN="${WM_PATTERN:-$HOME/wm-pattern.json}"
MODEL="${MODEL:-birefnet}"      # birefnet (full, most precise) | birefnet-lite | rmbg-1.4 | modnet
JOBS="${JOBS:-4}"               # images in flight in each part
SHARDS="${SHARDS:-2}"           # parts of the folder processed side by side by separate processes (full only)
QUALITY="${QUALITY:-90}"        # WebP quality
SAMPLES="${SAMPLES:-120}"       # photos the watermark pattern is learned from
TEST_COUNT="${TEST_COUNT:-15}"

healthy() { curl -fs --max-time 2 "$URL/api/health" >/dev/null 2>&1; }

start_server() {
  if healthy; then
    echo "The inference server is already running."
    return
  fi
  (ORT_PROVIDERS="${ORT_PROVIDERS:-cuda,cpu}" nohup node server/index.mjs >"$LOG" 2>&1 &)
  for _ in $(seq 1 30); do
    healthy && break
    sleep 1
  done
  if ! healthy; then
    echo "The server did not start. Last lines of $LOG:"
    tail -8 "$LOG"
    exit 1
  fi
  tail -1 "$LOG"
}

stop_server() {
  command -v pkill >/dev/null || { echo "pkill is not available here; stop the node process that runs server/index.mjs yourself."; return 1; }
  pkill -f "server/index.mjs" 2>/dev/null && echo "Inference server stopped." || echo "No inference server was running."
}

need_dir() {
  if [ -z "${1:-}" ] || [ ! -d "$1" ]; then
    echo "Give the folder of ORIGINAL photos (with the watermark and the background), for example:  $0 $2 ~/Brass"
    exit 2
  fi
}

learn() {
  need_dir "${1:-}" learn
  node scripts/learn-wm.mjs "$1" --out "$PATTERN" --samples "$SAMPLES"
}

ensure_pattern() {
  if [ ! -s "$PATTERN" ]; then
    echo "No watermark pattern yet at $PATTERN: learning it from $1 first."
    learn "$1"
  fi
}

batch() { # <input> <output> [extra batch options]
  local in="$1" out="$2"
  shift 2
  npm run batch --silent -- "$in" "$out" --model "$MODEL" --format webp --quality "$QUALITY" --remove-watermarks --wm-template "$PATTERN" --jobs "$JOBS" "$@"
}

# Several parts of the folder at once: each part is its own process (its own main thread) and the parts share the one GPU server.
run_parts() { # <input> <output>
  local in="$1" out="$2" cores per i status=0
  cores="$(nproc)"
  per=$((cores / SHARDS))
  [ "$per" -ge 1 ] || per=1
  export WM_WORKERS="$per" # watermark-finder threads per part, so the parts together do not oversubscribe the cores
  local pids=()
  for i in $(seq 1 "$SHARDS"); do
    batch "$in" "$out" --shard "$i/$SHARDS" >"$HOME/part-$i.log" 2>&1 &
    pids+=("$!")
  done
  echo "Running $SHARDS parts side by side ($JOBS images in flight each, $per watermark threads each). Logs: ~/part-N.log"
  alive() { for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
  while alive; do
    sleep 20
    local line=""
    for i in $(seq 1 "$SHARDS"); do
      line+=" | part $i: $(tr '\r' '\n' <"$HOME/part-$i.log" | grep -E ' done,' | tail -1 | sed 's/ *$//')"
    done
    echo "$(date +%H:%M:%S)$line"
  done
  for p in "${pids[@]}"; do wait "$p" || status=1; done
  for i in $(seq 1 "$SHARDS"); do
    echo "--- part $i"
    tr '\r' '\n' <"$HOME/part-$i.log" | grep -E 'Finished|failed:|Watermark removed|This process used' || true
  done
  # one report for the whole run
  local reports=("$out"/watermark-report-part*.csv)
  if [ -e "${reports[0]}" ]; then
    { head -1 "${reports[0]}"; tail -q -n +2 "${reports[@]}" | sort; } >"$out/watermark-report.csv"
    rm -f "${reports[@]}"
  fi
  return "$status"
}

case "${1:-help}" in
  start) start_server ;;
  stop) stop_server ;;
  status) healthy && curl -s "$URL/api/health" && echo || echo "The inference server is not running." ;;
  learn) learn "${2:-}" ;;
  test)
    need_dir "${2:-}" test
    start_server
    ensure_pattern "$2"
    rm -rf "$HOME/test-in" "$HOME/test-out" "$HOME/test-debug" "$HOME/test-results.zip"
    mkdir -p "$HOME/test-in"
    find "$2" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | shuf -n "$TEST_COUNT" | while IFS= read -r f; do
      cp "$f" "$HOME/test-in/$(basename "$(dirname "$f")" | cut -c1-24)-$(basename "$f")"
    done
    echo "Test images: $(ls "$HOME/test-in" | wc -l)"
    batch "$HOME/test-in" "$HOME/test-out" --debug-masks "$HOME/test-debug"
    (cd "$HOME" && zip -qr test-results.zip test-out test-debug)
    echo
    echo "Results: $HOME/test-results.zip ($(du -h "$HOME/test-results.zip" | cut -f1)). Download it from your PC with:"
    echo "  scp -P <port> ubuntu@<server-ip>:~/test-results.zip ."
    ;;
  full)
    need_dir "${2:-}" full
    OUT="${3:-$HOME/out}"
    start_server
    ensure_pattern "$2"
    STARTED=$SECONDS
    if [ "$SHARDS" -le 1 ]; then
      batch "$2" "$OUT" --zip "${OUT%/}.zip"
    else
      run_parts "$2" "$OUT" || echo "Some parts reported failed images: see ~/part-N.log (a re-run only redoes those)."
      (cd "$OUT" && zip -0qr "${OUT%/}.zip" .)
    fi
    echo
    echo "Finished in $(((SECONDS - STARTED) / 60)) min. Download ${OUT%/}.zip from your PC (scp), then stop or destroy the server so it stops billing."
    ;;
  help | *)
    cat <<'EOF'
Usage: bash scripts/server-run.sh <command> [folder]

  start            start the inference server in the background (log: ~/server.log)
  stop             stop it
  status           is it running?
  learn  <folder>  learn the watermark pattern from a folder of original photos (-> ~/wm-pattern.json)
  test   <folder>  run a 15-image test: learns the pattern if needed, writes ~/test-results.zip to download and check
  full   <folder> [out]   run every image in <folder>; results in [out] (default ~/out) and a zip next to it.
                          Finished files are skipped, so after an interruption run the same command again.

Settings (environment variables): MODEL (default birefnet), JOBS (4 images in flight per part), SHARDS (2 parts side by side),
QUALITY (90), SAMPLES (120), TEST_COUNT (15). Example:  SHARDS=3 bash scripts/server-run.sh full ~/Brass
EOF
    ;;
esac
