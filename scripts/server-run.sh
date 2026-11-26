#!/usr/bin/env bash
# Everyday commands for the GPU server: start/stop the inference server, learn the watermark pattern, run a small test, run everything.
#   bash scripts/server-run.sh help
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -s "$HOME/.nvm/nvm.sh" ]; then
  set +eu # nvm's own script is not written for bash's strict mode
  . "$HOME/.nvm/nvm.sh"
  set -eu
fi

PORT="${INFERENCE_PORT:-8787}"
URL="http://127.0.0.1:$PORT"
LOG="${SERVER_LOG:-$HOME/server.log}"
PATTERN="${WM_PATTERN:-$HOME/wm-pattern.json}"
BUNDLED_PATTERN="$ROOT/assets/wm-pattern.json"   # the pattern of the "Dharma Statues" watermark, learned from the real photos and shipped with the project
MODEL="${MODEL:-birefnet}"      # birefnet (full, most precise) | birefnet-lite | rmbg-1.4 | modnet
JOBS="${JOBS:-4}"               # images in flight in each part
CORES="$(nproc)"
DEFAULT_SHARDS=$((CORES / 4))     # one part per 4 cores, at least 2 and at most 8, so a bigger machine is used fully without tuning
[ "$DEFAULT_SHARDS" -ge 2 ] || DEFAULT_SHARDS=2
[ "$DEFAULT_SHARDS" -le 8 ] || DEFAULT_SHARDS=8
SHARDS="${SHARDS:-$DEFAULT_SHARDS}"   # parts of the folder processed side by side by separate processes (full only)
QUALITY="${QUALITY:-90}"        # WebP quality
MAX_EDGE="${MAX_EDGE:-2048}"    # longest side of the output in pixels: 2048 is about 2x faster and fits Shopify (0 = keep the original size)
SAMPLES="${SAMPLES:-120}"       # photos the watermark pattern is learned from
TEST_COUNT="${TEST_COUNT:-15}"
export LAMA_SESSIONS="${LAMA_SESSIONS:-1}"   # independent sessions of the watermark fill (1 is enough: a call takes ~60 ms on the GPU)

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
  if ! pkill -TERM -f "server/index.mjs" 2>/dev/null; then
    echo "No inference server was running."
    return 0
  fi
  # the server releases its models before it exits; give it time, and only then insist
  for _ in $(seq 1 15); do
    pgrep -f "server/index.mjs" >/dev/null 2>&1 || break
    sleep 1
  done
  pkill -KILL -f "server/index.mjs" 2>/dev/null || true
  echo "Inference server stopped."
}

need_zip() {
  command -v zip >/dev/null 2>&1 || { echo "zip is not installed, so the results could not be packed at the end. Install it first:  sudo apt install -y zip"; exit 1; }
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
  if [ -s "$PATTERN" ]; then
    return
  fi
  if [ -s "$BUNDLED_PATTERN" ]; then
    cp "$BUNDLED_PATTERN" "$PATTERN"
    echo "Using the watermark pattern that ships with the project (no learning needed). To learn a new one from your photos: bash scripts/server-run.sh learn <folder>"
    return
  fi
  echo "No watermark pattern yet at $PATTERN: learning it from $1 first."
  learn "$1"
}

batch() { # <input> <output> [extra batch options]
  local in="$1" out="$2"
  shift 2
  npm run batch --silent -- "$in" "$out" --model "$MODEL" --format webp --quality "$QUALITY" --remove-watermarks --wm-template "$PATTERN" --server "$URL" --jobs "$JOBS" $([ "$MAX_EDGE" != "0" ] && printf -- "--max-edge %s" "$MAX_EDGE") "$@"
}

# Several parts of the folder at once: each part is its own process (its own main thread) and the parts share the one GPU server.
run_parts() { # <input> <output>
  local in="$1" out="$2" per i status=0
  per=$((CORES / SHARDS))
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
      line+=" | part $i: $(tr '\r' '\n' <"$HOME/part-$i.log" | grep -E ' done,' | tail -1 | sed 's/ *$//' || true)"
    done
    echo "$(date +%H:%M:%S)$line"
  done
  local rc fatal=0
  for p in "${pids[@]}"; do
    rc=0
    wait "$p" || rc=$?
    [ "$rc" -eq 0 ] || status=1
    [ "$rc" -eq 4 ] && fatal=1
  done
  for i in $(seq 1 "$SHARDS"); do
    echo "--- part $i"
    tr '\r' '\n' <"$HOME/part-$i.log" | grep -E 'Finished|failed:|Watermark removed|This process used' || true
  done
  node scripts/merge-reports.mjs "$out" >/dev/null || true # keep what the parts reported so far, even for a stopped run
  if [ "$fatal" -eq 1 ]; then
    echo
    echo "THE RUN STOPPED because the GPU or the inference server stopped answering (see the STOPPED line above)."
    echo "If the GPU crashed: reboot the instance on the dashboard, log in again and run the same command again. Finished images are skipped."
    return 4
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
    need_zip
    start_server
    ensure_pattern "$2"
    rm -rf "$HOME/test-in" "$HOME/test-out" "$HOME/test-debug" "$HOME/test-results.zip"
    mkdir -p "$HOME/test-in"
    n=0
    while IFS= read -r f; do
      n=$((n + 1))
      cp "$f" "$HOME/test-in/$(printf '%03d' "$n")-$(basename "$f")" # numbered, so two folders' "1.jpg" never collide and no name is cut mid-character
    done < <(find "$2" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | shuf -n "$TEST_COUNT")
    echo "Test images: $(ls "$HOME/test-in" | wc -l)"
    batch "$HOME/test-in" "$HOME/test-out" --debug-masks "$HOME/test-debug" || echo "(some images failed: see the lines above. The results zip is still made.)"
    (cd "$HOME" && zip -qr test-results.zip test-out test-debug)
    echo
    echo "Results: $HOME/test-results.zip ($(du -h "$HOME/test-results.zip" | cut -f1)). Download it from your PC with:"
    echo "  scp -P <port> ubuntu@<server-ip>:~/test-results.zip ."
    ;;
  full)
    need_dir "${2:-}" full
    need_zip
    OUT="${3:-$HOME/out}"
    start_server
    ensure_pattern "$2"
    STARTED=$SECONDS
    if [ "$SHARDS" -le 1 ]; then
      batch "$2" "$OUT" --zip "${OUT%/}.zip"
    else
      rc=0
      run_parts "$2" "$OUT" || rc=$?
      [ "$rc" -eq 4 ] && exit 4 # stopped on a dead GPU: no zip of a half-finished set
      [ "$rc" -eq 0 ] || echo "Some parts reported failed images: see ~/part-N.log (a re-run only redoes those)."
      rm -f "${OUT%/}.zip" # zip would otherwise update an old archive and keep stale entries
      (cd "$OUT" && zip -0qr "${OUT%/}.zip" .)
    fi
    echo
    node scripts/merge-reports.mjs "$OUT" || true # one report for the whole run, and how many images need a look
    echo
    echo "Finished in $(((SECONDS - STARTED) / 60)) min. Download ${OUT%/}.zip from your PC (scp), then stop or destroy the server so it stops billing."
    ;;
  redo)
    need_dir "${2:-}" redo
    need_zip
    OUT="${3:-$HOME/out}"
    [ -s "$OUT/watermark-report.csv" ] || { echo "No $OUT/watermark-report.csv: run 'full' first."; exit 2; }
    start_server
    ensure_pattern "$2"
    rm -rf "$HOME/redo-in" "$HOME/redo-out" "$HOME/redo-debug" "$HOME/redo-results.zip"
    node scripts/redo-list.mjs "$OUT/watermark-report.csv" "$2" "$HOME/redo-in"
    [ -n "$(ls -A "$HOME/redo-in" 2>/dev/null)" ] || { echo "Nothing to redo."; exit 0; }
    # a looser copy of the pattern: the images in the report had a best match that was too weak to trust, so the best match is accepted here and the mask
    # overlays (redo-debug) are there to check by eye. The finished results in $OUT are not touched.
    node scripts/redo-list.mjs --relax "$PATTERN" "$HOME/wm-pattern-relaxed.json" "${REDO_THRESHOLD:-0.8}" "${REDO_STRENGTH:-0.35}"
    SAVED_PATTERN="$PATTERN"
    PATTERN="$HOME/wm-pattern-relaxed.json"
    batch "$HOME/redo-in" "$HOME/redo-out" --wm-double-check --debug-masks "$HOME/redo-debug" || echo "(some images failed: see the lines above.)"
    PATTERN="$SAVED_PATTERN"
    (cd "$HOME" && zip -qr redo-results.zip redo-out redo-debug)
    echo
    echo "Redone images: ~/redo-out (same folder layout as $OUT); the places that were cleaned: ~/redo-debug (red = removed area)."
    echo "Download ~/redo-results.zip and check them. To use them in the finished set:  cp -r ~/redo-out/. $OUT/   then rebuild the zip:  (cd $OUT && rm -f ../out.zip && zip -0qr ../out.zip .)"
    ;;
  help | *)
    cat <<'EOF'
Usage: bash scripts/server-run.sh <command> [folder]

  start            start the inference server in the background (log: ~/server.log)
  stop             stop it
  status           is it running?
  learn  <folder>  learn a new watermark pattern from a folder of original photos (-> ~/wm-pattern.json). Not needed for the
                   "Dharma Statues" watermark: its pattern ships with the project and is used automatically.
  test   <folder>  run a 15-image test: learns the pattern if needed, writes ~/test-results.zip to download and check
  full   <folder> [out]   run every image in <folder>; results in [out] (default ~/out) and a zip next to it.
                          Finished files are skipped, so after an interruption run the same command again.
  redo   <folder> [out]   run again only the images the report lists as uncertain, with a looser matcher; results in ~/redo-out (not mixed into [out]) and
                          ~/redo-results.zip with mask overlays to check by eye

Settings (environment variables): MODEL (default birefnet), JOBS (4 images in flight per part), SHARDS (parts side by side: one per 4 CPU cores, at least 2),
QUALITY (90), MAX_EDGE (2048; 0 = original size), LAMA_SESSIONS (1), SAMPLES (120), TEST_COUNT (15).
Example:  MAX_EDGE=0 bash scripts/server-run.sh full ~/Brass   (keep the original size)
EOF
    ;;
esac
