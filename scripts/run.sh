#!/usr/bin/env bash
# Run the measurements in results/ from scratch.
#
#   scripts/build.sh all          # once: build the runtimes
#   scripts/run.sh quick          # BitNet 2B4T + Bonsai 1.7B (~1-2 h)
#   scripts/run.sh full           # + Ternary-Bonsai-2 27B (needs >= 16 GB RAM)
#
# Every measurement is appended to $OUT ($WORK/results.jsonl) with the
# CPU, runtime commit, model file and flags; logs go to $WORK/logs/.
# THREADS (default 2) and WORK can be set in the environment.

. "$(dirname "$0")/common.sh"
. "$(dirname "$0")/measure.sh"

MODE=${1:-}
[ "$MODE" = quick ] || [ "$MODE" = full ] || die "usage: $0 quick|full"

SRC=$WORK/src
LL=$SRC/llama.cpp/build/bin
PR=$SRC/prism/build/bin
IK=$SRC/ik_llama.cpp/build/bin
for b in "$LL" "$PR" "$IK"; do
  [ -x "$b/llama-perplexity" ] || die "missing $b: run scripts/build.sh all"
done
FR="The capital of France is"
log "cpu: $(cpu_info) | mode: $MODE | out: $OUT"

m_load() { # label bin model: does the runtime load the file? (1 token)
  local label=$1 bin=$2 model=$3 tool rc err parsed
  tool=("$bin/llama-completion" -no-cnv)
  [ -x "$bin/llama-completion" ] || tool=("$bin/llama-cli")
  "${tool[@]}" -m "$model" -p "$FR" -n 1 --temp 0 -t "$THREADS" -c 512 \
    < /dev/null > /dev/null 2> "$WORK/logs/$label.load.log"; rc=$?
  err=$(grep -iE "error|failed|invalid|not a multiple|legacy" \
    "$WORK/logs/$label.load.log" | tail -1)
  parsed=$(python3 -c 'import json,sys; print(json.dumps(
    {"loads": sys.argv[1] == "0", "rc": int(sys.argv[1]), "message": sys.argv[2]}))' \
    "$rc" "$err")
  _record "$label" load "$bin" "$model" "$parsed"
  log "$label: rc=$rc ${err:0:80}"
}

garbage() { # label: was the last greedy text for this label garbage?
  python3 - "$OUT" "$1" <<'EOF'
import json, sys
text = ""
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["label"] == sys.argv[2] and r["kind"] == "greedy":
        text = r["text"]
sys.exit(0 if ("????" in text or "!!!!" in text or not text.strip()) else 1)
EOF
}

# ---------------------------------------------------------------- Bonsai 1.7B
for m in bonsai-q1_0 ternary-q2_0_g64 ternary-pq2_0 ternary-q2_0_legacy; do
  fetch_model "$m"
done
fetch_wikitext
Q1=$(model_path bonsai-q1_0); G64=$(model_path ternary-q2_0_g64)
PQ=$(model_path ternary-pq2_0); LEG=$(model_path ternary-q2_0_legacy)
mkdir -p "$WORK/logs" "$WORK/kld"

# which file loads where (incl. the legacy Q2_0 marked "recommended")
for rt in llama:$LL prism:$PR ik:$IK; do
  for f in q1_0:$Q1 q2_0_g64:$G64 pq2_0:$PQ q2_0_legacy:$LEG; do
    m_load "load-${rt%%:*}-${f%%:*}" "${rt#*:}" "${f#*:}"
  done
done

# greedy text + perplexity (8 x 512) for every supported pair
pairs="llama:$LL:q1_0:$Q1 prism:$PR:q1_0:$Q1 ik:$IK:q1_0:$Q1
llama:$LL:q2_0_g64:$G64 prism:$PR:q2_0_g64:$G64
prism:$PR:pq2_0:$PQ ik:$IK:pq2_0:$PQ"
for p in $pairs; do
  IFS=: read -r rt bin fl model <<< "$p"
  m_greedy "$rt-$fl" "$bin" "$model" 8 "$FR"
  m_ppl "$rt-$fl" "$bin" "$model" 8
done
for fl in q1_0:$Q1 pq2_0:$PQ; do
  m_ppl "ik-rtr-${fl%%:*}" "$IK" "${fl#*:}" 8 -rtr
done

# exact F16 references (exact for Q1_0 and for PQ2_0 without Hadamard)
F16Q1=$WORK/models/Bonsai-1.7B-F16-from-Q1_0.gguf
F16PQ=$WORK/models/Ternary-Bonsai-1.7B-F16-from-PQ2_0.gguf
[ -s "$F16Q1" ] || "$LL/llama-quantize" --allow-requantize "$Q1" "$F16Q1" F16 \
  > "$WORK/logs/quantize-q1-f16.log" 2>&1 || die "F16 from Q1_0"
[ -s "$F16PQ" ] || "$PR/llama-quantize" --allow-requantize "$PQ" "$F16PQ" F16 \
  > "$WORK/logs/quantize-pq-f16.log" 2>&1 || die "F16 from PQ2_0"
m_ppl llama-f16-from-q1_0 "$LL" "$F16Q1" 8
m_ppl prism-f16-from-pq2_0 "$PR" "$F16PQ" 8
m_kl_base base-f16-q1 "$LL" "$F16Q1" "$WORK/kld/f16-q1.kld" 8
m_kl_base base-f16-pq "$PR" "$F16PQ" "$WORK/kld/f16-pq.kld" 8
m_kl kl-llama-q1_0 "$LL" "$Q1" "$WORK/kld/f16-q1.kld"
m_kl kl-llama-q1_0-vecdot "$LL" "$Q1" "$WORK/kld/f16-q1.kld" --no-repack
m_kl kl-ik-q1_0 "$IK" "$Q1" "$WORK/kld/f16-q1.kld"
m_kl kl-ik-rtr-q1_0 "$IK" "$Q1" "$WORK/kld/f16-q1.kld" -rtr
m_kl kl-prism-pq2_0 "$PR" "$PQ" "$WORK/kld/f16-pq.kld"
m_kl kl-ik-pq2_0 "$IK" "$PQ" "$WORK/kld/f16-pq.kld"
m_kl kl-ik-rtr-pq2_0 "$IK" "$PQ" "$WORK/kld/f16-pq.kld" -rtr

# speed
for p in llama:$LL:q1_0:$Q1 prism:$PR:q1_0:$Q1 ik:$IK:q1_0:$Q1 \
         llama:$LL:q2_0_g64:$G64 prism:$PR:q2_0_g64:$G64 \
         prism:$PR:pq2_0:$PQ ik:$IK:pq2_0:$PQ; do
  IFS=: read -r rt bin fl model <<< "$p"
  m_bench "bench-$rt-$fl" "$bin" "$model" 512 64 3
done
for fl in q1_0:$Q1 pq2_0:$PQ; do
  m_bench "bench-ik-rtr-${fl%%:*}" "$IK" "${fl#*:}" 512 64 3 -rtr 1
done

# ---------------------------------------------------------- BitNet 2B4T
fetch_model bitnet-i2_s
BN=$(model_path bitnet-i2_s)
for v in official fixed 2025-06; do
  b=$SRC/bitnet-$v/build/bin
  if [ ! -x "$b/llama-perplexity" ]; then
    log "bitnet $v not built (scripts/build.sh bitnet-$v): skipped"; continue
  fi
  m_greedy "bitnet-$v" "$b" "$BN" 12 "$FR"
  if garbage "bitnet-$v"; then
    # on aarch64 the official build prints '????'; its perplexity is nan
    # and takes ~10 min per window, so it is recorded as garbage instead
    log "bitnet-$v: garbage output, perplexity skipped"
  else
    m_ppl "bitnet-$v" "$b" "$BN" 8
  fi
done
# ik_llama.cpp needs the I2_S file converted first (ik_llama.cpp#387)
IKBN=$WORK/models/bitnet-iq2_bn_r4-ik.gguf
[ -s "$IKBN" ] || "$IK/llama-quantize" --allow-requantize "$BN" "$IKBN" \
  iq2_bn_r4 > "$WORK/logs/quantize-bitnet-ik.log" 2>&1 \
  || die "ik conversion of the BitNet file"
m_greedy ik-bitnet "$IK" "$IKBN" 12 "$FR"
m_ppl ik-bitnet "$IK" "$IKBN" 8

# ---------------------------------------------------------- 27B (full only)
if [ "$MODE" = full ]; then
  fetch_model t27b-ptq1_0; fetch_model t27b-pq2_0
  PT=$(model_path t27b-ptq1_0); P2=$(model_path t27b-pq2_0)
  for f in ptq1_0:$PT pq2_0:$P2; do
    m_greedy "prism-27b-${f%%:*}" "$PR" "${f#*:}" 32 "$FR"
    m_ppl "prism-27b-${f%%:*}" "$PR" "${f#*:}" 2
  done
  m_ppl ik-rtr-27b-ptq1_0 "$IK" "$PT" 2 -rtr
  m_kl_base base-27b-pq "$PR" "$P2" "$WORK/kld/27b-pq.kld" 2
  m_kl kl-ik-27b-pq2_0 "$IK" "$P2" "$WORK/kld/27b-pq.kld"
  m_kl kl-ik-rtr-27b-pq2_0 "$IK" "$P2" "$WORK/kld/27b-pq.kld" -rtr
  m_bench bench-prism-27b-pq2_0 "$PR" "$P2" 128 32 2
  m_bench bench-ik-27b-pq2_0 "$IK" "$P2" 128 32 2 -rtr 0
  m_bench bench-ik-rtr-27b-pq2_0 "$IK" "$P2" 128 32 2 -rtr 1
  m_bench bench-ik-rtr-27b-ptq1_0 "$IK" "$PT" 128 32 2 -rtr 1
fi

log "done: $(wc -l < "$OUT") records in $OUT"
