# shellcheck shell=bash
# Measurement functions. Source after common.sh:
#   . scripts/common.sh; . scripts/measure.sh
#
# Each function runs one tool, keeps its log in $WORK/logs/<label>.*,
# and appends one JSON line to $OUT (default $WORK/results.jsonl) with
# the label, CPU, binary, runtime commit, model file, flags and the
# parsed numbers. <bin> is a build's bin/ directory.

OUT=${OUT:-$WORK/results.jsonl}
PARSE="$(dirname "${BASH_SOURCE[0]}")/parse_log.py"
WIKI="$WORK/data/wiki.test.raw"

_git_head() { # dir -> commit, only if dir is the top of its own git repo
  local top
  top=$(git -C "$1" rev-parse --show-toplevel 2>/dev/null) || return 0
  [ "$top" = "$(cd "$1" && pwd -P)" ] && git -C "$1" rev-parse HEAD
}

_record() { # label kind bin model parsed_json flags...
  local label=$1 kind=$2 bin=$3 model=$4 parsed=$5; shift 5
  local rev sub
  rev=$(_git_head "$bin/../.."); rev=${rev:-unknown}
  # bitnet.cpp: the submodule commit matters as much as the main one
  sub=$(_git_head "$bin/../../3rdparty/llama.cpp" 2>/dev/null)
  mkdir -p "$(dirname "$OUT")"
  python3 - "$label" "$kind" "$bin" "$rev" "$sub" "$model" "$parsed" \
    "$(cpu_info)" "$*" "$WORK" >> "$OUT" <<'EOF'
import json, os, sys
label, kind, binary, rev, sub, model, parsed, cpu, flags, work = sys.argv[1:11]
# binary path relative to WORK, so records do not carry local paths
rec = {"label": label, "kind": kind, "cpu": cpu, "runtime_commit": rev,
       "binary": os.path.relpath(os.path.abspath(binary), os.path.abspath(work)),
       "model": os.path.basename(model),
       "flags": flags}
if sub:
    rec["submodule_commit"] = sub
rec.update(json.loads(parsed))
print(json.dumps(rec, ensure_ascii=False))
EOF
}

m_ppl() { # label bin model chunks [flags...]
  local label=$1 bin=$2 model=$3 chunks=$4; shift 4
  local log="$WORK/logs/$label.ppl.log"
  mkdir -p "$WORK/logs"
  "$bin/llama-perplexity" -m "$model" -f "$WIKI" -c 512 --chunks "$chunks" \
    -t "$THREADS" "$@" < /dev/null > "$log" 2>&1 || log "$label: rc=$?"
  _record "$label" ppl "$bin" "$model" "$(python3 "$PARSE" ppl "$log")" \
    "-c 512 --chunks $chunks -t $THREADS $*"
  log "$label: $(python3 "$PARSE" ppl "$log" | cut -c1-60)"
}

m_kl_base() { # label bin model base_file chunks [flags...]
  local label=$1 bin=$2 model=$3 base=$4 chunks=$5; shift 5
  mkdir -p "$WORK/logs" "$(dirname "$base")"
  "$bin/llama-perplexity" -m "$model" -f "$WIKI" -c 512 --chunks "$chunks" \
    -t "$THREADS" --kl-divergence-base "$base" "$@" < /dev/null \
    > "$WORK/logs/$label.klbase.log" 2>&1 || die "$label: kl base failed"
  log "$label: base written ($base)"
}

m_kl() { # label bin model base_file [flags...]
  local label=$1 bin=$2 model=$3 base=$4; shift 4
  local log="$WORK/logs/$label.kl.log"
  "$bin/llama-perplexity" -m "$model" -c 512 -t "$THREADS" \
    --kl-divergence-base "$base" --kl-divergence "$@" < /dev/null \
    > "$log" 2>&1 || log "$label: rc=$?"
  _record "$label" kl "$bin" "$model" "$(python3 "$PARSE" kl "$log")" \
    "-c 512 -t $THREADS $*"
  log "$label: $(python3 "$PARSE" kl "$log" | cut -c1-70)"
}

m_greedy() { # label bin model n_tokens prompt [flags...]
  local label=$1 bin=$2 model=$3 n=$4 prompt=$5; shift 5
  local out err="$WORK/logs/$label.gen.log" tool=() parsed
  mkdir -p "$WORK/logs"
  if [ -x "$bin/llama-completion" ]; then
    tool=("$bin/llama-completion" -no-cnv)
  else
    tool=("$bin/llama-cli")
    "$bin/llama-cli" --help < /dev/null 2>&1 | grep -q -- '-no-cnv' \
      && tool+=(-no-cnv)
  fi
  out=$("${tool[@]}" -m "$model" -p "$prompt" -n "$n" --temp 0 \
    -t "$THREADS" -c 512 "$@" < /dev/null 2> "$err")
  parsed=$(python3 -c 'import json,sys; print(json.dumps(
    {"prompt": sys.argv[1], "n_tokens": int(sys.argv[2]), "text": sys.argv[3]}))' \
    "$prompt" "$n" "$out")
  _record "$label" greedy "$bin" "$model" "$parsed" "--temp 0 -n $n $*"
  log "$label: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-70)"
}

m_bench() { # label bin model pp tg reps [flags...]
  local label=$1 bin=$2 model=$3 pp=$4 tg=$5 reps=$6; shift 6
  local log="$WORK/logs/$label.bench.md"
  mkdir -p "$WORK/logs"
  "$bin/llama-bench" -m "$model" -p "$pp" -n "$tg" -t "$THREADS" \
    -r "$reps" "$@" -o md < /dev/null > "$log" 2>&1 || log "$label: rc=$?"
  _record "$label" bench "$bin" "$model" "$(python3 "$PARSE" bench "$log")" \
    "-p $pp -n $tg -t $THREADS -r $reps $*"
  log "$label: $(python3 "$PARSE" bench "$log")"
}
