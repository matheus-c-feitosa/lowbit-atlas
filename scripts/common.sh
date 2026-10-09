# shellcheck shell=bash
# Pinned inputs and small helpers shared by the other scripts.
# Source it: . "$(dirname "$0")/common.sh"

set -u

# Where everything is downloaded and built (override with WORK=...).
WORK=${WORK:-$PWD/work}
JOBS=${JOBS:-$(nproc)}
THREADS=${THREADS:-2}

# --- runtimes, pinned commits ----------------------------------------------
LLAMA_URL=https://github.com/ggml-org/llama.cpp
LLAMA_REV=c479922ac520a08969b4c1dc154d7bbb3c386d85
PRISM_URL=https://github.com/PrismML-Eng/llama.cpp
PRISM_REV=6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17
IK_URL=https://github.com/ikawrakow/ik_llama.cpp
IK_REV=04b4ebc2b32240c7495c33cb23ee35ac8108bb2a
BITNET_URL=https://github.com/microsoft/BitNet
BITNET_REV=0b341e582afbf9e1011f24744b554c96a3477eb5
BITNET_OLD_REV=404980eecae38affa4871c3e419eae3f44536a95
# isHuangXin/llama.cpp#2 (aarch64 I2_S layout fix), head at test time
BITNET_PR2_URL=https://github.com/isHuangXin/llama.cpp
BITNET_PR2_REV=8000be462617183252792344b869297945f49aab

# --- data -------------------------------------------------------------------
WIKI_URL=https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
WIKI_SHA=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08

# --- model files: name|repo|revision|file|sha256 ----------------------------
HF=https://huggingface.co
MODELS="
bitnet-i2_s|microsoft/bitnet-b1.58-2B-4T-gguf|a1f2f1c765812aa8af3f6eda4a313707064bba15|ggml-model-i2_s.gguf|4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162
bonsai-q1_0|prism-ml/Bonsai-1.7B-gguf|210a9e99f79cb184909d49595906526eb2b3dd9a|Bonsai-1.7B-Q1_0.gguf|3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3
ternary-q2_0_g64|prism-ml/Ternary-Bonsai-1.7B-gguf|983b5dec2ff16aab79990711ba0f828a499a7e6a|Ternary-Bonsai-1.7B-Q2_0_g64.gguf|6d0ecb3d9055969b5cde332b6fdb60e67ed3599e9f73e56b977731e1467e5c91
ternary-pq2_0|prism-ml/Ternary-Bonsai-1.7B-gguf|983b5dec2ff16aab79990711ba0f828a499a7e6a|Ternary-Bonsai-1.7B-PQ2_0.gguf|de68ba48a8dacb21979915991e7741b917869d71410a370df951c0c3a237ae50
ternary-q2_0_legacy|prism-ml/Ternary-Bonsai-1.7B-gguf|983b5dec2ff16aab79990711ba0f828a499a7e6a|Ternary-Bonsai-1.7B-Q2_0.gguf|d97d94eb564590c9f0300e54d3f87bbbb25a78693d0ade9f6e177973dcb8228a
t27b-ptq1_0|prism-ml/Ternary-Bonsai-2-27B-gguf|b072e1d3b35a0a630cece372c2127528e0994386|Ternary-Bonsai-2-27B-PTQ1_0.gguf|53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3
t27b-pq2_0|prism-ml/Ternary-Bonsai-2-27B-gguf|b072e1d3b35a0a630cece372c2127528e0994386|Ternary-Bonsai-2-27B-PQ2_0.gguf|3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1
"

# --- helpers ----------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

sha_ok() { # file expected_sha256
  [ -s "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

model_field() { # name field_number -> prints the field (2 repo .. 5 sha)
  printf '%s\n' "$MODELS" | awk -F'|' -v n="$1" -v f="$2" \
    '$1 == n { print $f; found = 1 } END { exit !found }'
}

model_path() { # name -> local path of the model file
  echo "$WORK/models/$(model_field "$1" 4)"
}

fetch_model() { # name: download once, then verify sha256
  local repo rev file sha out
  repo=$(model_field "$1" 2) || die "unknown model: $1"
  rev=$(model_field "$1" 3); file=$(model_field "$1" 4)
  sha=$(model_field "$1" 5); out="$WORK/models/$file"
  mkdir -p "$WORK/models"
  if ! sha_ok "$out" "$sha"; then
    log "downloading $repo@${rev:0:7}/$file"
    curl -fL --retry 3 --progress-bar -o "$out.part" "$HF/$repo/resolve/$rev/$file" \
      || die "download failed: $file"
    mv "$out.part" "$out"
  fi
  sha_ok "$out" "$sha" || die "sha256 mismatch: $out"
  log "ok $file"
}

fetch_wikitext() { # -> $WORK/data/wiki.test.raw, sha256 checked
  local d="$WORK/data"
  mkdir -p "$d"
  if ! sha_ok "$d/wiki.test.raw" "$WIKI_SHA"; then
    curl -fsSL --retry 3 -o "$d/w.zip" "$WIKI_URL" || die "wikitext download"
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extract(
'wikitext-2-raw/wiki.test.raw', sys.argv[2])" "$d/w.zip" "$d"
    mv "$d/wikitext-2-raw/wiki.test.raw" "$d/wiki.test.raw"
  fi
  sha_ok "$d/wiki.test.raw" "$WIKI_SHA" || die "wikitext sha256 mismatch"
  log "ok wiki.test.raw"
}

fetch_git() { # dir url rev: shallow checkout of one exact commit
  if [ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" != "$3" ]; then
    rm -rf "$1"; mkdir -p "$1"
    git -C "$1" init -q
    git -C "$1" fetch -q --depth 1 "$2" "$3" || die "git fetch $2 $3"
    git -C "$1" checkout -q FETCH_HEAD
  fi
  log "ok $(basename "$1") @ ${3:0:7}"
}

cpu_info() { # one line describing the CPU, for the results
  local m
  m=$(lscpu | sed -n 's/^Model name: *//p' | head -1)
  echo "$(uname -m) | $m | $(nproc) CPUs"
}
