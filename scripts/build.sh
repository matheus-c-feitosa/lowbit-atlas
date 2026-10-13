#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the runtimes at the pinned commits.
#
#   scripts/build.sh llama prism ik bitnet      # or: scripts/build.sh all
#
# Uses the system C/C++ compiler unless CC/CXX are set. The measurements
# in results/ used clang 18 on aarch64 and gcc 13.3 on x86_64.
# Binaries end up in $WORK/src/<runtime>/build*/bin.

. "$(dirname "$0")/common.sh"

ARCH=$(uname -m)
CMAKE_CC=()
[ -n "${CC:-}" ] && CMAKE_CC+=(-DCMAKE_C_COMPILER="$CC")
[ -n "${CXX:-}" ] && CMAKE_CC+=(-DCMAKE_CXX_COMPILER="$CXX")

build_ggml() { # name url rev: llama.cpp and the Prism fork
  local src="$WORK/src/$1"
  fetch_git "$src" "$2" "$3"
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release "${CMAKE_CC[@]}" \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_OPENSSL=OFF \
    > "$src/configure.log" 2>&1 || die "$1: configure (see $src/configure.log)"
  cmake --build "$src/build" -j "$JOBS" --target llama-completion \
    llama-perplexity llama-bench llama-tokenize llama-quantize \
    test-backend-ops > "$src/build.log" 2>&1 \
    || die "$1: build (see $src/build.log)"
  log "built $1"
}

build_ik() {
  local src="$WORK/src/ik_llama.cpp" flags=()
  fetch_git "$src" "$IK_URL" "$IK_REV"
  if [ "$ARCH" = aarch64 ]; then
    # With clang, -march=native does not enable dotprod on aarch64, and
    # without it ik_llama.cpp drops its optimized kernels
    # (ikawrakow/ik_llama.cpp#361). -mcpu=native does; GGML_ARCH_FLAGS is
    # the way the maintainer recommends passing it (ik_llama.cpp#2651).
    flags=(-DGGML_NATIVE=OFF -DGGML_ARCH_FLAGS=-mcpu=native)
  else
    flags=(-DGGML_NATIVE=ON)
  fi
  # GGML_IQK_FA_ALL_QUANTS=OFF only shortens the build: it affects flash
  # attention with a quantized KV cache, which is not used here.
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release "${CMAKE_CC[@]}" \
    "${flags[@]}" -DGGML_IQK_FA_ALL_QUANTS=OFF -DLLAMA_BUILD_SERVER=OFF \
    > "$src/configure.log" 2>&1 || die "ik: configure (see $src/configure.log)"
  cmake --build "$src/build" -j "$JOBS" --target llama-cli \
    llama-perplexity llama-bench llama-quantize > "$src/build.log" 2>&1 \
    || die "ik: build (see $src/build.log)"
  log "built ik_llama.cpp"
}

# bitnet.cpp variants (all built like setup_env.py does for i2_s):
#   official  main as published (0b341e5, submodule 390c307)
#   fixed     main + isHuangXin/llama.cpp#2 + the one-line activation fix
#             from microsoft/BitNet#588 (SiLU -> ReLU²)
#   605       main + isHuangXin/llama.cpp#2 + microsoft/BitNet#605, which
#             reads the activation from <arch>.hidden_activation (files
#             without the key keep SiLU); run with
#             --override-kv bitnet-b1.58.hidden_activation=str:relu2
#   2025-06   404980e, the last commit before the I2_S kernel rewrite
build_bitnet() { # variant
  local v=$1 src="$WORK/src/bitnet-$1" rev sub_url sub_rev gen flags
  case $v in
    official|fixed|605) rev=$BITNET_REV ;;
    2025-06) rev=$BITNET_OLD_REV ;;
    *) die "unknown bitnet variant: $v" ;;
  esac
  if [ "$(git -C "$src" rev-parse HEAD 2>/dev/null)" != "$rev" ]; then
    rm -rf "$src"
    git clone -q "$BITNET_URL" "$src" || die "clone bitnet.cpp"
    git -C "$src" checkout -q "$rev" || die "checkout $rev"
  fi
  sub_url=$(git -C "$src" config -f .gitmodules submodule.3rdparty/llama.cpp.url)
  sub_rev=$(git -C "$src" ls-tree HEAD 3rdparty/llama.cpp | awk '{print $3}')
  if [ "$v" = fixed ] || [ "$v" = 605 ]; then
    sub_url=$BITNET_PR2_URL; sub_rev=$BITNET_PR2_REV
  fi
  fetch_git "$src/3rdparty/llama.cpp" "$sub_url" "$sub_rev"
  if [ "$v" = 605 ]; then
    local d="$src/3rdparty/llama.cpp"
    if ! grep -q LLM_KV_HIDDEN_ACT "$d/src/models/bitnet.cpp"; then
      git -C "$d" fetch -q --depth 2 "$BITNET_PR2_URL" "$BITNET_605_REV" \
        || die "bitnet 605: fetch $BITNET_605_REV"
      git -C "$d" -c user.name=build -c user.email=build@localhost \
        cherry-pick -n "$BITNET_605_REV" > "$src/cherry-pick.log" 2>&1 \
        || die "bitnet 605: cherry-pick (see $src/cherry-pick.log)"
    fi
  fi
  if [ "$v" = fixed ]; then
    # microsoft/BitNet#588: BitNet b1.58 uses ReLU², not SiLU, in the FFN.
    sed -i 's/LLM_FFN_SILU, LLM_FFN_PAR, il);/LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);/' \
      "$src/3rdparty/llama.cpp/src/models/bitnet.cpp"
    grep -q LLM_FFN_RELU_SQR "$src/3rdparty/llama.cpp/src/models/bitnet.cpp" \
      || die "bitnet fixed: activation patch did not apply"
  fi
  if [ "$ARCH" = aarch64 ]; then
    gen=(utils/codegen_tl1.py --model bitnet_b1_58-3B --BM 160,320,320
         --BK 64,128,64 --bm 32,64,32)
    flags=(-DBITNET_ARM_TL1=OFF)
  else
    gen=(utils/codegen_tl2.py --model bitnet_b1_58-3B --BM 160,320,320
         --BK 96,96,96 --bm 32,32,32)
    flags=(-DBITNET_X86_TL2=OFF)
  fi
  (cd "$src" && python3 "${gen[@]}") > "$src/codegen.log" 2>&1 \
    || die "bitnet $v: codegen (see $src/codegen.log)"
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release "${CMAKE_CC[@]}" \
    "${flags[@]}" -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF \
    > "$src/configure.log" 2>&1 || die "bitnet $v: configure"
  cmake --build "$src/build" -j "$JOBS" --target llama-perplexity \
    llama-tokenize > "$src/build.log" 2>&1 || die "bitnet $v: build"
  # newer trees have llama-completion; the 2025 tree only has llama-cli
  cmake --build "$src/build" -j "$JOBS" --target llama-completion \
    >> "$src/build.log" 2>&1 \
    || cmake --build "$src/build" -j "$JOBS" --target llama-cli \
    >> "$src/build.log" 2>&1 || die "bitnet $v: build"
  log "built bitnet.cpp ($v)"
}

[ $# -gt 0 ] || die "usage: $0 all | llama prism ik bitnet[-official|-fixed|-605|-2025-06]"
[ "$1" = all ] && set -- llama prism ik bitnet
for what in "$@"; do
  case $what in
    llama) build_ggml llama.cpp "$LLAMA_URL" "$LLAMA_REV" ;;
    prism) build_ggml prism "$PRISM_URL" "$PRISM_REV" ;;
    ik) build_ik ;;
    bitnet) for v in official fixed 605 2025-06; do build_bitnet $v; done ;;
    bitnet-*) build_bitnet "${what#bitnet-}" ;;
    *) die "unknown target: $what" ;;
  esac
done
