# Method

How the numbers in `results/` were taken, which references they are compared against, and the pitfalls found along the way. Every number has a pinned input: runtime commit, model file revision and sha256, CPU class, compiler and flags.

## 1. What is measured

Two questions, for each model file × runtime × CPU class:

1. **Is the output correct?** Compared against a reference, not just "looks coherent".
2. **How fast is it?** Prompt processing (pp) and generation (tg), with `llama-bench`.

| Axis | Covered |
|---|---|
| Models | BitNet-b1.58-2B-4T (I2_S); Bonsai-1.7B (Q1_0); Ternary-Bonsai-1.7B (Q2_0 g64, PQ2_0); Ternary-Bonsai-2-27B (PTQ1_0, PQ2_0) |
| Runtimes | llama.cpp `c479922`; Prism fork `6bfcd79`; ik_llama.cpp `04b4ebc`; bitnet.cpp `0b341e5` and older commits, with and without community fixes |
| CPU classes | aarch64 NEON + dotprod (Neoverse-N1); aarch64 ARMv8.0 (built with `-march=armv8-a`, no dotprod); x86-64 AVX2 (Intel Xeon @ 2.2 GHz, AMD EPYC 7B12); x86-64 AVX and SSE-only (built with explicit flags on an AVX2 machine) |
| Compilers | clang 18.1.3 on aarch64, gcc 13.3 on x86-64 |

**Not covered:** GPUs, AVX-512, Arm with i8mm or SVE, real (not emulated) older CPUs, long context, other languages than English (one Portuguese prompt aside).

## 2. Protocol

- **Text:** WikiText-2 `wiki.test.raw` (sha256 `173c87a5…dd08`).
- **Perplexity:** `llama-perplexity -c 512`, 8 windows (2 for the 27B model, which peaked at 7.7 GB of RAM and takes hours per run on these machines), `-t 2`. Each window starts with BOS, and the score is the NLL of the second half of the window.
- **Greedy text:** 8–32 tokens at `--temp 0` from fixed prompts, mainly `The capital of France is`.
- **Speed:** `llama-bench -p 512 -n 64 -r 3` on the 1.7B models; shorter runs (`-p 128 -n 32 -r 2`) on the 27B model.

## 3. References

A runtime can only be called wrong relative to something. Two kinds of reference were used.

**Exact reference from the same file.** For Q1_0, and for PQ2_0 files without a Hadamard rotation (the Ternary-Bonsai-1.7B), the weights are exactly representable in F16: `llama-quantize --allow-requantize <file> F16` gives a model with the same weights and no activation quantization. It is exact, but it comes from the same code base, so it is not independent.

**Independent reference: HF transformers on the same token IDs** (BitNet-b1.58-2B-4T):

- Model `microsoft/bitnet-b1.58-2B-4T` @ `04c3b9a`, `BitNetForCausalLM` (transformers 5.16.1), float32.
- The official GGUF has no `tokenizer.ggml.pre`. GGUF runtimes therefore tokenize WikiText into about 16% more tokens than the HF tokenizer, and perplexities on the two tokenizations are not comparable (already noted in microsoft/BitNet#588 and PR #508).
- So the reference scores **the exact token IDs** that `llama-tokenize` produces: 325,587 tokens, identical across bitnet.cpp and ik_llama.cpp. The first 4,096 IDs, used for 8 windows, have sha256 `a04e9858…a0dd` (uint32 little-endian).
- The `llama-perplexity` protocol is reproduced exactly.

## 4. What counts as a divergence

Two valid implementations of the same file do not give identical numbers. Activations are quantized differently, sums happen in a different order, and different repack layouts are used. Measured against the exact reference:

| File | Runtime / path | KLD vs exact F16 | Same top-1 |
|---|---|---|---|
| Q1_0 | llama.cpp repack | 0.000224 | 99.31% |
| Q1_0 | llama.cpp vec_dot | 0.000238 | 99.17% |
| Q1_0 | ik_llama.cpp | 0.000248 | 99.46% |
| PQ2_0 | Prism | 0.000986 | 98.48% |
| PQ2_0 | ik_llama.cpp | 0.000946 | 98.73% |

The natural spread **depends on the type**: ~2–3e-4 for Q1_0 and ~1e-3 for PQ2_0. A fixed threshold calibrated on one type gives false positives on the other. ik_llama.cpp and Prism are 1.2e-3 apart on PQ2_0 only because each is ~1e-3 from the reference in a different direction.

**Criterion used:** a runtime diverges when it is **clearly farther from the reference than the other runtimes** (for example, more than 2× the best). Garbage output, a crash, or the refusal of a valid file also count. Runtime-vs-runtime comparisons without a reference are used only for triage.

## 5. Pitfalls found along the way

**5.1 Perplexity alone cannot tell a correct path from a wrong one at the 0.1% level.** The same Q1_0 file gives 23.1535–23.1794 depending on which code path the runtime picks for the CPU (vec_dot vs repack), with no error on either side. Both paths are at the same distance from the exact reference, and the PPL difference is noise from where each path rounds. Differences up to ~0.15% across CPUs and runtimes showed no error here.

**5.2 The `ln(PPL(Q)/PPL(base))` ratio of `llama-perplexity` is biased by ~0.5%.** In a control (same binary, same file, compared against its own base), the ratio is 1.0049, not 1.0, and some per-token KLD values are negative. The base logits are stored as uint16 after clamping at 16 nats below the maximum. That is a deliberate space-saving design, from the PR that added the feature (ggml-org/llama.cpp#5076). When the correct token falls below the clamp, the base NLL is capped, which deflates PPL(base). Use the full-precision PPL(Q) of each run for parity, and KLD/top-1 for distance; mind a noise floor of |KLD per token| ≲ 6e-5.

**5.3 Same tokens, or the comparison is meaningless.** See §3: a missing `tokenizer.ggml.pre` changes the token count by ~16%.

**5.4 Check which source file a fix actually touches.** In bitnet.cpp, `src/ggml-bitnet-mad.cpp` is no longer compiled on current main: since the submodule moved to `390c307` (July 2026), the I2_S kernels that run are in the submodule's `ggml/src/ggml-cpu/quants.c`. That the file had dropped out of the build was noted in microsoft/BitNet#626; the cause is the submodule switch. Patches to `src/ggml-bitnet-mad.cpp` therefore do not change the output of current main. Counting object files in the build directory is a quick check.

**5.5 Build flags change which kernels run.**

- With clang, `-march=native` does not enable dotprod on aarch64, and ik_llama.cpp then silently drops its optimized kernels (ikawrakow/ik_llama.cpp#361); gcc does enable it. `-mcpu=native` works with both.
- In ik_llama.cpp, run-time repacking (`-rtr`) changes speed by up to ~10× and is not the default.

**5.6 File size does not predict speed.** What matters is whether the runtime has a repacked or GEMM path for that type on that CPU. Without one, prompt processing runs close to generation speed (pp/tg ≈ 1.3). The smaller Ternary-Bonsai-2-27B file (PTQ1_0) is about 6× slower than the larger one (PQ2_0) for prompt processing on AVX2 x86 (PrismML-Eng/llama.cpp#324).

**5.7 Greedy text is not portable across CPUs.** With the same file, the same binary version and greedy decoding, a near-tie can flip one token and change the rest of the answer. In about 1% of positions the top-1 token differs between valid paths. Exact-text comparisons across machines (CI, "reproducible" evaluations) need a tolerance or a distance metric.

## 6. Machines and noise

- An aarch64 cloud VM (Neoverse-N1, 2 cores, 11 GiB).
- Hosted x86-64 notebook VMs (4 vCPUs). The CPU model changes between sessions, and each run records it.
- Other processes were running on both. Speed numbers come from `llama-bench` with repetitions and standard deviation; treat differences under ~5% as noise.

## 7. Reproducing

`scripts/` rebuilds the runtimes at the pinned commits (`build.sh`), downloads and verifies every file (`common.sh`), and runs each measurement while recording CPU, commits and flags (`measure.sh`). End-to-end reproduction scripts are being finalized.

## 8. Limits

- One file per format, short context, English text.
- Two physical machine types, with older CPU classes emulated by build flags.
- KL measured on aarch64 only (x86: PPL and greedy text).
- The 27B runs use 2 windows.
- "Not found" is not "does not exist": check the linked issues for later changes.
