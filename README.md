# Low-bit LLMs on CPU: what runs correctly and fast where (Oct 2026)

By [Matheus Feitosa](https://github.com/matheus-c-feitosa) · October 2026

> **Status: draft, not published yet.**

An independent check of low-bit (1-bit, ternary, 2-bit) LLM files on CPU. For each file, runtime and CPU class, it answers two questions: **does the output match a reference, and how fast is it?** Every number comes with the runtime commit, file sha256, CPU and flags (`results/`), and `scripts/` rebuilds and reruns it.

## Key findings

1. **BitNet-b1.58-2B-4T: the official runtime (bitnet.cpp) gives wrong output on CPU, silently, and has done so for months.**
   - On x86-64 the text looks fine, but perplexity is ~6× the reference: the FFN uses SiLU where the model uses ReLU² ([microsoft/BitNet#588](https://github.com/microsoft/BitNet/issues/588), reported July 2026).
   - On aarch64 the output is garbage (`????`) since commit `112f853`, Nov 2025 ([#470](https://github.com/microsoft/BitNet/issues/470)).
   - Fixes have been proposed but **none is merged**. We tested them against a reference: on current main, **only the one-line activation fix from #588 (plus isHuangXin/llama.cpp#2 on aarch64) gives correct output**; two other proposed fixes do not reach the code that runs. See [BitNet](#bitnet-b158-2b-4t-which-fix-actually-works).
2. **The Bonsai family (Prism) runs correctly** in llama.cpp and the Prism fork on every CPU class tested, and in ik_llama.cpp where tested. The differences are in **speed** and in **which file loads where**: the file marked "recommended" on one model card does not load in llama.cpp or the Prism fork.
3. **Speed depends on the format × CPU × runtime path, not on file size.** The smallest 27B file is ~6× slower to process prompts on AVX2 x86. ik_llama.cpp with `-rtr` is the fastest in most cases measured (not for PQ2_0 generation on aarch64), and `-rtr` is off by default: without it, ik_llama.cpp is often the slowest.

## BitNet-b1.58-2B-4T: which fix actually works

Same official GGUF (`ggml-model-i2_s.gguf`, sha256 `4221b252…`) and the same 4,096 token IDs everywhere. WikiText-2, 8 × 512 windows. The reference is HF transformers (`BitNetForCausalLM`, float32) scoring the same token IDs. Greedy text continues "The capital of France is".

| Runtime / fix | CPU | Greedy text | PPL | vs reference |
|---|---|---|---|---|
| **HF transformers (reference)** | — | "Paris. Paris is a city in the Île-de-F…" | **18.79** | — |
| bitnet.cpp main (`0b341e5`) | x86-64 AVX2 | "a small city, and the capital of France is a small" | 113.33 | **6.0× — wrong** (same with gcc 13.3 and clang 18) |
| bitnet.cpp main | aarch64 | `????????????` | — | **garbage** |
| main + [isHuangXin/llama.cpp#2](https://github.com/isHuangXin/llama.cpp/pull/2) | aarch64 | "a small city, and…" | 113.26 | 6.0× — layout fixed, activation still wrong |
| **main + llama.cpp#2 + the one-line fix from [#588](https://github.com/microsoft/BitNet/issues/588)** | aarch64 NEON+dotprod | "Paris. Paris is a city that is known for its rich" | **18.84** | **+0.27% — correct** |
| **main + the one-line fix from #588** | x86-64 AVX2 | "Paris. Paris is a city that is known for its rich" | **18.87** | **+0.39% — correct** |
| main + llama.cpp#2 + the #588 line (same recipe as aarch64) | x86-64 AVX2 | same | 18.87 | identical to the line above: one recipe for both |
| main + [#633](https://github.com/microsoft/BitNet/pull/633) | aarch64 | `????????????` | — | no effect: edits a file current main does not compile |
| [#586](https://github.com/microsoft/BitNet/pull/586) on its own base (`01eb415`, Mar 2026) | aarch64 | "Paris. Paris is a city…" | 18.85 | correct, but does not apply to current main (same reason) |
| bitnet.cpp `404980e` (June 2025) | aarch64 / x86-64 | "Paris. Paris is a city…" | 18.85 / 18.81 | correct (before the regression) |
| **ik_llama.cpp `04b4ebc`** (file converted to `iq2_bn_r4`, as [ik#387](https://github.com/ikawrakow/ik_llama.cpp/issues/387) documents) | aarch64 / x86-64 | "Paris. Paris is a city…" | **18.81 / 18.85** | **correct** |

All correct implementations land within +0.1–0.4% of the reference. A GGUF runtime quantizes activations to 8 bits and the reference runs in float32, so a small positive offset is expected. The wrong one is at +503%.

**Cross-check with the numbers published in #588.** We reran the protocol posted there by isaacmadine96-art (20 × 512 windows, `-t 4`) on our own x86-64 machine:

| Build | Their result | Ours |
|---|---|---|
| June 2025 (`765741d`; same build code as `404980e`) | 19.6002 ± 0.85043 | 19.6002 ± 0.85043 |
| current main (`16da220` = `0b341e5`) | 118.7272 ± 6.21259 | 118.7272 ± 6.21259 |
| June 2025, `tokenizer.ggml.pre=llama-bpe` | 16.7775 | 16.7775 |
| current main + #588 fix, `llama-bpe` | — | 16.7876 |

The results are identical to the fourth decimal on a different machine, so the two measurements confirm each other.

**What to use today:**

- **Easiest:** ik_llama.cpp. Convert the file once with `llama-quantize --allow-requantize ggml-model-i2_s.gguf out.gguf iq2_bn_r4`.
- **To stay on bitnet.cpp:** apply the one-line change from #588 (`LLM_FFN_SILU` → `LLM_FFN_RELU_SQR` in `3rdparty/llama.cpp/src/models/bitnet.cpp`). On aarch64, also apply isHuangXin/llama.cpp#2. `scripts/build.sh bitnet-fixed` builds exactly this.

**Credit:**

- The ~6× perplexity and its cause were reported in microsoft/BitNet#588 by hmeiland, and confirmed there by several people.
- The aarch64 regression was bisected to `112f853` in microsoft/BitNet#470 by aqn96.
- That `ggml-bitnet-mad.cpp` had dropped out of the build was noted in #626.

What this report adds is the comparison of the fixes against a reference on identical token IDs, on both aarch64 and x86-64, plus an exact reproduction of the numbers published in #588. Details: [`results/bitnet-b1.58-2b-4t.json`](results/bitnet-b1.58-2b-4t.json).

## Bonsai 1.7B (Prism): correct everywhere tested

Three formats (PQ2_0 loads only in Prism) × five CPU classes × llama.cpp and the Prism fork, plus ik_llama.cpp on aarch64 NEON+dotprod. The CPU classes are aarch64 NEON+dotprod, aarch64 ARMv8.0, x86-64 AVX2, x86-64 AVX and x86-64 SSE.

- Kernel tests: `test-backend-ops` 0 failures (llama.cpp 310/310, Prism 833/833) in every class.
- Perplexity spread across classes and runtimes is ≤ 0.15%, within the spread between valid code paths (see [METHOD](METHOD.md#5-pitfalls-found-along-the-way)). Greedy text is identical everywhere.
- KL against an exact F16 reference (aarch64 NEON+dotprod) puts llama.cpp, Prism and ik_llama.cpp at the type's natural spread (Q1_0 ~2.2–2.5e-4, PQ2_0 ~0.8–1.0e-3). One case sits at the edge: ik_llama.cpp **with `-rtr`** on Q1_0 is 2.0× farther than the best path (4.5e-4, same top-1 98.7%). It quantizes activations differently (`Q8_K128`). Greedy text is unchanged, and it is not counted as a divergence, but it is the closest to the line.

**Which file loads where:**

| File | llama.cpp | Prism | ik_llama.cpp |
|---|---|---|---|
| Bonsai-1.7B Q1_0 | ✔ | ✔ | ✔ |
| Ternary-Bonsai-1.7B Q2_0 g64 | ✔ | ✔ | ✘ (no type 42) |
| Ternary-Bonsai-1.7B PQ2_0 | ✘ (type 142 unknown) | ✔ | ✔ |
| Ternary-Bonsai-1.7B **Q2_0 ("recommended" on the card)** | ✘ | ✘ (clear message: legacy layout) | — |

The legacy Q2_0 file problem was already reported (PrismML-Eng/llama.cpp#167, Ternary-Bonsai-27B-gguf discussions #64/#65). The 1.7B card still recommends it at revision `983b5de`.

**Speed** (aarch64 Neoverse-N1, 2 threads, `llama-bench -p 512 -n 64 -r 3`, tokens/s):

| File | llama.cpp | Prism | ik_llama.cpp | ik_llama.cpp `-rtr` |
|---|---|---|---|---|
| Q1_0 | 51.8 / 14.1 | 51.7 / 14.2 | 34.9 / 14.2 | **72.0 / 23.3** |
| PQ2_0 | — | 15.3 / 11.1 | 6.4 / 1.8 | **68.5** / 8.1 |

## Ternary-Bonsai-2 27B (Prism): correct, speed depends on the file

- PTQ1_0 aarch64 vs x86-64: PPL within 0.03%.
- ik_llama.cpp vs Prism on PQ2_0 (x86-64): KLD 2.1e-4 with `-rtr` (same top-1 99.2%), 2.6e-4 without (98.8%). 2 windows only.
- PTQ1_0 and PQ2_0 hold bit-identical weights (PrismML-Eng/llama.cpp#148). In ik_llama.cpp with `-rtr` they give the same PPL to every printed digit. In Prism the PPL differs slightly (8.7507 vs 8.7515), presumably because each type has its own kernel.
- On AVX2 x86-64, PTQ1_0 processes prompts ~6× slower than PQ2_0 in Prism (no repacked path; PrismML-Eng/llama.cpp#324).
- ik_llama.cpp with `-rtr` is ~1.9× (prompt) and ~2.4× (generation) faster than Prism on PQ2_0. Without `-rtr`, prompt processing is ~3.5× slower.
- ik_llama.cpp labels this PQ2_0 file "IQ5_K". That is cosmetic: `general.file_type` 141 means PQ2_0 in Prism and IQ5_K in ik_llama.cpp.

## How it was measured

See [METHOD.md](METHOD.md): protocol, the two kinds of reference, what counts as a divergence, and seven pitfalls.

## Reproduce

`scripts/` pins every input. `build.sh` builds llama.cpp, Prism, ik_llama.cpp and three bitnet.cpp variants at fixed commits; `common.sh` downloads and checks every file; `measure.sh` records each measurement with CPU, commits and flags. *The end-to-end runner is pending.*

## Feedback

Found an error, or have a CPU, runtime or model file that is not covered here? Please open an issue in this repository. Include the runtime commit, the CPU (`lscpu` on Linux), the file, and the exact command and output. Corrections are especially welcome: if one of the linked fixes gets merged, or a number here does not reproduce, the report will be updated and the change credited.

## Limitations

- One file per format, 512-token context, English text, 8 windows (2 for the 27B model).
- Real hardware: aarch64 Neoverse-N1 and x86-64 AVX2 (Intel Xeon @ 2.2 GHz, AMD EPYC 7B12). The older classes were emulated with build flags.
- No GPU, AVX-512, Arm i8mm/SVE or Apple Silicon.
- KL against an exact reference was measured on aarch64 only. On x86-64: PPL, greedy text, and KL between runtimes for the 27B model. For BitNet, only PPL was compared with the HF reference, not KL.
- Runtimes move fast: the linked issues may have changed since October 2026.

## AI assistance

This work was done with the help of Claude Code (Anthropic), an AI coding assistant. It drafted scripts and text, helped analyze logs, and ran many of the measurements on the author's machines under the author's direction. The author reviewed every number against the logs and is responsible for the content. The AI is not an author. Commits made with its help carry an `Assisted-by:` trailer.

## License

- Scripts in `scripts/`: [Apache-2.0](LICENSE).
- Report text, tables and data in `results/`: [CC BY 4.0](LICENSE-CC-BY-4.0.txt).
- No model weights or third-party runtime code are redistributed here. The scripts download them from their original repositories, under their own licenses.
