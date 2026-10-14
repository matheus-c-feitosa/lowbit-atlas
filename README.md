# Low-bit LLMs on CPU: what runs correctly and fast where (Oct 2026)

By [Matheus Feitosa](https://github.com/matheus-c-feitosa) · October 2026

> **First edition, October 2026.** Measured, and reproduced from scratch with the scripts in this repository, on aarch64 (Neoverse-N1) and x86-64 (Intel Xeon, AVX2).

An independent check of low-bit (1-bit, ternary, 2-bit) LLM files on CPU. For each file, runtime and CPU class, it answers two questions: **does the output match a reference, and how fast is it?** Every number comes with the runtime commit, file sha256, CPU and flags (`results/`), and `scripts/` rebuilds and reruns it.

## Key findings

1. **BitNet-b1.58-2B-4T: the official runtime (bitnet.cpp) gives wrong output on CPU, silently, and has done so for months.**
   - On x86-64 the text looks fine, but perplexity is ~6× the reference: the FFN uses SiLU where the model uses ReLU² ([microsoft/BitNet#588](https://github.com/microsoft/BitNet/issues/588), reported July 2026).
   - On aarch64 the output is garbage (`????`) since commit `112f853`, Nov 2025 ([#470](https://github.com/microsoft/BitNet/issues/470)).
   - Fixes have been proposed but **none is merged**. We tested them against a reference: on current main, **only the activation fix from #588 gives correct output** (plus isHuangXin/llama.cpp#2 on aarch64). It can be applied either as a one-line change or, safely for the other BitNet models, through [#605](https://github.com/microsoft/BitNet/pull/605) plus a runtime key. Two other proposed fixes do not reach the code that runs. See [BitNet](#bitnet-b158-2b-4t-which-fix-actually-works).
2. **The Bonsai family (Prism) runs correctly** in llama.cpp and the Prism fork on every CPU class tested, and in ik_llama.cpp where tested. The differences are in **speed** and in **which file loads where**: the file marked "recommended" on one model card does not load in any of the three runtimes.
3. **Speed depends on the format × CPU × runtime path, not on file size.** The smallest 27B file is ~6× slower to process prompts on AVX2 x86. ik_llama.cpp with `-rtr` is the fastest in most cases measured (not for PQ2_0 generation on aarch64), and `-rtr` is off by default: without it, ik_llama.cpp is often the slowest. On x86-64 at the measured commits, Q2_0 has no SIMD kernel at all, so the Ternary-Bonsai-1.7B Q2_0 g64 file processes prompts ~13× slower than its PQ2_0 version. llama.cpp merged an AVX2 kernel for it on 2026-10-13: measured before and after on the same VM, it is ~6× faster with unchanged accuracy.

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
| main + llama.cpp#2 + [#605](https://github.com/microsoft/BitNet/pull/605), file as published | aarch64 | "a small city, and…" | — | still wrong: files without the activation key keep SiLU |
| **main + llama.cpp#2 + #605, run with `--override-kv bitnet-b1.58.hidden_activation=str:relu2`** | aarch64 / x86-64 | "Paris. Paris is a city that is known for its rich" | **18.84 / 18.87** | **+0.27% / +0.39% — correct**, identical to the one-line fix; files without the key keep SiLU |
| main + [#633](https://github.com/microsoft/BitNet/pull/633) | aarch64 | `????????????` | — | no effect: edits a file current main does not compile |
| [#586](https://github.com/microsoft/BitNet/pull/586) on its own base (`01eb415`, Mar 2026) | aarch64 | "Paris. Paris is a city…" | 18.85 | correct, but does not apply to current main (same reason) |
| bitnet.cpp `404980e` (June 2025) | aarch64 / x86-64 | "Paris. Paris is a city…" | 18.85 / 18.81 (clang), 18.84 (gcc) | correct (before the regression) |
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

- **Easiest:** ik_llama.cpp. Convert the file once with `llama-quantize --allow-requantize ggml-model-i2_s.gguf out.gguf iq2_bn_r4`. Measured correct on aarch64 and x86-64.
- **bitnet.cpp, safe for every BitNet model:** apply #605 (on aarch64, also isHuangXin/llama.cpp#2), and run with `--override-kv bitnet-b1.58.hidden_activation=str:relu2`. Files without that key keep SiLU, so by design the older 1bitLLM models (`bitnet_b1_58-large`, `-3B`), which go through the same graph code and do use SiLU, should not be affected. Those models were not tested here. `scripts/build.sh bitnet-605` builds this. Measured on aarch64 (18.84) and x86-64 (18.87).
- **bitnet.cpp, quickest, but only for BitNet-b1.58-2B-4T:** the one-line change from #588 (`LLM_FFN_SILU` → `LLM_FFN_RELU_SQR` in `3rdparty/llama.cpp/src/models/bitnet.cpp`; on aarch64, plus isHuangXin/llama.cpp#2). It switches every model that goes through this graph code to ReLU², so **it breaks the 1bitLLM models**. That is why the same change was not accepted in llama.cpp, as discussed in #588. `scripts/build.sh bitnet-fixed` builds this. A narrower variant applies ReLU² only when `model.arch == LLM_ARCH_BITNET_B158 && model.type == LLM_TYPE_2B` (homeofe, #602); it was not tested here.
- **Tokenizer:** the published GGUF has no `tokenizer.ggml.pre`, so llama.cpp-based runtimes fall back to a default pre-tokenizer and produce ~16% more tokens than the HF tokenizer. `tokenizer.ggml.pre=str:llama-bpe` makes the tokenization match the model's own tokenizer (checked in #588). Perplexities on different tokenizations are not directly comparable (19.60 vs 16.78 in our rerun of the #588 protocol), so the tables above keep the default tokenization and score every runtime on identical tokens.
- **Pass all keys in one flag, comma-separated:** `--override-kv bitnet-b1.58.hidden_activation=str:relu2,tokenizer.ggml.pre=str:llama-bpe`. Repeating `--override-kv` prints a DEPRECATED warning ("only last value will be used"). In our tests with `llama-completion` and `llama-perplexity`, both values still applied (window-1 PPL 9.48 either way), but #602 reports that only the last one took effect in its setup.
- **For chat:** #602 (homeofe, paulanthonyhebbes-sudo) reports that the GGUF's built-in chat template is outdated (use the one in the model's `tokenizer_config.json`). With SiLU the model never emits `<|eot_id|>` and keeps going until the token limit; with ReLU² it stops normally. Not tested here.
- **Where this was tested:** aarch64 with NEON + dotprod (Neoverse-N1) and x86-64 with AVX2. Other platforms have separate open bitnet.cpp bugs that are not covered here: x86 without AVX2 (#547, #617), Arm with i8mm such as Apple M2 and later or Graviton3 (#618, a build failure), BLAS builds such as the macOS default (#601), and build failures on macOS (#611).

**Credit:**

- The ~6× perplexity and its cause were reported in microsoft/BitNet#588 by hmeiland, and confirmed there by several people.
- The aarch64 regression was bisected to `112f853` in microsoft/BitNet#470 by aqn96.
- That `ggml-bitnet-mad.cpp` had dropped out of the build was noted in #626.
- The metadata-based fix is #605, by praneshnikhar.
- The caveat that the one-line change breaks the 1bitLLM models, and the tokenizer finding, come from the #588 thread (purpleskulll, dondetir, isaacmadine96-art).

What this report adds is the comparison of the fixes against a reference on identical token IDs, on both aarch64 and x86-64, plus an exact reproduction of the numbers published in #588. Details: [`results/bitnet-b1.58-2b-4t.json`](results/bitnet-b1.58-2b-4t.json).

## Bonsai 1.7B (Prism): correct everywhere tested

Three formats (PQ2_0 loads only in Prism) × five CPU classes × llama.cpp and the Prism fork, plus ik_llama.cpp on aarch64 NEON+dotprod. The CPU classes are aarch64 NEON+dotprod, aarch64 ARMv8.0, x86-64 AVX2, x86-64 AVX and x86-64 SSE.

- Kernel tests: `test-backend-ops` 0 failures (llama.cpp 310/310, Prism 833/833) in every class.
- Perplexity spread across classes and runtimes is ≤ 0.15%, within the spread between valid code paths (see [METHOD](METHOD.md#5-pitfalls-found-along-the-way)). Greedy text is identical everywhere.
- KL against an exact F16 reference puts llama.cpp, Prism and ik_llama.cpp at the type's natural spread, with the same pattern on aarch64 and x86-64 (Q1_0 ~2.1–2.5e-4, PQ2_0 ~0.75–1.0e-3). One case sits at the edge: ik_llama.cpp **with `-rtr`** on Q1_0 is ~2× farther than the best path (4.5e-4 on aarch64, same top-1 98.7%; 4.2e-4 on x86-64). It quantizes activations differently (`Q8_K128`). Greedy text is unchanged, and it is not counted as a divergence, but it is the closest to the line.

**Which file loads where:**

| File | llama.cpp | Prism | ik_llama.cpp |
|---|---|---|---|
| Bonsai-1.7B Q1_0 | ✔ | ✔ | ✔ |
| Ternary-Bonsai-1.7B Q2_0 g64 | ✔ | ✔ | ✘ (no type 42) |
| Ternary-Bonsai-1.7B PQ2_0 | ✘ (type 142 unknown) | ✔ | ✔ |
| Ternary-Bonsai-1.7B **Q2_0 ("recommended" on the card)** | ✘ | ✘ (clear message: legacy layout) | ✘ |

The legacy Q2_0 file problem was already reported (PrismML-Eng/llama.cpp#167, Ternary-Bonsai-27B-gguf discussions #64/#65). The 1.7B card still recommends it at revision `983b5de`.

**Speed** (aarch64 Neoverse-N1, 2 threads, `llama-bench -p 512 -n 64 -r 3`, tokens/s):

| File | llama.cpp | Prism | ik_llama.cpp | ik_llama.cpp `-rtr` |
|---|---|---|---|---|
| Q1_0 | 51.8 / 14.1 | 51.7 / 14.2 | 34.9 / 14.2 | **72.0 / 23.3** |
| PQ2_0 | — | 15.3 / 11.1 | 6.4 / 1.8 | **68.5** / 8.1 |

**Speed** (x86-64 Intel Xeon @ 2.2 GHz, AVX2, hosted VM with 4 vCPUs, 2 threads, same `llama-bench` command, tokens/s):

| File | llama.cpp | Prism | ik_llama.cpp | ik_llama.cpp `-rtr` |
|---|---|---|---|---|
| Q1_0 | 22.5 / 16.0 | 38.4 / 18.3 | 31.2 / 17.3 | **75.9 / 23.6** |
| Q2_0 g64 | 2.1 / 1.6 | 2.1 / 1.6 | — | — |
| PQ2_0 | — | 27.8 / 12.9 | 12.0 / 12.9 | **71.4 / 17.3** |

On x86-64 at these commits, llama.cpp has no repacked path for Q1_0, while Prism does. Neither runtime has a SIMD kernel for Q2_0 at these commits: it runs generic C code, ~13× slower than PQ2_0.

**Update, 2026-10-13:** llama.cpp merged an AVX2 Q2_0 kernel (ggml-org/llama.cpp#30447, `0a731cc`). Measured at that commit and at its parent (`f851e24`), in one session on the same kind of VM (Intel Xeon @ 2.2 GHz, AVX2 + FMA, no AVX-512, gcc 13.3, 2 threads):

| Q2_0 g64 in llama.cpp | Speed, tokens/s (prompt / generation) | KL vs exact F16 | Same top-1 | Greedy text |
|---|---|---|---|---|
| before: generic C | 1.77 / 1.41 | 2.32e-4 | 99.1% | "Paris. The capital of France is Paris" |
| after: AVX2 kernel | **11.3 / 8.5** (6.4× / 6.0×) | 2.27e-4 | 99.2% | same |

- **The kernel is correct:** KL is unchanged, the text is the same, and the PPL moved by −0.05% (18.3723 → 18.3632).
- **It is ~6× faster end to end.** Its author reported 8–9× on the matrix multiplication alone, on AMD Zen 2. The end-to-end numbers also include the other operations.
- **Compare speeds within one session only:** this session ran ~15% slower than the table above (1.77 vs 2.1 before the kernel).
- The F16 reference is converted from the Q2_0 g64 file itself. The conversion is exact, because the file holds only Q2_0 and F32 tensors.
- The commits pinned in `scripts/` predate the kernel, so the Q2_0 row in the table above is the generic path.

## Ternary-Bonsai-2 27B (Prism): correct, speed depends on the file

- PTQ1_0 aarch64 vs x86-64: PPL within 0.03%.
- ik_llama.cpp vs Prism on PQ2_0 (x86-64): KLD 2.1e-4 with `-rtr` (same top-1 99.2%), 2.6e-4 without (98.8%). 2 windows only.
- PTQ1_0 and PQ2_0 hold bit-identical weights (PrismML-Eng/llama.cpp#148). In ik_llama.cpp with `-rtr` they give the same PPL to every printed digit. In Prism the PPL differs slightly (8.7507 vs 8.7515), presumably because each type has its own kernel.
- On AVX2 x86-64, PTQ1_0 processes prompts ~6× slower than PQ2_0 in Prism (no repacked path; PrismML-Eng/llama.cpp#324).
- ik_llama.cpp with `-rtr` is ~1.9–2.1× (prompt) and ~2.3–2.4× (generation) faster than Prism on PQ2_0, in two sessions. Without `-rtr`, prompt processing is ~3.1–3.5× slower.
- **Update, 2026-10-11:** llama.cpp merged support for this model's architecture (the Hadamard rotation; ggml-org/llama.cpp#29600), but not the PQ2_0/PTQ1_0 types (`master` still has 43 ggml types). The published 27B files therefore still need the Prism fork or ik_llama.cpp.
- ik_llama.cpp labels this PQ2_0 file "IQ5_K". That is cosmetic: `general.file_type` 141 means PQ2_0 in Prism and IQ5_K in ik_llama.cpp.

## How it was measured

See [METHOD.md](METHOD.md): protocol, the two kinds of reference, what counts as a divergence, and seven pitfalls.

## Reproduce

```bash
WORK=$PWD/work scripts/build.sh all     # llama.cpp, Prism, ik_llama.cpp, 3 bitnet.cpp variants
WORK=$PWD/work scripts/run.sh quick     # BitNet + Bonsai 1.7B; "full" adds the 27B (>= 16 GB RAM)
                                        # "27b" runs only the 27B part (needs prism and ik)
scripts/compare.py work/results.jsonl   # SAME / CLOSE / DIFFERS vs the numbers published here
```

Every input is pinned: runtime commits, model files (revision and sha256) and WikiText-2. `run.sh` records each measurement with the CPU, the runtime and submodule commits, and the flags.

**Reproduced from scratch on aarch64** (Neoverse-N1, clang 18, 2026-10-11): all builds, then `run.sh quick` in about 2 h. Result: **43 of 43** comparable measurements came out SAME (greedy text, perplexity, KL, which file loads where). Speed is not judged: prompt processing came within 2.5% of the published numbers, generation within 10%.

**Reproduced from scratch on x86-64** (hosted VM, Intel Xeon @ 2.2 GHz, gcc 13.3, right after publication): 32 SAME, 3 CLOSE, 0 DIFFERS, plus 13 measurements that were new on x86-64 and are now part of `compare.py`. The 3 CLOSE are within the spread between valid code paths:

- Q2_0 g64 in llama.cpp and Prism gave 18.3723, against 18.3707 on the AMD EPYC used before. On x86-64 this type runs generic C code, which the compiler vectorizes differently for each CPU.
- bitnet.cpp June 2025 gave 18.8431 with gcc, against 18.8137 with clang.

**The 27B part, reproduced from scratch on x86-64** (Intel Xeon @ 2.2 GHz, gcc 13.3, `run.sh 27b`, 2026-10-14): about 15 min of builds and 2 h of measurements on a 4-vCPU VM. **5 of 5** perplexity and KL values came out SAME. Speed is not judged; the ratios between runtimes moved by up to 16% (see the 27B section).

- The greedy-text check first said DIFFERS. That was a bug in `compare.py`, not in the model: the 27B ends each sentence with a newline, and the expected text had been stored without them. Fixed.
- The text was then generated again, with the same function, on another x86-64 VM (AMD EPYC 7B12): both files gave text byte-identical to aarch64. On the Xeon, the run's summary was cut at 150 columns; the text matched up to that point ("…The capital of Spain is M").
- `quick` and `27b` together are `full`. On x86-64 they ran in separate sessions.

## Feedback

Found an error, or have a CPU, runtime or model file that is not covered here? Please open an issue in this repository. Include the runtime commit, the CPU (`lscpu` on Linux), the file, and the exact command and output. Corrections are especially welcome: if one of the linked fixes gets merged, or a number here does not reproduce, the report will be updated and the change credited.

## Limitations

- One file per format, 512-token context, English text, 8 windows (2 for the 27B model).
- Real hardware: aarch64 Neoverse-N1 and x86-64 AVX2 (Intel Xeon @ 2.2 GHz, AMD EPYC 7B12). The older classes were emulated with build flags.
- No GPU, AVX-512, Arm i8mm/SVE or Apple Silicon.
- KL against an exact reference: Bonsai 1.7B on aarch64 and x86-64; 27B only between runtimes (x86-64). For BitNet, only PPL was compared with the HF reference, not KL.
- Runtimes move fast: the linked issues may have changed since October 2026.

## AI assistance

This work was done with the help of Claude Code (Anthropic), an AI coding assistant. It drafted scripts and text, helped analyze logs, and ran many of the measurements on the author's machines under the author's direction. The author reviewed every number against the logs and is responsible for the content. The AI is not an author. Commits made with its help carry an `Assisted-by:` trailer.

## License

- Scripts in `scripts/`: [Apache-2.0](LICENSE).
- Report text, tables and data in `results/`: [CC BY 4.0](LICENSE-CC-BY-4.0.txt).
- No model weights or third-party runtime code are redistributed here. The scripts download them from their original repositories, under their own licenses.
