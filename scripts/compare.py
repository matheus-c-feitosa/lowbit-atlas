#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare a fresh run (results.jsonl from run.sh) with the published numbers.

    scripts/compare.py work/results.jsonl

For each record, finds the published value for the same label and CPU
architecture (values from results/*.json) and prints:
  SAME     identical to the printed precision (same code path)
  CLOSE    within 0.2% PPL / 2x KLD (the spread between valid code paths)
  DIFFERS  outside that range: worth a look
  NEW      nothing published for this label on this architecture
  PARTIAL  fewer windows than published and no per-window value to compare
Speed is only reported (machines differ), never judged.
"""
import json
import sys

# label -> value, per architecture. aarch64: Neoverse-N1 (NEON+dotprod);
# x86_64: AVX2 without AVX-512. Sources: results/*.json.
PPL = {
    "aarch64": {
        "llama-q1_0": 23.155823, "prism-q1_0": 23.155823, "ik-q1_0": 23.1423,
        "ik-rtr-q1_0": 23.1650, "llama-q2_0_g64": 18.373330,
        "prism-q2_0_g64": 18.373330, "prism-pq2_0": 18.3701,
        "ik-pq2_0": 18.3528, "ik-rtr-pq2_0": 18.3449,
        "llama-f16-from-q1_0": 23.1607, "prism-f16-from-pq2_0": 18.3791,
        "bitnet-fixed": 18.8444, "bitnet-605": 18.8444,
        "bitnet-2025-06": 18.8515,
        "ik-bitnet": 18.8127, "prism-27b-ptq1_0": 8.7532,
    },
    "x86_64": {
        "llama-q1_0": 23.1792, "prism-q1_0": 23.1535,
        "llama-q2_0_g64": 18.3707, "prism-q2_0_g64": 18.3707,
        "prism-pq2_0": 18.3694, "bitnet-official": 113.3283,
        "bitnet-fixed": 18.8676, "bitnet-2025-06": 18.8137,
        "bitnet-605": 18.8676,  # expected = fixed; not yet measured on x86
        "ik-bitnet": 18.8519, "prism-27b-ptq1_0": 8.7507,
        "prism-27b-pq2_0": 8.7515, "ik-rtr-27b-ptq1_0": 8.7552,
    },
}
# Per-window running PPL, to check runs with fewer windows (BitNet only).
_FIX_ARM = [14.4323, 21.8927, 19.8875, 21.9400, 19.7718, 18.6801, 18.7215,
            18.8444]
_FIX_X86 = [14.5202, 22.0500, 19.9557, 21.9956, 19.8102, 18.6962, 18.7371,
            18.8676]
WINDOWS = {
    "aarch64": {
        "bitnet-fixed": _FIX_ARM, "bitnet-605": _FIX_ARM,
        "bitnet-2025-06": [14.2660, 21.8564, 19.7821, 21.8715, 19.7656,
                           18.7084, 18.7179, 18.8515],
        "ik-bitnet": [14.2446, 21.8109, 19.7668, 21.8767, 19.6838, 18.6085,
                      18.6467, 18.8127],
    },
    "x86_64": {
        "bitnet-fixed": _FIX_X86, "bitnet-605": _FIX_X86,
        "bitnet-official": [47.2744, 90.3237, 92.0372, 108.6810, 102.6649,
                            104.7082, 109.9777, 113.3283],
        "bitnet-2025-06": [14.5193, 22.0187, 19.8675, 21.9324, 19.7624,
                           18.6729, 18.6838, 18.8137],
    },
}
FULL_WINDOWS = 8  # 2 for the 27B labels

KLD = {
    "aarch64": {
        "kl-llama-q1_0": 0.000224, "kl-llama-q1_0-vecdot": 0.000238,
        "kl-ik-q1_0": 0.000248, "kl-ik-rtr-q1_0": 0.000450,
        "kl-prism-pq2_0": 0.000986, "kl-ik-pq2_0": 0.000946,
        "kl-ik-rtr-pq2_0": 0.000771,
    },
    "x86_64": {"kl-ik-27b-pq2_0": 0.000261, "kl-ik-rtr-27b-pq2_0": 0.000212},
}
TEXT = {
    "q1_0": " Paris, and the capital of Spain is",
    "q2_0_g64": " Paris. The capital of France is Paris",
    "pq2_0": " Paris. The capital of France is Paris",
    "bitnet-fixed": " Paris. Paris is a city that is known for its rich",
    "bitnet-2025-06": " Paris. Paris is a city that is known for its rich",
    "bitnet-605": " Paris. Paris is a city that is known for its rich",
    "bitnet-605-nokey": " a small city, and the capital of France is a small",
    "ik-bitnet": " Paris. Paris is a city that is known for its rich",
}
LOADS = {  # runtime, file -> loads?
    ("llama", "q1_0"): True, ("llama", "q2_0_g64"): True,
    ("llama", "pq2_0"): False, ("llama", "q2_0_legacy"): False,
    ("prism", "q1_0"): True, ("prism", "q2_0_g64"): True,
    ("prism", "pq2_0"): True, ("prism", "q2_0_legacy"): False,
    ("ik", "q1_0"): True, ("ik", "q2_0_g64"): False, ("ik", "pq2_0"): True,
    ("ik", "q2_0_legacy"): False,
}
PROMPT = "The capital of France is"


def arch(rec):
    return rec["cpu"].split("|")[0].strip()


def judge_ppl(got, want):
    if got is None:
        return "DIFFERS"
    if round(got, 4) == round(want, 4):
        return "SAME"
    return "CLOSE" if abs(got / want - 1) <= 0.002 else "DIFFERS"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rows, counts = [], {}
    for line in open(sys.argv[1]):
        r = json.loads(line)
        a, label, kind = arch(r), r["label"], r["kind"]
        status, got, want = "NEW", "", ""
        if kind == "ppl":
            got = r.get("ppl")
            want = PPL.get(a, {}).get(label)
            n = len(r.get("ppl_windows") or [])
            full = 2 if "27b" in label else FULL_WINDOWS
            if want is not None and 0 < n < full:
                win = WINDOWS.get(a, {}).get(label)
                if win:
                    want = win[n - 1]
                    status = judge_ppl(got, want)
                else:
                    status = "PARTIAL"
                want = f"{want} ({n} of {full} windows)"
            elif want is not None:
                status = judge_ppl(got, want)
        elif kind == "kl":
            got = r.get("kld_mean")
            want = KLD.get(a, {}).get(label)
            if want is not None and got is not None:
                status = "SAME" if abs(got - want) < 5e-6 else (
                    "CLOSE" if 0.5 <= got / want <= 2 else "DIFFERS")
        elif kind == "greedy":
            got = r.get("text", "").replace(PROMPT, "", 1).replace("\n", " ")
            key = label if label in TEXT else label.split("-", 1)[-1]
            want = TEXT.get(key)
            if label == "bitnet-official":
                want = "????????????" if a == "aarch64" else \
                    " a small city, and the capital of France is a small"
            if want is not None:
                status = "SAME" if got.strip() == want.strip() else "DIFFERS"
        elif kind == "load":
            _, rt, fl = label.split("-", 2)
            want = LOADS.get((rt, fl))
            got = r.get("loads")
            if want is not None:
                status = "SAME" if got == want else "DIFFERS"
        elif kind == "bench":
            got = " ".join(f"{k} {v['tps']}" for k, v in r["bench"].items())
            status = "INFO"
        counts[status] = counts.get(status, 0) + 1
        rows.append((status, kind, label, a, got, want))
    w = max(len(x[2]) for x in rows) if rows else 10
    for status, kind, label, a, got, want in rows:
        print(f"{status:8} {kind:6} {label:{w}} {a:8} got={got} "
              f"published={want if want != '' else '-'}")
    print("\nsummary:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    sys.exit(1 if counts.get("DIFFERS") else 0)


if __name__ == "__main__":
    main()
