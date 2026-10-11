#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parse llama.cpp / ik_llama.cpp tool output into JSON.

    parse_log.py ppl   <llama-perplexity log>
    parse_log.py kl    <llama-perplexity --kl-divergence log>
    parse_log.py bench <llama-bench -o md output>

Prints one JSON object. Fields that are missing in the log are null.
Both runtimes print the same KL block; the final PPL line differs:
  llama.cpp: "Final estimate: PPL = 23.1558 +/- 1.80119"
  ik:        "Final estimate: PPL over 8 chunks for n_ctx=512 = 18.81 +/- 1.30"
"""
import json
import re
import sys

NUM = r"([0-9.]+|nan|inf)"


def num(s):
    # nan/inf are kept as strings so the output stays valid JSON
    if s is None:
        return None
    return s if s in ("nan", "inf") else float(s)


def ppl(text):
    m = re.search(r"Final estimate: PPL(?: over \d+ chunks for n_ctx=\d+)? = "
                  + NUM + r" \+/- " + NUM, text)
    windows = [num(w) for w in re.findall(r"\[\d+\]" + NUM, text)]
    return {"ppl": num(m.group(1)) if m else None,
            "ppl_stderr": num(m.group(2)) if m else None,
            "ppl_windows": windows}


def kl(text):
    def grab(pattern, n=1):
        m = re.search(pattern, text)
        return [num(m.group(i)) for i in range(1, n + 1)] if m else [None] * n
    ppl_q, = grab(r"Mean PPL\(Q\)\s*:\s*" + NUM)
    kld, kld_err = grab(r"Mean\s+KLD:\s*" + NUM + r"\s*±\s*" + NUM, 2)
    kld_max, = grab(r"Maximum KLD:\s*" + NUM)
    kld_p99, = grab(r"99\.0%\s+KLD:\s*" + NUM)
    rms, rms_err = grab(r"RMS Δp\s*:\s*" + NUM + r"\s*±\s*" + NUM, 2)
    top1, top1_err = grab(r"Same top p:\s*" + NUM + r"\s*±\s*" + NUM, 2)
    return {"ppl_q": ppl_q, "kld_mean": kld, "kld_mean_err": kld_err,
            "kld_max": kld_max, "kld_p99": kld_p99,
            "rms_dp_pct": rms, "rms_dp_err": rms_err,
            "top1_same_pct": top1, "top1_same_err": top1_err}


def bench(text):
    tests = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        m = re.match(r"([0-9.]+) ± ([0-9.]+)", cells[-1])
        if m and re.match(r"(pp|tg)\d+", cells[-2]):
            tests[cells[-2]] = {"tps": float(m.group(1)),
                                "sd": float(m.group(2))}
    return {"bench": tests}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("ppl", "kl", "bench"):
        sys.exit(__doc__)
    with open(sys.argv[2], encoding="utf-8", errors="replace") as f:
        text = f.read()
    out = {"ppl": ppl, "kl": kl, "bench": bench}[sys.argv[1]](text)
    print(json.dumps(out, allow_nan=False))


if __name__ == "__main__":
    main()
