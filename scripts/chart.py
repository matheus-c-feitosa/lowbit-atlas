#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Draw the summary figures from results/*.json (standard library only).

    scripts/chart.py            # writes assets/key-findings.svg and
                                # assets/social-preview.svg

To get PNGs, open the SVGs in any browser, or for example:
    chromium --headless --no-sandbox --hide-scrollbars \
      --window-size=1600,900 --screenshot=key-findings.png key-findings.svg
"""
import json
import math
import os
from xml.sax.saxutils import escape

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FONT = "DejaVu Sans, Noto Sans, Helvetica, Arial, sans-serif"
INK, GRAY, LIGHT = "#1f2328", "#59636e", "#d1d9e0"
GREEN, RED, BLUE = "#1a7f37", "#cf222e", "#0969da"
REPO = "github.com/matheus-c-feitosa/lowbit-atlas"


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def text(x, y, s, size=18, color=INK, weight="normal", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}">'
            f'{escape(s)}</text>')


def bitnet_rows():
    runs = {r["id"]: r for r in load("bitnet-b1.58-2b-4t.json")["runs"]}
    ref = runs["reference-transformers"]["ppl"]

    def span(*ids):
        v = [runs[i]["ppl"] for i in ids]
        return min(v) / ref, max(v) / ref, min(v), max(v)

    rows = []
    lo, hi, a, b = span("ik-arm", "ik-x86")
    rows.append(("ik_llama.cpp", "x86 and ARM", hi, f"{a:.2f}–{b:.2f}"))
    lo, hi, a, b = span("bitnetcpp-arm-pr2-relu2", "bitnetcpp-x86-588fix")
    rows.append(("bitnet.cpp + one-line fix (#588)",
                 "x86, and ARM with llama.cpp#2", hi,
                 f"{a:.2f}–{b:.2f}"))
    p = runs["bitnetcpp-x86-official"]["ppl"]
    rows.append(("bitnet.cpp main", "x86: text looks fine", p / ref,
                 f"{p:.1f}"))
    rows.append(("bitnet.cpp main", "ARM: prints " +
                 runs["bitnetcpp-arm-official"]["greedy"], None, ""))
    return ref, rows


def speed_rows():
    s = load("bonsai-1.7b.json")["x86_reproduction"]["speed_tps_pp512_tg64"]
    return [
        ("ik_llama.cpp -rtr", "PQ2_0", s["ik-rtr-pq2_0"][0]),
        ("Prism", "PQ2_0", s["prism-pq2_0"][0]),
        ("ik_llama.cpp, no -rtr", "PQ2_0", s["ik-pq2_0"][0]),
        ("Prism", "Q2_0 g64, no SIMD kernel yet*", s["prism-q2_0_g64"][0]),
    ]


def key_findings():
    W, H = 1600, 900
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
           f'height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           text(60, 78, "Low-bit LLMs on CPU: same model, very different "
                "results", 40, weight="bold"),
           text(60, 114, "Independent check against references, Oct 2026 · "
                + REPO, 20, GRAY)]

    # left: BitNet correctness, ratio to the reference on a log axis
    ref, rows = bitnet_rows()
    x0, y0 = 60, 190
    out += [text(x0, y0, "BitNet b1.58 2B4T: is the output right?", 26,
                 weight="bold"),
            text(x0, y0 + 30, f"Perplexity vs. HF reference ({ref:.2f}) on "
                 "identical tokens, log scale", 17, GRAY)]
    ax0, ax1, gx, vx, top = x0 + 380, x0 + 575, x0 + 650, x0 + 685, y0 + 70
    ticks = [1, 2, 5, 10]

    def ax(r):
        return ax0 + (ax1 - ax0) * math.log10(r)

    for t in ticks:
        out.append(f'<line x1="{ax(t):.1f}" y1="{top}" x2="{ax(t):.1f}" '
                   f'y2="{top + len(rows) * 92 - 30}" stroke="{LIGHT}"/>')
        out.append(text(ax(t), top + len(rows) * 92 - 8, f"{t}×", 15, GRAY,
                        anchor="middle"))
    out.append(text(gx, top + len(rows) * 92 - 8, "garbage", 15, GRAY,
                    anchor="middle"))
    for i, (name, where, ratio, val) in enumerate(rows):
        y = top + 30 + i * 92
        out.append(text(x0, y, name, 19, weight="bold"))
        out.append(text(x0, y + 24, where, 16, GRAY))
        if ratio is None:
            cx, color = gx, INK
            out.append(f'<path d="M{cx - 9} {y - 15} l18 18 M{cx + 9} '
                       f'{y - 15} l-18 18" stroke="{color}" '
                       f'stroke-width="4"/>')
        else:
            cx = ax(ratio)
            color = GREEN if ratio < 1.01 else RED
            out.append(f'<line x1="{ax0}" y1="{y - 6}" x2="{cx:.1f}" '
                       f'y2="{y - 6}" stroke="{color}" stroke-width="4"/>')
            out.append(f'<circle cx="{cx:.1f}" cy="{y - 6}" r="10" '
                       f'fill="{color}"/>')
        if ratio is not None and ratio >= 1.01:
            val = f"{val} ({ratio:.1f}×)"
        out.append(text(vx, y, val, 17, color, "bold"))

    # right: Bonsai speed, prompt processing on x86
    x0 = 930
    out += [text(x0, y0, "Ternary-Bonsai 1.7B: how fast? (x86 AVX2)", 26,
                 weight="bold"),
            text(x0, y0 + 30, "Prompt processing, tokens/s (llama-bench "
                 "pp512, 2 threads, one session)", 17, GRAY)]
    sp = speed_rows()
    vmax, bx, bw = max(v for _, _, v in sp), x0 + 240, 280
    for i, (name, fmt, v) in enumerate(sp):
        y = top + 30 + i * 92
        out.append(text(x0, y, name, 19, weight="bold"))
        out.append(text(x0, y + 24, fmt, 16, GRAY))
        w = max(3, bw * v / vmax)
        out.append(f'<rect x="{bx}" y="{y - 22}" width="{w:.1f}" '
                   f'height="30" fill="{BLUE}" rx="3"/>')
        out.append(text(bx + w + 12, y, f"{v:.1f}", 18, INK, "bold"))
    ratio = sp[0][2] / sp[-1][2]
    out.append(text(x0, top + 4 * 92 + 34, f"Same model, {ratio:.0f}× apart "
                    "depending on file and runtime.", 19, INK, "bold"))
    out.append(text(x0, top + 4 * 92 + 62, "*Not in Prism yet (read in its "
                    "code). llama.cpp merged an AVX2", 15, GRAY))
    out.append(text(x0, top + 4 * 92 + 82, "Q2_0 kernel on 2026-10-13: ~6× "
                    "faster, measured separately.", 15, GRAY))

    out.append(f'<line x1="60" y1="{H - 70}" x2="{W - 60}" y2="{H - 70}" '
               f'stroke="{LIGHT}"/>')
    out.append(text(60, H - 36, "Intel Xeon @ 2.2 GHz (AVX2) and Arm "
                    "Neoverse-N1. Every number comes with its commit, file "
                    "sha256 and the scripts to rerun it.", 17, GRAY))
    out.append("</svg>")
    return "\n".join(out)


def social_preview():
    W, H = 1280, 640
    ref, rows = bitnet_rows()
    sp = speed_rows()
    lines = [
        "Microsoft's bitnet.cpp gives wrong output on CPU: "
        f"{rows[2][2]:.0f}× perplexity on x86,",
        "garbage on ARM. Only the activation fix works on current main.",
        "Bonsai (Prism) runs correctly; speed for the same model varies "
        f"{sp[0][2] / sp[-1][2]:.0f}×",
        "with file and runtime (ik_llama.cpp -rtr is usually fastest).",
        "Reproducible: commits, sha256 and scripts for every number.",
    ]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
           f'height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<rect width="{W}" height="12" fill="{BLUE}"/>',
           text(70, 140, "Low-bit LLMs on CPU", 64, weight="bold"),
           text(70, 210, "what runs correctly and fast where", 44, GRAY)]
    y = 310
    for i, s in enumerate(lines):
        bullet = i in (0, 2, 4)
        if bullet:
            out.append(f'<circle cx="82" cy="{y - 9}" r="7" fill="{BLUE}"/>')
        out.append(text(108, y, s, 27))
        y += 46 if bullet and i != 4 else 62
    out.append(text(70, H - 50, REPO, 26, BLUE, "bold"))
    out.append("</svg>")
    return "\n".join(out)


def main():
    d = os.path.join(ROOT, "assets")
    os.makedirs(d, exist_ok=True)
    for name, svg in (("key-findings.svg", key_findings()),
                      ("social-preview.svg", social_preview())):
        with open(os.path.join(d, name), "w") as f:
            f.write(svg + "\n")
        print("wrote", os.path.join("assets", name))


if __name__ == "__main__":
    main()
