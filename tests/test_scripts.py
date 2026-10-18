# SPDX-License-Identifier: Apache-2.0
"""Checks of the scripts themselves: no downloads, no builds, a few seconds.

    python3 -m unittest discover tests
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import compare  # noqa: E402
import parse_log  # noqa: E402


def run_compare(records):
    """Write records as JSONL, run compare.py, return (exit code, output)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "results.jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        p = subprocess.run([sys.executable, "-I", str(SCRIPTS / "compare.py"),
                            path], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def rec(label, kind, arch="aarch64", **fields):
    return {"label": label, "kind": kind, "cpu": f"{arch} | test | 2 CPUs",
            **fields}


def statuses(out):
    return [line.split()[0] for line in out.splitlines()
            if line[:1].isupper() and not line.startswith("summary")]


class ParseLog(unittest.TestCase):
    def test_final_ppl_both_runtimes(self):
        self.assertEqual(parse_log.ppl(
            "[1]13.9,[2]20.1,\nFinal estimate: PPL = 23.1558 +/- 1.80119")
            ["ppl"], 23.1558)
        self.assertEqual(parse_log.ppl(
            "Final estimate: PPL over 8 chunks for n_ctx=512 = 18.81 +/- 1.30")
            ["ppl"], 18.81)

    def test_nan_windows_without_final_line(self):
        # what llama-perplexity prints when the logits are nan
        out = parse_log.ppl("[1]nan,[2]-nan,\n"
                            "Unexpected negative standard deviation")
        self.assertIsNone(out["ppl"])
        self.assertEqual(out["ppl_windows"], ["nan", "nan"])

    def test_kl_nan_negative_and_missing(self):
        self.assertEqual(parse_log.kl("Mean    KLD:        nan ±        nan")
                         ["kld_mean"], "nan")
        self.assertEqual(parse_log.kl("Mean    KLD:       -nan ±       -nan")
                         ["kld_mean"], "nan")
        self.assertEqual(parse_log.kl("Mean    KLD:  -0.000012 ±   0.000003")
                         ["kld_mean"], -0.000012)
        self.assertIsNone(parse_log.kl("")["kld_mean"])

    def test_missing_log_gives_nulls(self):
        p = subprocess.run([sys.executable, "-I", str(SCRIPTS / "parse_log.py"),
                            "kl", "/nonexistent/x.log"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIsNone(json.loads(p.stdout)["kld_mean"])


class Compare(unittest.TestCase):
    def test_every_published_value_is_same(self):
        records = []
        for a, table in compare.PPL.items():
            for label, v in table.items():
                n = 2 if "27b" in label else compare.FULL_WINDOWS
                records.append(rec(label, "ppl", a, ppl=v,
                                   ppl_windows=[v] * n))
        for a, table in compare.KLD.items():
            for label, v in table.items():
                records.append(rec(label, "kl", a, kld_mean=v))
        for (rt, fl), v in compare.LOADS.items():
            records.append(rec(f"load-{rt}-{fl}", "load", loads=v, rc=1))
        for key, text in compare.TEXT.items():
            label = key if key.startswith(("bitnet", "ik-")) else f"prism-{key}"
            # the 27B ends each sentence with a newline (bug fixed 2026-10-14)
            text = text.replace(". ", ".\n") if "27b" in key else text
            records.append(rec(label, "greedy", text=compare.PROMPT + text))
        for a, text in (("aarch64", "????????????"),
                        ("x86_64", " a small city, and the capital of France"
                                   " is a small")):
            records.append(rec("bitnet-official", "greedy", a,
                               text=compare.PROMPT + text))
        code, out = run_compare(records)
        self.assertEqual(code, 0, out)
        self.assertEqual(set(statuses(out)), {"SAME"}, out)

    def test_failed_kl_is_differs_not_new(self):
        for value in (None, "nan"):
            code, out = run_compare(
                [rec("kl-prism-pq2_0", "kl", kld_mean=value, rc=139)])
            self.assertEqual(code, 1, out)
            self.assertEqual(statuses(out), ["DIFFERS"], out)
            self.assertIn("exit code 139", out)

    def test_crash_shows_exit_code(self):
        code, out = run_compare([
            rec("prism-pq2_0", "greedy", text="", rc=139),
            rec("prism-pq2_0", "ppl", ppl=None, ppl_windows=[], rc=139),
            rec("bench-prism-pq2_0", "bench", bench={}, rc=139)])
        self.assertEqual(code, 1, out)
        self.assertEqual(statuses(out), ["DIFFERS", "DIFFERS", "INFO"], out)
        self.assertEqual(out.count("exit code 139"), 3, out)

    def test_rerun_keeps_the_last_record(self):
        code, out = run_compare([
            rec("prism-q1_0", "ppl", ppl=None, ppl_windows=[], rc=1),
            rec("prism-q1_0", "ppl", ppl=23.155823, ppl_windows=[0] * 8)])
        self.assertEqual(code, 0, out)
        self.assertEqual(statuses(out), ["SAME"], out)
        self.assertIn("1 older records replaced", out)

    def test_records_without_rc_still_work(self):
        # results.jsonl written before records carried the exit code
        code, out = run_compare(
            [rec("prism-q1_0", "ppl", ppl=23.1558, ppl_windows=[0] * 8)])
        self.assertEqual((code, statuses(out)), (0, ["SAME"]), out)


class Measure(unittest.TestCase):
    def test_crashing_runtime_is_recorded_with_its_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            work, bindir = pathlib.Path(d, "work"), pathlib.Path(d, "bin")
            bindir.mkdir()
            for tool in ("llama-completion", "llama-perplexity", "llama-bench"):
                t = bindir / tool
                t.write_text("#!/bin/sh\nkill -SEGV $$\n")
                t.chmod(0o755)
            model = pathlib.Path(d, "m.gguf")
            model.touch()
            script = (f'. "{SCRIPTS}/common.sh"; . "{SCRIPTS}/measure.sh"\n'
                      f'm_greedy x "{bindir}" "{model}" 8 "The capital is"\n'
                      f'm_ppl x "{bindir}" "{model}" 8\n'
                      f'm_kl x "{bindir}" "{model}" "{d}/base.kld"\n'
                      f'm_bench x "{bindir}" "{model}" 512 64 1\n')
            subprocess.run(["bash", "-c", script], capture_output=True,
                           env={**os.environ, "WORK": str(work)}, check=True)
            out = [json.loads(x) for x in
                   (work / "results.jsonl").read_text().splitlines()]
        self.assertEqual([r["kind"] for r in out],
                         ["greedy", "ppl", "kl", "bench"])
        self.assertEqual({r["rc"] for r in out}, {139})


if __name__ == "__main__":
    unittest.main()
