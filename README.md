# Low-bit LLMs on CPU: what runs correctly and fast where (Oct 2026)

> **Status: draft, not published yet.**

An independent, reproducible check of low-bit (1-bit, ternary, 2-bit) LLM files on CPU across runtimes and CPU classes: does the output match a reference implementation, and how fast is it?

## Contents (planned)

- `README.md` — the report: findings, tables, limitations
- `METHOD.md` — how the measurements were taken, and pitfalls found along the way
- `results/` — measurements as JSON (runtime commit, file sha256, CPU, flags, metrics)
- `scripts/` — scripts to rebuild the runtimes at pinned commits, fetch the files and reproduce every number

## License

- Scripts in `scripts/`: [Apache-2.0](LICENSE).
- Report text, tables and data in `results/`: [CC BY 4.0](LICENSE-CC-BY-4.0.txt).
- No model weights or third-party runtime code are redistributed here. The scripts download them from their original repositories, under their own licenses.
