"""Evaluate the forensics and fraud-ring components without labeled claims.

Writes `models/benchmarks/tampering_benchmark.json` (known edits applied to
clean bills, scored by the production forensic checks) and
`models/benchmarks/ring_benchmark.json` (generated claim populations with
known rings, scored by `src.graph.rings`). The web app's Model insights page
and case file read both files; nothing is recomputed at request time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schemas import PROJECT_ROOT
from src.forensics.benchmark import run_tampering_benchmark
from src.graph.ring_benchmark import RingScenario, run_ring_benchmark

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "benchmarks"
TAMPERING_FILE = "tampering_benchmark.json"
RING_FILE = "ring_benchmark.json"


def _write(directory: Path, name: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def run_tampering(output_dir: Path, n_bills: int, seed: int, images_dir: Optional[Path]) -> None:
    started = time.time()

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  tampering: {done}/{total} samples scored", flush=True)

    report = run_tampering_benchmark(n_bills=n_bills, seed=seed, images_dir=images_dir, progress=progress)
    report["runtime_seconds"] = round(time.time() - started, 1)
    path = _write(output_dir, TAMPERING_FILE, report)
    for condition, block in report["results"].items():
        verdict = block["methods"]["combined_verdict"]
        print(f"  {condition}: verdict detection {_pct(verdict['detection_rate'])}, "
              f"false positives {_pct(verdict['false_positive_rate'])}")
    print(f"wrote {path}")


def run_rings(output_dir: Path, n_claims: int, seeds: tuple[int, ...]) -> None:
    started = time.time()
    report = run_ring_benchmark(RingScenario(n_claims=n_claims), seeds=seeds,
                                progress=lambda msg: print(f"  rings: {msg}", flush=True))
    report["runtime_seconds"] = round(time.time() - started, 1)
    path = _write(output_dir, RING_FILE, report)
    for row in report["noise_sweep"]:
        print(f"  noise {row['noise']:.1f}: recovery raw {_pct(row['raw']['ring_recovery_rate'])}, "
              f"normalized {_pct(row['normalized']['ring_recovery_rate'])}, "
              f"fuzzy {_pct(row['fuzzy']['ring_recovery_rate'])}")
    print(f"wrote {path}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=["tampering", "rings", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bills", type=int, default=40, help="rendered bills (ignored with --images-dir)")
    parser.add_argument("--images-dir", type=Path, help="folder of clean real bill images to tamper instead")
    parser.add_argument("--n-claims", type=int, default=15420, help="claims per generated ring population")
    parser.add_argument("--ring-seeds", type=int, default=3, help="populations averaged per setting")
    args = parser.parse_args(argv)
    try:
        if args.component in ("tampering", "both"):
            run_tampering(args.output_dir, args.n_bills, args.seed, args.images_dir)
        if args.component in ("rings", "both"):
            run_rings(args.output_dir, args.n_claims, tuple(args.seed + i for i in range(args.ring_seeds)))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
