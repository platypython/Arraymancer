#!/usr/bin/env python3
"""Check and time allocating Metal reductions against pinned PyTorch MPS.

Build reduction_runner.nim with release, metal and --path:src first. Inputs stay
resident; both implementations synchronize only at sample boundaries. Three
paired rounds alternate backend order and require every median and p95 to win.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

os.environ.update({name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch

CASES = ("prod", "sum_rows", "sum_columns", "mean_rows", "mean_columns",
         "prod_rows", "prod_columns")
ARTIFACTS = (
    "src/arraymancer/tensor/aggregate_metal.nim",
    "src/arraymancer/tensor/backend/native/metal_bridge.mm",
    "src/arraymancer/tensor/backend/native/metal_kernels.metal",
    "benchmarks/pytorch_head_to_head/reduction_runner.nim",
    "benchmarks/pytorch_head_to_head/benchmark_reductions.py",
    "build/apple/libarraymancer_metal.dylib",
    "build/apple/arraymancer.metallib",
)


def operation(name: str, value: torch.Tensor) -> Callable[[], torch.Tensor]:
    """Resolve the callable before timing; include output allocation each call."""
    if name == "prod":
        return lambda: value.prod()
    kind, direction = name.split("_")
    axis = -1 if direction == "rows" else -2
    if kind == "sum":
        return lambda: value.sum(dim=axis, keepdim=True)
    if kind == "mean":
        return lambda: value.mean(dim=axis, keepdim=True)
    return lambda: value.prod(dim=axis, keepdim=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/reduction_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/reduction_benchmark"))
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 13, 18) or torch.__version__.split("+")[0] != "2.15.0":
        raise RuntimeError("Requires pinned CPython 3.13.18 and PyTorch 2.15.0")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = [Path(path) for path in ARTIFACTS] + [args.runner]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    rows, columns = 128, 128
    # Nonzero, nonconstant products near one avoid meaningless underflow comparisons.
    source = (1 + ((torch.arange(rows * columns) % 17) - 8) * 0.0001).float().reshape(rows, columns)
    manifest = {"rows": rows, "columns": columns, "values": source.flatten().tolist(),
                "warmup": 20, "repeats": 15, "iterations": 50}
    input_path = args.output / "input.json"
    input_path.write_text(json.dumps(manifest) + "\n")
    gpu = source.to("mps")
    callables = {name: operation(name, gpu) for name in CASES}
    expected: dict[str, Any] = {}
    for name in CASES:
        cpu_result = operation(name, source)()
        actual = callables[name]()
        torch.testing.assert_close(actual.cpu(), cpu_result, atol=5e-5, rtol=5e-5)
        expected[name] = {"shape": list(cpu_result.shape), "values": cpu_result.flatten().tolist()}

    def native() -> dict[str, Any]:
        """Preserve child output before checking shape and every numerical value."""
        result = subprocess.run([str(args.runner.resolve()), str(input_path.resolve())],
                                check=True, capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        for name in CASES:
            actual = report[name]["observation"]
            if actual["shape"] != expected[name]["shape"]:
                raise RuntimeError(f"{name}: shape mismatch")
            torch.testing.assert_close(torch.tensor(actual["values"]),
                                       torch.tensor(expected[name]["values"]), atol=5e-5, rtol=5e-5)
        return report

    def reference() -> dict[str, Any]:
        """Match the native runner's warmup, samples, iterations and synchronization."""
        report = {}
        for name, call in callables.items():
            for _ in range(manifest["warmup"]):
                call()
            torch.mps.synchronize()
            samples = []
            for _ in range(manifest["repeats"]):
                torch.mps.synchronize()
                start = time.perf_counter_ns()
                for _ in range(manifest["iterations"]):
                    result = call()
                torch.mps.synchronize()
                samples.append((time.perf_counter_ns() - start) / 1e6 / manifest["iterations"])
                del result
            report[name] = {"samples_ms": samples, "median_ms": statistics.median(samples),
                            "p95_ms": sorted(samples)[math.ceil(len(samples) * .95) - 1]}
        return report

    rounds = []
    failures = []
    native()  # Complete native correctness before either implementation is timed competitively.
    for index in range(3):
        if index % 2 == 0:
            torch_result, native_result = reference(), native()
        else:
            native_result, torch_result = native(), reference()
        rounds.append({"order": "torch-first" if index % 2 == 0 else "arraymancer-first",
                       "pytorch": torch_result, "arraymancer": native_result})
        for name in CASES:
            for metric in ("median_ms", "p95_ms"):
                if native_result[name][metric] >= torch_result[name][metric]:
                    failures.append(f"round {index + 1}: {name} {metric}")
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)] for path in artifacts):
        raise RuntimeError("Sources or native artifacts changed during measurement")
    report = {"torch": torch.__version__, "python": sys.version,
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
              "artifacts_sha256": hashes,
              "rounds": rounds, "failures": failures}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"failures": failures, "results": str(args.output / "results.json")}))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
