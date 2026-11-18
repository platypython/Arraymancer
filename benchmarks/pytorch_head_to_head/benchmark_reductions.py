#!/usr/bin/env python3
"""Check and time allocating CPU/Metal reductions and scans against pinned PyTorch.

Float32 retains the existing 20-case benchmark. Integer/bool scans are CPU-only;
build their runner with:
  nim c -d:release --path:src --out:build/integer_scan_runner benchmarks/pytorch_head_to_head/integer_scan_runner.nim
Inputs stay resident; both implementations synchronize only at sample boundaries.
Three paired rounds alternate backend order and require every median and p95 to win.
Use 100 samples of 50 allocating calls; warmup matches one complete sample.
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
         "prod_rows", "prod_columns", "mean_rows_backward", "mean_columns_backward",
         "prod_backward", "prod_rows_backward", "prod_columns_backward",
         "cumsum_rows", "cumsum_columns", "cumsum_rows_backward", "cumsum_columns_backward",
         "cumprod_rows", "cumprod_columns", "cumprod_rows_backward", "cumprod_columns_backward")

INTEGER_CASES = ("cumsum_rows", "cumsum_columns", "cumprod_rows", "cumprod_columns")
INTEGER_DTYPES = ("bool", "int8", "int16", "int32", "int64",
                  "uint8", "uint16", "uint32", "uint64")
INTEGER_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool, "int8": torch.int8, "int16": torch.int16, "int32": torch.int32,
    "int64": torch.int64, "uint8": torch.uint8, "uint16": torch.uint16,
    "uint32": torch.uint32, "uint64": torch.uint64,
}
INTEGER_ARTIFACTS = (
    "src/arraymancer/tensor/aggregate.nim",
    "src/arraymancer/tensor/private/p_accelerate.nim",
    "benchmarks/pytorch_head_to_head/integer_scan_runner.nim",
    "benchmarks/pytorch_head_to_head/benchmark_reductions.py",
)
ARTIFACTS = (
    "src/arraymancer/tensor/aggregate.nim",
    "src/arraymancer/tensor/private/p_accelerate.nim",
    "src/arraymancer/autograd/gates_reduce.nim",
    "benchmarks/pytorch_head_to_head/reduction_runner.nim",
    "benchmarks/pytorch_head_to_head/benchmark_reductions.py",
)
METAL_ARTIFACTS = (
    "src/arraymancer/tensor/aggregate_metal.nim",
    "src/arraymancer/tensor/backend/metal_abi.nim",
    "src/arraymancer/tensor/backend/native/apple_abi.h",
    "src/arraymancer/tensor/backend/native/metal_bridge.mm",
    "src/arraymancer/tensor/backend/native/metal_kernels.metal",
    "build/apple/libarraymancer_metal.dylib",
    "build/apple/arraymancer.metallib",
)


def operation(name: str, value: torch.Tensor) -> Callable[[], torch.Tensor]:
    """Resolve the callable before timing; include output allocation each call."""
    if name.endswith("_backward"):
        axis = -1 if "_rows_" in name else -2

        def backward() -> torch.Tensor:
            leaf = value.detach().requires_grad_(True)
            if name == "prod_backward":
                output = leaf.prod()
            elif name.startswith("mean_"):
                output = leaf.mean(dim=axis, keepdim=True)
            elif name.startswith("cumsum_"):
                output = leaf.cumsum(dim=axis)
            elif name.startswith("cumprod_"):
                output = leaf.cumprod(dim=axis)
            else:
                output = leaf.prod(dim=axis, keepdim=True)
            output.backward(torch.ones_like(output))
            assert leaf.grad is not None
            return leaf.grad

        return backward
    if name == "prod":
        return lambda: value.prod()
    kind, direction = name.split("_")
    axis = -1 if direction == "rows" else -2
    if kind == "sum":
        return lambda: value.sum(dim=axis, keepdim=True)
    if kind == "mean":
        return lambda: value.mean(dim=axis, keepdim=True)
    if kind == "cumsum":
        return lambda: value.cumsum(dim=axis)
    if kind == "cumprod":
        return lambda: value.cumprod(dim=axis)
    return lambda: value.prod(dim=axis, keepdim=True)


def integer_operation(name: str, value: torch.Tensor) -> Callable[[], torch.Tensor]:
    """Resolve an allocating int64-output scan on a resident native-dtype input."""
    axis = -1 if name.endswith("_rows") else -2
    if name.startswith("cumsum_"):
        return lambda: value.cumsum(dim=axis)
    if name.startswith("cumprod_"):
        return lambda: value.cumprod(dim=axis)
    raise ValueError(f"Unknown integer scan: {name}")


def run_integer_benchmark(args: argparse.Namespace) -> int:
    rows, columns = 128, 128
    words = [
        -2 if index % 4 == 0 else -1 if index % 4 == 1 else 1 if index % 4 == 2 else 2
        for index in range(rows * columns)
    ]
    source = torch.tensor(words, dtype=torch.int64).reshape(rows, columns)
    source = source.to(dtype=INTEGER_TORCH_DTYPES[args.dtype])
    manifest = {
        "dtype": args.dtype,
        "rows": rows,
        "columns": columns,
        "words": words,
        "warmup": 50,
        "repeats": 100,
        "iterations": 50,
    }
    input_path = args.output / "input.json"
    input_path.write_text(json.dumps(manifest) + "\n")
    callables = {name: integer_operation(name, source) for name in INTEGER_CASES}
    expected: dict[str, dict[str, Any]] = {}
    for name in INTEGER_CASES:
        result = callables[name]()
        expected[name] = {
            "shape": list(result.shape),
            "dtype": str(result.dtype).removeprefix("torch."),
            "values": [int(value) for value in result.reshape(-1).tolist()],
        }

    artifact_paths = [Path(path) for path in INTEGER_ARTIFACTS] + [args.runner]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in artifact_paths}
    native_command = [str(args.runner.resolve()), str(input_path.resolve())]
    compile_command = [
        "nim", "c", "-d:release", "--path:src", f"--out:{args.runner}",
        "benchmarks/pytorch_head_to_head/integer_scan_runner.nim",
    ]

    def native() -> dict[str, Any]:
        """Preserve child output before exact shape, dtype and value checks."""
        result = subprocess.run(native_command, check=True, capture_output=True,
                                text=True, timeout=120)
        report = json.loads(result.stdout)
        for name in INTEGER_CASES:
            actual = report[name]["observation"]
            if actual["shape"] != expected[name]["shape"]:
                raise RuntimeError(f"{name}: shape mismatch")
            if actual["dtype"] != expected[name]["dtype"]:
                raise RuntimeError(f"{name}: dtype mismatch")
            if any(type(value) is not int for value in actual["values"]):
                raise RuntimeError(f"{name}: output values must be JSON integers")
            if actual["values"] != expected[name]["values"]:
                raise RuntimeError(f"{name}: integer values mismatch")
        return report

    def reference() -> dict[str, Any]:
        """Match native warmup, samples and allocating calls on the CPU."""
        report = {}
        for name, call in callables.items():
            for _ in range(manifest["warmup"]):
                call()
            samples = []
            for _ in range(manifest["repeats"]):
                start = time.perf_counter_ns()
                for _ in range(manifest["iterations"]):
                    call()
                samples.append((time.perf_counter_ns() - start) / 1e6
                               / manifest["iterations"])
            report[name] = {
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p95_ms": sorted(samples)[math.ceil(len(samples) * .95) - 1],
            }
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
        for name in INTEGER_CASES:
            for metric in ("median_ms", "p95_ms"):
                if native_result[name][metric] >= torch_result[name][metric]:
                    failures.append(f"round {index + 1}: {name} {metric}")
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]
           for path in artifact_paths):
        raise RuntimeError("Sources or native artifacts changed during measurement")
    report = {
        "torch": torch.__version__,
        "torch_git_revision": torch.version.git_version,
        "python": sys.version,
        "backend": "cpu",
        "dtype": args.dtype,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
        "artifacts_sha256": hashes,
        "commands": {"compile": compile_command, "invoke": native_command},
        "rounds": rounds,
        "failures": failures,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"failures": failures, "results": str(args.output / "results.json")}))
    return int(bool(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "mps"), default=None)
    parser.add_argument("--dtype", choices=("float32", *INTEGER_DTYPES), default="float32")
    parser.add_argument("--runner", type=Path, help="defaults by dtype and backend")
    parser.add_argument("--output", type=Path, default=Path("build/reduction_benchmark"))
    args = parser.parse_args()
    if args.dtype == "float32":
        args.backend = args.backend or "mps"
    else:
        if args.backend not in (None, "cpu"):
            parser.error("integer and bool dtypes require --backend cpu")
        args.backend = "cpu"
    if args.runner is None:
        if args.dtype == "float32":
            args.runner = Path("build/reduction_runner_cpu" if args.backend == "cpu"
                               else "build/reduction_runner")
        else:
            args.runner = Path("build/integer_scan_runner")
    if sys.version_info[:3] != (3, 13, 18) or torch.__version__.split("+")[0] != "2.15.0":
        raise RuntimeError("Requires pinned CPython 3.13.18 and PyTorch 2.15.0")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dtype != "float32":
        return run_integer_benchmark(args)
    artifact_paths = ARTIFACTS + (METAL_ARTIFACTS if args.backend == "mps" else ())
    artifacts = [Path(path) for path in artifact_paths] + [args.runner]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    rows, columns = 128, 128
    # Nonzero, nonconstant products near one avoid meaningless underflow comparisons.
    source = (1 + ((torch.arange(rows * columns) % 17) - 8) * 0.0001).float().reshape(rows, columns)
    manifest = {"rows": rows, "columns": columns, "values": source.flatten().tolist(),
                "warmup": 50, "repeats": 100, "iterations": 50}
    input_path = args.output / "input.json"
    input_path.write_text(json.dumps(manifest) + "\n")
    resident = source.to(args.backend)
    synchronize = torch.mps.synchronize if args.backend == "mps" else lambda: None
    callables = {name: operation(name, resident) for name in CASES}
    expected: dict[str, Any] = {}
    for name in CASES:
        cpu_result = operation(name, source)()
        if args.backend == "mps":
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
            synchronize()
            samples = []
            for _ in range(manifest["repeats"]):
                synchronize()
                start = time.perf_counter_ns()
                for _ in range(manifest["iterations"]):
                    result = call()
                synchronize()
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
    report = {"torch": torch.__version__, "python": sys.version, "backend": args.backend,
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
              "artifacts_sha256": hashes,
              "rounds": rounds, "failures": failures}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"failures": failures, "results": str(args.output / "results.json")}))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
