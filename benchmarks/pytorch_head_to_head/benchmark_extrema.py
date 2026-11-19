#!/usr/bin/env python3
"""Benchmark CPU amin/amax against pinned PyTorch 2.15.0.

Build from the repository root with:
  nim c -d:release --path:src --out:build/extrema_runner benchmarks/pytorch_head_to_head/extrema_runner.nim
Run all 160 mandatory actions with:
  python benchmarks/pytorch_head_to_head/benchmark_extrema.py
Optional --runner and --output paths select the native executable and report directory.
Each action gets 50 warmups and 100 samples of 50 allocating calls, in three
alternating backend-order pairs. All 160 actions must beat PyTorch in median and
p95 in every paired round (960 comparisons). Inputs are finite, deterministic,
and already resident; methods and axes are bound before timing.
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
import torch

DTYPES = ("bool", "int8", "int16", "int32", "int64", "uint8", "float32", "float64")
TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool, "int8": torch.int8, "int16": torch.int16,
    "int32": torch.int32, "int64": torch.int64, "uint8": torch.uint8,
    "float32": torch.float32, "float64": torch.float64,
}
OPS = ("amin", "amax")
KEEPDIMS = (False, True)
CASES = ("whole", "rows", "columns", "axes_0_2", "axes_2_1")
ACTIONS = tuple(
    f"{op}_{'keepdim' if keepdim else 'flat'}_{case}"
    for op in OPS for keepdim in KEEPDIMS for case in CASES
)
TORCH_VERSION = "2.15.0"
TORCH_GIT_REVISION = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
PYTHON_VERSION = (3, 13, 18)
ROWS, COLUMNS = 128, 128
WARMUP, REPEATS, ITERATIONS, PAIRED_ROUNDS = 50, 100, 50, 3
ROOT = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/extrema_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_extrema.py")
SOURCE_PATHS = (
    Path("src/arraymancer/tensor/aggregate.nim"),
    Path("src/arraymancer/tensor/private/p_accelerate.nim"),
    Path("src/arraymancer/tensor/private/p_extrema.nim"),
    Path("tests/parity/reference.json"), Path("nim.cfg"),
    RUNNER_SOURCE, SCRIPT_SOURCE,
)


def make_sources(dtype: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create resident 2D and 3D inputs with exactly representable bounded values."""
    count = ROWS * COLUMNS
    if dtype == "bool":
        raw = torch.arange(count, dtype=torch.int64) % 2
    elif dtype in ("float32", "float64"):
        raw = ((torch.arange(count, dtype=torch.int64) % 31) - 15).to(torch.float64)
        raw = raw + ((torch.arange(count, dtype=torch.int64) % 7).to(torch.float64) / 8)
    else:
        raw = torch.arange(count, dtype=torch.int64) % 31 - 15
    matrix = raw.reshape(ROWS, COLUMNS).to(dtype=TORCH_DTYPES[dtype])
    cube = matrix.reshape(16, 32, 32)
    values = matrix.flatten().tolist()
    payload: dict[str, Any] = {
        "rows": ROWS, "columns": COLUMNS, "shape": [ROWS, COLUMNS],
        "values": values,
    }
    return {"matrix": matrix, "cube": cube}, payload


def bind_actions(sources: dict[str, torch.Tensor]) -> dict[str, Callable[[], torch.Tensor]]:
    bound: dict[str, Callable[[], torch.Tensor]] = {}
    for op in OPS:
        method = getattr(torch, op)
        for keepdim in KEEPDIMS:
            for case in CASES:
                value = sources["cube"] if case in ("axes_0_2", "axes_2_1") else sources["matrix"]
                dim: int | tuple[int, ...] | None = {
                    "whole": None, "rows": 1, "columns": 0,
                    "axes_0_2": (0, 2), "axes_2_1": (2, 1),
                }[case]
                bound[f"{op}_{'keepdim' if keepdim else 'flat'}_{case}"] = (
                    lambda method=method, value=value, dim=dim, keepdim=keepdim:
                    method(value, dim=dim, keepdim=keepdim)
                )
    return bound


def run_group(args: argparse.Namespace, dtype: str, failures: list[str]) -> dict[str, Any]:
    sources, payload = make_sources(dtype)
    group = dtype
    manifest = {
        "dtype": dtype, "output_dtype": dtype, **payload,
        "actions": list(ACTIONS), "warmup": WARMUP,
        "repeats": REPEATS, "iterations": ITERATIONS,
    }
    input_path = args.output / f"input_{group}.json"
    input_path.write_text(json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    callables = bind_actions(sources)
    expected = {name: callables[name]() for name in ACTIONS}
    native_runs = 0

    def native() -> dict[str, Any]:
        nonlocal native_runs
        completed = subprocess.run(
            [str(args.runner), str(input_path.resolve())], check=True,
            capture_output=True, text=True, timeout=120,
        )
        try:
            report = json.loads(completed.stdout)
            if type(report) is not dict or set(report) != set(ACTIONS):
                raise ValueError("native action set differs from strict manifest")
            raw_path = args.output / f"native_{group}_{native_runs}.json"
            raw_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            native_runs += 1
            for name in ACTIONS:
                entry = report[name]
                if type(entry) is not dict or set(entry) != {
                    "observation", "samples_ms", "median_ms", "p95_ms",
                }:
                    raise ValueError(f"{name}: malformed timing record")
                actual = entry["observation"]
                reference = expected[name]
                if (type(actual) is not dict or set(actual) != {"input_dtype", "dtype", "shape", "values"}
                        or actual["input_dtype"] != dtype
                        or actual["dtype"] != dtype
                        or actual["shape"] != list(reference.shape)
                        or type(actual["values"]) is not list
                        or len(actual["values"]) != reference.numel()):
                    raise ValueError(f"{name}: malformed or mismatched output observation")
                expected_values = reference.reshape(-1).tolist()
                values = actual["values"]
                if dtype in ("bool", "int8", "int16", "int32", "int64", "uint8"):
                    if dtype == "bool":
                        if any(type(v) is not bool for v in values):
                            raise ValueError(f"{name}: bool output is not JSON boolean")
                        converted = values
                    else:
                        if any(type(v) is not str for v in values):
                            raise ValueError(f"{name}: integer output must be exact decimal strings")
                        converted = [int(v, 10) for v in values]
                    if converted != expected_values:
                        raise ValueError(f"{name}: exact output values mismatch")
                else:
                    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                        raise ValueError(f"{name}: float output must contain finite JSON numbers")
                    actual_tensor = torch.tensor(values, dtype=TORCH_DTYPES[dtype]).reshape(actual["shape"])
                    torch.testing.assert_close(actual_tensor, reference, atol=0, rtol=0, equal_nan=False)
                    if any(
                        expected_value == 0 and math.copysign(1.0, float(actual_value)) !=
                        math.copysign(1.0, float(expected_value))
                        for actual_value, expected_value in zip(values, expected_values)
                    ):
                        raise ValueError(f"{name}: signed-zero output mismatch")
                samples = entry["samples_ms"]
                if (type(samples) is not list or len(samples) != REPEATS or
                        any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples)):
                    raise ValueError(f"{name}: invalid native samples")
                ordered = sorted(samples)
                median = statistics.median(samples)
                p95 = ordered[math.ceil(0.95 * REPEATS) - 1]
                if entry["median_ms"] != median or entry["p95_ms"] != p95:
                    raise ValueError(f"{name}: reported timing statistics mismatch samples")
            return report
        except (KeyError, TypeError, ValueError, AssertionError) as error:
            failed_path = args.output / f"failed_native_{group}_{native_runs}.json"
            failed_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
            raise RuntimeError(f"{group}: invalid native report retained at {failed_path}: {error}") from error

    def reference() -> dict[str, Any]:
        report: dict[str, Any] = {}
        for name in ACTIONS:
            call = callables[name]
            for _ in range(WARMUP):
                call()
            samples = []
            for _ in range(REPEATS):
                result: torch.Tensor | None = None
                start = time.perf_counter_ns()
                for _ in range(ITERATIONS):
                    result = call()
                samples.append((time.perf_counter_ns() - start) / 1e6 / ITERATIONS)
                del result
            ordered = sorted(samples)
            report[name] = {
                "samples_ms": samples, "median_ms": statistics.median(samples),
                "p95_ms": ordered[math.ceil(0.95 * REPEATS) - 1],
            }
        return report

    native()  # Independent correctness observation before timing.
    rounds = []
    for index in range(PAIRED_ROUNDS):
        order = "pytorch-first" if index % 2 == 0 else "arraymancer-first"
        if index % 2 == 0:
            pytorch_result, native_result = reference(), native()
        else:
            native_result, pytorch_result = native(), reference()
        rounds.append({"round": index + 1, "order": order, "pytorch": pytorch_result,
                       "arraymancer": native_result})
        (args.output / f"rounds_{group}.json").write_text(
            json.dumps(rounds, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in ACTIONS:
            for metric in ("median_ms", "p95_ms"):
                if native_result[name][metric] >= pytorch_result[name][metric]:
                    failures.append(f"{group}: round {index + 1} {name} {metric} did not win")
    return {"input_dtype": dtype, "output_dtype": dtype, "actions": list(ACTIONS),
            "input_sha256": input_hash, "correctness": "passed", "rounds": rounds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/extrema_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/extrema_benchmark"))
    args = parser.parse_args()
    args.runner = (ROOT / args.runner).resolve() if not args.runner.is_absolute() else args.runner
    args.output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    if sys.version_info[:3] != PYTHON_VERSION:
        raise RuntimeError("Requires pinned CPython 3.13.18")
    if (torch.__version__.split("+")[0] != TORCH_VERSION or
            torch.version.git_version != TORCH_GIT_REVISION):
        raise RuntimeError("Requires pinned PyTorch 2.15.0 wheel revision " + TORCH_GIT_REVISION)
    if not args.runner.is_file():
        raise RuntimeError(f"Native runner not found: {args.runner}; build it using the docstring command")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = {str(path): ROOT / path for path in SOURCE_PATHS}
    artifacts["native_runner"] = args.runner
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in artifacts.items()}
    compile_command = ["nim", "c", "-d:release", "--path:src", f"--out:{args.runner}", str(ROOT / RUNNER_SOURCE)]
    failures: list[str] = []
    input_hashes: dict[str, str] = {}
    groups = []
    for dtype in DTYPES:
        group = run_group(args, dtype, failures)
        groups.append(group)
        input_hashes[dtype] = group["input_sha256"]
    changed = [label for label, path in artifacts.items()
               if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[label]]
    if changed:
        raise RuntimeError("Sources or runner changed during measurement: " + ", ".join(changed))
    for dtype, digest in input_hashes.items():
        path = args.output / f"input_{dtype}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Input manifest changed during measurement: {dtype}")
    action_count = sum(len(group["actions"]) for group in groups)
    comparison_count = action_count * PAIRED_ROUNDS * 2
    if len(groups) != 8 or action_count != 160 or comparison_count != 960:
        failures.append("extrema protocol requires 8 groups, 160 actions and 960 comparisons")
    report = {
        "torch": torch.__version__, "torch_git_revision": torch.version.git_version,
        "python": sys.version, "backend": "cpu",
        "protocol": {"input_dtypes": list(DTYPES), "output_dtypes": list(DTYPES),
                     "actions_per_group": list(ACTIONS), "group_count": len(groups),
                     "action_count": action_count, "warmup_calls_per_action": WARMUP,
                     "samples_per_action_per_round": REPEATS,
                     "allocating_calls_per_sample": ITERATIONS,
                     "paired_rounds_per_group": PAIRED_ROUNDS,
                     "comparisons": comparison_count},
        "source_sha256": {label: digest for label, digest in hashes.items() if label != "native_runner"},
        "runner_sha256": hashes["native_runner"], "input_sha256": input_hashes,
        "commands": {"build_cwd": str(ROOT), "build_example": compile_command,
                     "invoke_example": [str(args.runner), "<input-manifest.json>"]},
        "groups": groups, "failures": failures, "passed": not failures,
    }
    results_path = args.output / "results.json"
    results_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"groups": len(groups), "actions": action_count,
                      "comparisons": comparison_count, "failures": failures,
                      "passed": not failures, "results": str(results_path)}))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
