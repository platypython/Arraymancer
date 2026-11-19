#!/usr/bin/env python3
"""Benchmark allocating CPU min/max-with-indices against pinned PyTorch 2.15.0.

Build from the repository root with:
  nim c -d:release --path:src --out:build/paired_extrema_runner benchmarks/pytorch_head_to_head/paired_extrema_runner.nim
Run all 224 mandatory actions with:
  python benchmarks/pytorch_head_to_head/benchmark_paired_extrema.py
Each action gets 50 warmups and 100 samples of 50 allocating calls in three
alternating backend-order pairs. Every median and p95 must strictly beat
PyTorch in every pair (1,344 comparisons). Inputs, methods, and axes are
resident and bound before timing; every call returns both values and indices.
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

DTYPES = ("float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool")
TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32, "float64": torch.float64,
    "int8": torch.int8, "int16": torch.int16, "int32": torch.int32,
    "int64": torch.int64, "uint8": torch.uint8, "bool": torch.bool,
}
OPS = ("min", "max")
KEEPDIMS = (False, True)
PROFILES = (
    "rows", "rows_head", "rows_tail", "columns",
    "cube_axis0", "cube_axis1", "cube_axis2",
)
ACTIONS = tuple(
    f"{op}_{'keepdim' if keepdim else 'flat'}_{profile}"
    for op in OPS for keepdim in KEEPDIMS for profile in PROFILES
)
TORCH_VERSION = "2.15.0"
TORCH_GIT_REVISION = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
PYTHON_VERSION = (3, 13, 18)
ROWS, COLUMNS, CUBE_SHAPE = 128, 128, (16, 32, 32)
WARMUP, REPEATS, ITERATIONS, PAIRED_ROUNDS = 50, 100, 50, 3
ROOT = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/paired_extrema_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_paired_extrema.py")
SOURCE_PATHS = tuple(map(Path, (
    "src/arraymancer/tensor/private/p_argextrema.nim",
    "src/arraymancer/tensor/private/p_extrema.nim",
    "src/arraymancer/tensor/private/p_accelerate.nim",
    "src/arraymancer/tensor/aggregate.nim",
    "src/arraymancer.nim",
    "nim.cfg",
    "tests/parity/reference.json",
    str(RUNNER_SOURCE), str(SCRIPT_SOURCE),
)))
MANIFEST_FIELDS = (
    "input_dtype", "values_dtype", "indices_dtype", "rows", "columns",
    "shape", "values", "cube_shape", "cube_values", "rows_head_values",
    "rows_tail_values", "actions", "warmup", "repeats", "iterations",
)


def fence_values(dtype: str) -> tuple[int | float, int | float]:
    if dtype in ("float32", "float64"):
        return -1024.0, 1024.0
    limits = torch.iinfo(TORCH_DTYPES[dtype])
    return limits.min, limits.max


def make_sources(dtype: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create resident base/head/tail matrices; cube is a view of base."""
    count = ROWS * COLUMNS
    if dtype == "bool":
        matrix = (torch.arange(count, dtype=torch.int64) % 3 != 0).reshape(ROWS, COLUMNS)
        rows_head = torch.ones((ROWS, COLUMNS), dtype=torch.bool)
        rows_head[:, -1] = False
        rows_tail = torch.zeros((ROWS, COLUMNS), dtype=torch.bool)
        rows_tail[:, -1] = True
    else:
        matrix = (((torch.arange(count, dtype=torch.int64) % 63) + 1)
                  .reshape(ROWS, COLUMNS).to(dtype=TORCH_DTYPES[dtype]))
        low, high = fence_values(dtype)
        rows_head = matrix.clone()
        rows_tail = matrix.clone()
        rows_head[:, 0], rows_head[:, -1] = high, low
        rows_tail[:, 0], rows_tail[:, -1] = low, high

    cube = matrix.reshape(CUBE_SHAPE)
    if (len({matrix.data_ptr(), rows_head.data_ptr(), rows_tail.data_ptr()}) != 3 or
            cube.data_ptr() != matrix.data_ptr()):
        raise RuntimeError("paired-extrema inputs must be three resident tensors with a base-sharing cube view")
    flat_values = matrix.reshape(-1).tolist()
    payload: dict[str, Any] = {
        "rows": ROWS,
        "columns": COLUMNS,
        "shape": [ROWS, COLUMNS],
        "values": flat_values,
        "cube_shape": list(CUBE_SHAPE),
        "cube_values": flat_values,
        "rows_head_values": rows_head.reshape(-1).tolist(),
        "rows_tail_values": rows_tail.reshape(-1).tolist(),
    }
    return {
        "base": matrix,
        "rows_head": rows_head,
        "rows_tail": rows_tail,
        "cube": cube,
    }, payload


def action_spec(name: str) -> tuple[str, bool, str, int]:
    op, mode, profile = name.split("_", 2)
    axis = {
        "rows": 1, "rows_head": 1, "rows_tail": 1, "columns": 0,
        "cube_axis0": 0, "cube_axis1": 1, "cube_axis2": 2,
    }[profile]
    return op, mode == "keepdim", profile, axis


def bind_actions(sources: dict[str, torch.Tensor]) -> dict[str, Callable[[], Any]]:
    """Bind real torch.min/max dim calls, tensors, axes, and keepdim upfront."""
    bound: dict[str, Callable[[], Any]] = {}
    for name in ACTIONS:
        op, keepdim, profile, axis = action_spec(name)
        value = sources[profile] if profile in ("rows_head", "rows_tail") else (
            sources["cube"] if profile.startswith("cube_") else sources["base"]
        )
        method = getattr(torch, op)
        bound[name] = lambda method=method, value=value, axis=axis, keepdim=keepdim: method(
            value, dim=axis, keepdim=keepdim)
    return bound


def retain_native_failure(args: argparse.Namespace, dtype: str, index: int,
                          returncode: int | None, stdout: str, stderr: str,
                          error: str) -> Path:
    path = args.output / f"failed_native_{dtype}_{index}.json"
    path.write_text(json.dumps({
        "input_dtype": dtype,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _expected_shape(name: str, result: Any) -> list[int]:
    _, keepdim, _, _ = action_spec(name)
    shape = list(result.values.shape)
    if keepdim and not shape:
        raise ValueError(f"{name}: benchmark unexpectedly produced scalar output")
    return shape


def _validate_observation(name: str, observation: Any, reference: Any,
                          input_dtype: str) -> None:
    fields = {
        "input_dtype", "values_dtype", "indices_dtype", "values_shape",
        "indices_shape", "values", "indices", "values_indices_distinct",
    }
    if type(observation) is not dict or set(observation) != fields:
        raise ValueError(f"{name}: malformed paired-output observation fields")
    expected_shape = _expected_shape(name, reference)
    values_shape, indices_shape = observation["values_shape"], observation["indices_shape"]
    if (type(values_shape) is not list or type(indices_shape) is not list or
            any(type(dimension) is not int for dimension in values_shape + indices_shape) or
            observation["input_dtype"] != input_dtype or
            observation["values_dtype"] != input_dtype or
            observation["indices_dtype"] != "int64" or
            values_shape != expected_shape or indices_shape != expected_shape or
            observation["values_indices_distinct"] is not True):
        raise ValueError(f"{name}: input/output dtype, shape, or independent-output mismatch")

    values, indices = observation["values"], observation["indices"]
    expected_values = reference.values.reshape(-1).tolist()
    expected_indices = reference.indices.reshape(-1).tolist()
    if (type(values) is not list or type(indices) is not list or
            len(values) != len(expected_values) or len(indices) != len(expected_indices)):
        raise ValueError(f"{name}: paired output value counts do not match the exact shapes")
    op, _, profile, _ = action_spec(name)
    reduced_length = {
        "rows": COLUMNS, "rows_head": COLUMNS, "rows_tail": COLUMNS,
        "columns": ROWS, "cube_axis0": CUBE_SHAPE[0],
        "cube_axis1": CUBE_SHAPE[1], "cube_axis2": CUBE_SHAPE[2],
    }[profile]
    if any(type(index) is not int or not 0 <= index < reduced_length for index in indices):
        raise ValueError(f"{name}: indices are not exact in-range int64 candidates")
    if input_dtype == "bool":
        if any(type(value) is not bool for value in values):
            raise ValueError(f"{name}: bool values must remain exact JSON booleans")
    elif input_dtype in ("int8", "int16", "int32", "int64", "uint8"):
        if any(type(value) is not int for value in values):
            raise ValueError(f"{name}: integer values must remain exact JSON integers")
    elif any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError(f"{name}: floating values must be finite JSON floats")
    if values != expected_values or indices != expected_indices:
        raise ValueError(f"{name}: exact PyTorch paired outputs mismatch")
    if op not in OPS:
        raise ValueError(f"{name}: unexpected operation")


def run_group(args: argparse.Namespace, dtype: str, failures: list[str]) -> dict[str, Any]:
    sources, payload = make_sources(dtype)
    callables = bind_actions(sources)
    manifest = {
        "input_dtype": dtype,
        "values_dtype": dtype,
        "indices_dtype": "int64",
        **payload,
        "actions": list(ACTIONS),
        "warmup": WARMUP,
        "repeats": REPEATS,
        "iterations": ITERATIONS,
    }
    input_path = args.output / f"input_{dtype}.json"
    input_path.write_text(json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    expected = {name: callables[name]() for name in ACTIONS}
    for name, output in expected.items():
        if output.values.dtype != TORCH_DTYPES[dtype] or output.indices.dtype != torch.int64:
            raise RuntimeError(f"{name}: PyTorch must return values in {dtype} and int64 indices")

    native_runs = 0

    def native() -> dict[str, Any]:
        nonlocal native_runs
        index = native_runs
        native_runs += 1
        stem = args.output / f"native_{dtype}_{index}"
        try:
            completed = subprocess.run(
                [str(args.runner), str(input_path.resolve())],
                check=False, capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired as error:
            stdout, stderr = _text(error.stdout), _text(error.stderr)
            path = retain_native_failure(
                args, dtype, index, None, stdout, stderr, "native runner timed out")
            raise RuntimeError(f"{dtype}: native timeout evidence retained at {path}") from error
        except OSError as error:
            path = retain_native_failure(args, dtype, index, None, "", "", repr(error))
            raise RuntimeError(f"{dtype}: native invocation evidence retained at {path}") from error

        stdout, stderr = completed.stdout, completed.stderr
        (stem.with_suffix(".stdout.txt")).write_text(stdout, encoding="utf-8")
        (stem.with_suffix(".stderr.txt")).write_text(stderr, encoding="utf-8")
        try:
            if completed.returncode != 0:
                raise ValueError(f"native runner exited with status {completed.returncode}")
            report = json.loads(stdout)
            (stem.with_suffix(".json")).write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if type(report) is not dict or set(report) != set(ACTIONS):
                raise ValueError("native action set differs from strict manifest")
            for name in ACTIONS:
                entry = report[name]
                if type(entry) is not dict or set(entry) != {
                        "observation", "samples_ms", "median_ms", "p95_ms"}:
                    raise ValueError(f"{name}: malformed timing record")
                _validate_observation(name, entry["observation"], expected[name], dtype)
                samples = entry["samples_ms"]
                if (type(samples) is not list or len(samples) != REPEATS or
                        any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                            for value in samples)):
                    raise ValueError(f"{name}: invalid native samples")
                ordered = sorted(samples)
                median = statistics.median(samples)
                p95 = ordered[math.ceil(0.95 * REPEATS) - 1]
                if (type(entry["median_ms"]) not in (int, float) or
                        type(entry["p95_ms"]) not in (int, float) or
                        entry["median_ms"] != median or entry["p95_ms"] != p95):
                    raise ValueError(f"{name}: reported timing statistics mismatch samples")
            return report
        except (KeyError, TypeError, ValueError, AssertionError) as error:
            path = retain_native_failure(
                args, dtype, index, completed.returncode, stdout, stderr, str(error))
            raise RuntimeError(f"{dtype}: invalid native report retained at {path}: {error}") from error

    def reference() -> dict[str, Any]:
        report: dict[str, Any] = {}
        for name in ACTIONS:
            call = callables[name]
            for _ in range(WARMUP):
                result = call()
            samples = []
            for _ in range(REPEATS):
                start = time.perf_counter_ns()
                for _ in range(ITERATIONS):
                    result = call()
                samples.append((time.perf_counter_ns() - start) / 1e6 / ITERATIONS)
                del result
            ordered = sorted(samples)
            report[name] = {
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p95_ms": ordered[math.ceil(0.95 * REPEATS) - 1],
            }
        return report

    native()  # Keep an independent exact observation before any timed round.
    rounds = []
    for index in range(PAIRED_ROUNDS):
        order = "pytorch-first" if index % 2 == 0 else "arraymancer-first"
        if index % 2 == 0:
            pytorch_result, native_result = reference(), native()
        else:
            native_result, pytorch_result = native(), reference()
        rounds.append({
            "round": index + 1,
            "order": order,
            "pytorch": pytorch_result,
            "arraymancer": native_result,
        })
        (args.output / f"rounds_{dtype}.json").write_text(
            json.dumps(rounds, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in ACTIONS:
            for metric in ("median_ms", "p95_ms"):
                if native_result[name][metric] >= pytorch_result[name][metric]:
                    failures.append(f"{dtype}: round {index + 1} {name} {metric} did not win")
    return {
        "input_dtype": dtype,
        "values_dtype": dtype,
        "indices_dtype": "int64",
        "actions": list(ACTIONS),
        "input_sha256": input_hash,
        "correctness": "passed",
        "rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/paired_extrema_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/paired_extrema_benchmark"))
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
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest()
              for label, path in artifacts.items()}
    compile_command = [
        "nim", "c", "-d:release", "--path:src", f"--out:{args.runner}",
        str(ROOT / RUNNER_SOURCE),
    ]
    nim_version = subprocess.run(
        ["nim", "--version"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    failures: list[str] = []
    groups: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for dtype in DTYPES:
        try:
            group = run_group(args, dtype, failures)
            groups.append(group)
            input_hashes[dtype] = group["input_sha256"]
        except RuntimeError as error:
            failures.append(str(error))
            failed_input = args.output / f"input_{dtype}.json"
            if failed_input.is_file():
                input_hashes[dtype] = hashlib.sha256(failed_input.read_bytes()).hexdigest()
            rounds_path = args.output / f"rounds_{dtype}.json"
            failed_rounds = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.is_file() else []
            path = args.output / f"group_failure_{dtype}.json"
            path.write_text(json.dumps({
                "input_dtype": dtype,
                "values_dtype": dtype,
                "indices_dtype": "int64",
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "error": str(error),
                "rounds": failed_rounds,
            }, indent=2) + "\n", encoding="utf-8")
            groups.append({
                "input_dtype": dtype,
                "values_dtype": dtype,
                "indices_dtype": "int64",
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "correctness": "failed",
                "error": str(error),
                "rounds": failed_rounds,
            })
    changed = [label for label, path in artifacts.items()
               if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[label]]
    if changed:
        failures.append("Sources or runner changed during measurement: " + ", ".join(changed))
    for dtype, digest in input_hashes.items():
        path = args.output / f"input_{dtype}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(f"Input manifest changed during measurement: {dtype}")
    action_count = len(groups) * len(ACTIONS)
    completed_action_count = sum(
        len(group["actions"]) for group in groups if group.get("correctness") == "passed")
    comparison_count = sum(
        len(group["actions"]) * len(group.get("rounds", [])) * 2 for group in groups)
    if (len(groups) != len(DTYPES) or action_count != 224 or completed_action_count != 224
            or comparison_count != 1344):
        failures.append("paired-extrema protocol requires 8 groups, 224 completed actions and 1,344 comparisons")
    report = {
        "torch": torch.__version__,
        "torch_git_revision": torch.version.git_version,
        "python": sys.version,
        "backend": "cpu",
        "runtime": {
            "nim_compiler": nim_version,
            "python": sys.version,
            "platform": sys.platform,
            "machine": __import__("platform").machine(),
        },
        "protocol": {
            "input_dtypes": list(DTYPES),
            "values_dtype": "same as input_dtype",
            "indices_dtype": "int64",
            "operations": list(OPS),
            "keepdim_values": list(KEEPDIMS),
            "profiles": list(PROFILES),
            "actions_per_group": list(ACTIONS),
            "group_count": len(groups),
            "action_count": action_count,
            "completed_action_count": completed_action_count,
            "warmup_calls_per_action": WARMUP,
            "samples_per_action_per_round": REPEATS,
            "allocating_calls_per_sample": ITERATIONS,
            "paired_rounds_per_group": PAIRED_ROUNDS,
            "comparisons": comparison_count,
            "request_schema": {
                "fields": list(MANIFEST_FIELDS),
                "integer_values": "JSON integers; preserve exact signed boundaries",
                "boolean_values": "JSON booleans",
                "cube_values": "required and equal to base values",
            },
            "response_schema": {
                "actions": list(ACTIONS),
                "entry_fields": ["observation", "samples_ms", "median_ms", "p95_ms"],
                "observation_fields": [
                    "input_dtype", "values_dtype", "indices_dtype", "values_shape",
                    "indices_shape", "values", "indices", "values_indices_distinct",
                ],
                "integer_values": "exact JSON integers",
                "boolean_values": "JSON booleans",
                "indices": "exact JSON int64 values",
            },
        },
        "source_sha256": {label: digest for label, digest in hashes.items()
                           if label != "native_runner"},
        "runner_sha256": hashes["native_runner"],
        "input_sha256": input_hashes,
        "commands": {
            "build_cwd": str(ROOT),
            "build_command": compile_command,
            "compiler_version": nim_version,
            "invoke_command_template": [str(args.runner), "<input-manifest.json>"],
        },
        "groups": groups,
        "failures": failures,
        "passed": not failures,
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "groups": len(groups),
        "actions": action_count,
        "completed_actions": completed_action_count,
        "comparisons": comparison_count,
        "failures": failures,
        "passed": not failures,
        "results": str(result_path),
    }))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
