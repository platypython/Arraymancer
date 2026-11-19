#!/usr/bin/env python3
"""Benchmark paired CPU min/max-with-indices autograd against pinned PyTorch.

Build from the repository root with:
  nim c -d:release --path:src --out:build/paired_extrema_gradient_runner benchmarks/pytorch_head_to_head/paired_extrema_gradient_runner.nim
Run all 56 mandatory actions with:
  python benchmarks/pytorch_head_to_head/benchmark_paired_extrema_gradient.py
Each action gets 50 warmups and 100 samples of 50 allocating training calls in
three alternating backend-order pairs. Every median and p95 must strictly beat
PyTorch in every pair (336 comparisons). Each call creates a fresh leaf and
backpropagates a resident finite, nonzero seed through the actual paired
min/max-with-indices result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

os.environ.update({name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
import torch

DTYPES = ("float32", "float64")
TORCH_DTYPES: dict[str, torch.dtype] = {"float32": torch.float32, "float64": torch.float64}
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
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/paired_extrema_gradient_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_paired_extrema_gradient.py")
SOURCE_PATHS = tuple(map(Path, (
    "src/arraymancer/autograd/gates_reduce.nim",
    "src/arraymancer/autograd/autograd_common.nim",
    "src/arraymancer/autograd/private/p_extrema_gradient.nim",
    "src/arraymancer/tensor/aggregate.nim",
    "src/arraymancer/tensor/private/p_argextrema.nim",
    "src/arraymancer/tensor/private/p_extrema.nim",
    "src/arraymancer/tensor/private/p_accelerate.nim",
    "nim.cfg", "tests/parity/reference.json", str(RUNNER_SOURCE), str(SCRIPT_SOURCE),
)))
MANIFEST_FIELDS = (
    "input_dtype", "values_dtype", "indices_dtype", "rows", "columns",
    "shape", "values", "cube_shape", "cube_values", "rows_head_values",
    "rows_tail_values", "actions", "seeds", "warmup", "repeats", "iterations",
)
OBSERVATION_FIELDS = (
    "input_dtype", "values_dtype", "indices_dtype", "gradient_dtype",
    "values_shape", "indices_shape", "gradient_shape", "values", "indices",
    "gradient", "values_indices_distinct", "values_gradient_distinct",
    "indices_gradient_distinct", "values_input_distinct", "indices_input_distinct",
    "gradient_input_distinct",
)


def make_sources(dtype: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create exactly three resident matrices; cube is a view of the base."""
    count = ROWS * COLUMNS
    matrix = (((torch.arange(count, dtype=torch.int64) % 63) + 1)
              .reshape(ROWS, COLUMNS).to(dtype=TORCH_DTYPES[dtype]))
    rows_head = matrix.clone()
    rows_tail = matrix.clone()
    rows_head[:, 0], rows_head[:, -1] = 1024, -1024
    rows_tail[:, 0], rows_tail[:, -1] = -1024, 1024
    cube = matrix.reshape(CUBE_SHAPE)
    if (len({matrix.data_ptr(), rows_head.data_ptr(), rows_tail.data_ptr()}) != 3 or
            cube.data_ptr() != matrix.data_ptr()):
        raise RuntimeError("paired-extrema inputs must be three resident matrices with a base-sharing cube view")
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


def output_shape(input_shape: tuple[int, ...], axis: int, keepdim: bool) -> tuple[int, ...]:
    return tuple(1 if index == axis else size for index, size in enumerate(input_shape)) if keepdim else tuple(
        size for index, size in enumerate(input_shape) if index != axis)


def seed_for(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    count = math.prod(shape)
    values = [index % 7 - 3 + 0.25 for index in range(count)]
    if any(not math.isfinite(value) or value == 0 for value in values):
        raise RuntimeError("benchmark seeds must be finite and nonzero")
    return torch.tensor(values, dtype=dtype).reshape(shape)


def _train(base: torch.Tensor, seed: torch.Tensor,
           method: Callable[..., Any], axis: int, keepdim: bool
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run actual paired reduction and seeded backward on a fresh leaf."""
    leaf = base.detach().requires_grad_(True)
    pair = method(leaf, dim=axis, keepdim=keepdim)
    pair.values.backward(seed)
    if leaf.grad is None:
        raise RuntimeError("PyTorch paired-extrema call produced no leaf gradient")
    return pair.values, pair.indices, leaf.grad, leaf


def bind_actions(sources: dict[str, torch.Tensor], dtype: str
                 ) -> tuple[dict[str, Callable[[], Any]], dict[str, Any]]:
    """Bind each real torch.min/max call and matching resident seed upfront."""
    callables: dict[str, Callable[[], Any]] = {}
    seeds: dict[str, Any] = {}
    for name in ACTIONS:
        op, keepdim, profile, axis = action_spec(name)
        base = sources[profile] if profile in ("rows_head", "rows_tail") else (
            sources["cube"] if profile.startswith("cube_") else sources["base"]
        )
        seed = seed_for(output_shape(tuple(base.shape), axis, keepdim), TORCH_DTYPES[dtype])
        seeds[name] = {
            "dtype": dtype,
            "shape": list(seed.shape),
            "values": seed.reshape(-1).tolist(),
        }
        method = getattr(torch, op)
        callables[name] = lambda base=base, seed=seed, method=method, axis=axis, keepdim=keepdim: _train(
            base, seed, method, axis, keepdim)
    return callables, seeds


def _reference_observation(result: Any, dtype: str) -> dict[str, Any]:
    values, indices, gradient, leaf = result
    return {
        "input_dtype": dtype,
        "values_dtype": str(values.dtype).removeprefix("torch."),
        "indices_dtype": str(indices.dtype).removeprefix("torch."),
        "gradient_dtype": str(gradient.dtype).removeprefix("torch."),
        "values_shape": list(values.shape),
        "indices_shape": list(indices.shape),
        "gradient_shape": list(gradient.shape),
        "values": values.reshape(-1).tolist(),
        "indices": indices.reshape(-1).tolist(),
        "gradient": gradient.reshape(-1).tolist(),
        "values_indices_distinct": values.data_ptr() != indices.data_ptr(),
        "values_gradient_distinct": values.data_ptr() != gradient.data_ptr(),
        "indices_gradient_distinct": indices.data_ptr() != gradient.data_ptr(),
        "values_input_distinct": values.data_ptr() != leaf.data_ptr(),
        "indices_input_distinct": indices.data_ptr() != leaf.data_ptr(),
        "gradient_input_distinct": gradient.data_ptr() != leaf.data_ptr(),
    }


def _validate_observation(name: str, observation: Any, reference: dict[str, Any],
                          input_dtype: str) -> None:
    if type(observation) is not dict or set(observation) != set(OBSERVATION_FIELDS):
        raise ValueError(f"{name}: malformed paired-extrema gradient observation fields")
    op, keepdim, profile, axis = action_spec(name)
    input_shape = CUBE_SHAPE if profile.startswith("cube_") else (ROWS, COLUMNS)
    expected_values_shape = list(output_shape(input_shape, axis, keepdim))
    expected_gradient_shape = list(input_shape)
    shape_fields = (
        ("values_shape", expected_values_shape),
        ("indices_shape", expected_values_shape),
        ("gradient_shape", expected_gradient_shape),
    )
    if (observation["input_dtype"] != input_dtype or
            observation["values_dtype"] != input_dtype or
            observation["indices_dtype"] != "int64" or
            observation["gradient_dtype"] != input_dtype):
        raise ValueError(f"{name}: input/output dtype mismatch")
    for field, expected_shape in shape_fields:
        shape = observation[field]
        if (type(shape) is not list or any(type(size) is not int for size in shape) or
                shape != expected_shape):
            raise ValueError(f"{name}: {field} does not match the exact expected shape")
    for field in OBSERVATION_FIELDS[-6:]:
        if type(observation[field]) is not bool or not observation[field]:
            raise ValueError(f"{name}: {field} must prove independent storage")

    values, indices, gradient = (observation["values"], observation["indices"],
                                 observation["gradient"])
    expected_values, expected_indices, expected_gradient = (
        reference["values"], reference["indices"], reference["gradient"])
    if (type(values) is not list or type(indices) is not list or type(gradient) is not list or
            len(values) != len(expected_values) or len(indices) != len(expected_indices) or
            len(gradient) != len(expected_gradient)):
        raise ValueError(f"{name}: output element counts do not match exact shapes")
    if (any(type(value) is not float or not math.isfinite(value) for value in values) or
            any(type(index) is not int for index in indices) or
            any(type(value) is not float or not math.isfinite(value) for value in gradient)):
        raise ValueError(f"{name}: values/gradient must be finite floats and indices exact integers")
    reduced_length = {
        "rows": COLUMNS, "rows_head": COLUMNS, "rows_tail": COLUMNS,
        "columns": ROWS, "cube_axis0": CUBE_SHAPE[0],
        "cube_axis1": CUBE_SHAPE[1], "cube_axis2": CUBE_SHAPE[2],
    }[profile]
    if any(index < 0 or index >= reduced_length for index in indices):
        raise ValueError(f"{name}: indices are outside the reduced axis")
    if values != expected_values or indices != expected_indices or gradient != expected_gradient:
        raise ValueError(f"{name}: exact PyTorch values, indices, or seeded gradient mismatch")
    if op not in OPS:
        raise ValueError(f"{name}: unexpected operation")


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def retain_native_failure(args: argparse.Namespace, dtype: str, index: int,
                          command: list[str], returncode: int | None,
                          stdout: str, stderr: str, error: str) -> Path:
    path = args.output / f"failed_native_{dtype}_{index}.json"
    path.write_text(json.dumps({
        "input_dtype": dtype,
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_group(args: argparse.Namespace, dtype: str, failures: list[str]) -> dict[str, Any]:
    sources, payload = make_sources(dtype)
    callables, seeds = bind_actions(sources, dtype)
    manifest = {
        "input_dtype": dtype,
        "values_dtype": dtype,
        "indices_dtype": "int64",
        **payload,
        "actions": list(ACTIONS),
        "seeds": seeds,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "iterations": ITERATIONS,
    }
    input_path = args.output / f"input_{dtype}.json"
    input_path.write_text(json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    expected = {name: _reference_observation(callables[name](), dtype) for name in ACTIONS}
    for name in ACTIONS:
        observed = expected[name]
        if (observed["values_dtype"] != dtype or observed["indices_dtype"] != "int64" or
                observed["gradient_dtype"] != dtype):
            raise RuntimeError(f"{name}: PyTorch returned an unexpected paired output dtype")

    native_runs = 0
    native_invocations: list[list[str]] = []

    def native() -> dict[str, Any]:
        nonlocal native_runs
        index = native_runs
        native_runs += 1
        command = [str(args.runner), str(input_path.resolve())]
        native_invocations.append(command)
        stem = args.output / f"native_{dtype}_{index}"
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired as error:
            stdout, stderr = _text(error.stdout), _text(error.stderr)
            path = retain_native_failure(args, dtype, index, command, None,
                                         stdout, stderr, "native runner timed out")
            raise RuntimeError(f"{dtype}: native timeout evidence retained at {path}") from error
        except OSError as error:
            path = retain_native_failure(args, dtype, index, command, None, "", "", repr(error))
            raise RuntimeError(f"{dtype}: native invocation evidence retained at {path}") from error

        stdout, stderr = completed.stdout, completed.stderr
        stem.with_suffix(".stdout.txt").write_text(stdout, encoding="utf-8")
        stem.with_suffix(".stderr.txt").write_text(stderr, encoding="utf-8")
        try:
            if completed.returncode != 0:
                raise ValueError(f"native runner exited with status {completed.returncode}")
            report = json.loads(stdout)
            if type(report) is not dict or set(report) != set(ACTIONS):
                raise ValueError("native action set differs from strict manifest")
            (stem.with_suffix(".json")).write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
            path = retain_native_failure(args, dtype, index, command, completed.returncode,
                                         stdout, stderr, str(error))
            raise RuntimeError(f"{dtype}: invalid native report retained at {path}: {error}") from error

    def reference() -> dict[str, Any]:
        report: dict[str, Any] = {}
        for name in ACTIONS:
            call = callables[name]
            for _ in range(WARMUP):
                result = call()
            del result
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

    native()  # Preserve independent correctness evidence before timed rounds.
    rounds: list[dict[str, Any]] = []
    timing_failures: list[dict[str, Any]] = []
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
            json.dumps(rounds, indent=2) + "\n", encoding="utf-8")
        for name in ACTIONS:
            for metric in ("median_ms", "p95_ms"):
                if native_result[name][metric] >= pytorch_result[name][metric]:
                    item = {
                        "round": index + 1, "action": name, "metric": metric,
                        "pytorch_ms": pytorch_result[name][metric],
                        "arraymancer_ms": native_result[name][metric],
                    }
                    timing_failures.append(item)
                    failures.append(f"{dtype}: round {index + 1} {name} {metric} did not win")
        if timing_failures:
            (args.output / f"failed_timing_{dtype}.json").write_text(
                json.dumps(timing_failures, indent=2) + "\n", encoding="utf-8")

    return {
        "input_dtype": dtype,
        "values_dtype": dtype,
        "indices_dtype": "int64",
        "actions": list(ACTIONS),
        "input_sha256": input_hash,
        "correctness": "passed",
        "native_invocations": native_invocations,
        "rounds": rounds,
        "timing_failures": timing_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/paired_extrema_gradient_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/paired_extrema_gradient_benchmark"))
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
        except (RuntimeError, OSError, ValueError, AssertionError) as error:
            failures.append(str(error))
            failed_input = args.output / f"input_{dtype}.json"
            if failed_input.is_file():
                input_hashes[dtype] = hashlib.sha256(failed_input.read_bytes()).hexdigest()
            rounds_path = args.output / f"rounds_{dtype}.json"
            failed_rounds = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.is_file() else []
            failure_path = args.output / f"group_failure_{dtype}.json"
            failure_path.write_text(json.dumps({
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
    if (len(groups) != len(DTYPES) or action_count != 56 or completed_action_count != 56
            or comparison_count != 336):
        failures.append("paired-extrema gradient protocol requires 2 groups, 56 completed actions and 336 comparisons")
    report = {
        "torch": torch.__version__,
        "torch_git_revision": torch.version.git_version,
        "python": sys.version,
        "backend": "cpu",
        "runtime": {
            "nim_compiler": nim_version,
            "python": sys.version,
            "platform": sys.platform,
            "machine": platform.machine(),
        },
        "protocol": {
            "input_dtypes": list(DTYPES),
            "values_dtype": "same as input_dtype",
            "indices_dtype": "int64",
            "gradient_dtype": "same as input_dtype",
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
            "fresh_graph_per_call": True,
            "seed": "i % 7 - 3 + 0.25; output-shaped, finite, nonzero, resident before timing",
            "input_storage": "three resident matrices; cube is a view of base",
            "response_schema": {
                "actions": list(ACTIONS),
                "entry_fields": ["observation", "samples_ms", "median_ms", "p95_ms"],
                "observation_fields": list(OBSERVATION_FIELDS),
                "indices": "exact JSON int64 values",
                "independent_allocations": "all three outputs pairwise distinct and distinct from input",
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
            "invoke_commands": {
                group["input_dtype"]: group.get("native_invocations", []) for group in groups
            },
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
