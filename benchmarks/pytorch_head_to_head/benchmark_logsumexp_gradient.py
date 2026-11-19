#!/usr/bin/env python3
"""Benchmark seeded CPU logsumexp training actions against pinned PyTorch.

Build from the repository root with:
  nim c -d:release --path:src --out:build/logsumexp_gradient_runner benchmarks/pytorch_head_to_head/logsumexp_gradient_runner.nim
Run all 72 mandatory actions with:
  build/parity-oracle/bin/python benchmarks/pytorch_head_to_head/benchmark_logsumexp_gradient.py
Each action gets 50 warmups and 100 samples of 50 fresh-graph training calls
in three alternating backend-order rounds. Every median and p95 must strictly
beat PyTorch in every round. All dtype/action correctness and storage checks
finish before any competitive timing starts.
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

DTYPES = ("float32", "float64", "complex64", "complex128")
TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
KEEPDIMS = (False, True)
PROFILES = (
    "whole_matrix", "whole_cube",
    "rows", "rows_head", "rows_tail", "columns",
    "cube_axis0", "cube_axis1", "cube_axis2",
)
ACTIONS = tuple(
    f"{'keepdim' if keepdim else 'flat'}_{profile}"
    for keepdim in KEEPDIMS for profile in PROFILES
)
TORCH_VERSION = "2.15.0"
TORCH_GIT_REVISION = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
PYTHON_VERSION = (3, 13, 18)
ROWS, COLUMNS, CUBE_SHAPE = 128, 128, (16, 32, 32)
WARMUP, REPEATS, ITERATIONS, PAIRED_ROUNDS = 50, 100, 50, 3
ROOT = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/logsumexp_gradient_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_logsumexp_gradient.py")
SOURCE_PATHS = tuple(map(Path, (
    "src/arraymancer/autograd/gates_reduce.nim",
    "src/arraymancer/autograd/autograd_common.nim",
    "src/arraymancer/autograd/private/p_logsumexp_gradient.nim",
    "src/arraymancer/tensor/private/p_logsumexp.nim",
    "src/arraymancer/tensor/private/p_extrema.nim",
    "src/arraymancer/tensor/private/p_accelerate.nim",
    "src/arraymancer/tensor/aggregate.nim",
    "src/arraymancer/tensor/data_structure.nim",
    "src/arraymancer/tensor/init_cpu.nim",
    "src/arraymancer/tensor/backend/global_config.nim",
    "src/arraymancer.nim",
    "nim.cfg",
    "tests/parity/reference.json",
    str(RUNNER_SOURCE), str(SCRIPT_SOURCE),
)))
MANIFEST_FIELDS = (
    "input_dtype", "output_dtype", "rows", "columns", "shape", "values",
    "cube_shape", "cube_values", "rows_head_values", "rows_tail_values",
    "profiles", "actions", "seeds", "warmup", "repeats", "iterations",
)
OBSERVATION_FIELDS = {
    "input_dtype", "output_dtype", "gradient_dtype", "output_shape",
    "gradient_shape", "output_values", "gradient_values", "output_data_ptr",
    "gradient_data_ptr", "output_is_contiguous", "gradient_is_contiguous",
    "output_disjoint_from_inputs", "gradient_disjoint_from_inputs",
    "output_disjoint_from_seed", "gradient_disjoint_from_seed",
    "output_gradient_disjoint", "output_distinct_from_repeat",
    "gradient_distinct_from_repeat",
}
PROFILE_SHAPES = {
    "whole_matrix": (ROWS, COLUMNS),
    "whole_cube": CUBE_SHAPE,
    "rows": (ROWS, COLUMNS),
    "rows_head": (ROWS, COLUMNS),
    "rows_tail": (ROWS, COLUMNS),
    "columns": (ROWS, COLUMNS),
    "cube_axis0": CUBE_SHAPE,
    "cube_axis1": CUBE_SHAPE,
    "cube_axis2": CUBE_SHAPE,
}
PROFILE_AXES = {
    "rows": 1, "rows_head": 1, "rows_tail": 1, "columns": 0,
    "cube_axis0": 0, "cube_axis1": 1, "cube_axis2": 2,
}


def source_value(dtype: str, index: int, profile: int) -> float | complex:
    column = index % COLUMNS
    if dtype.startswith("complex"):
        if profile == 1:
            if column == 0:
                return complex(2.0, 0.125)
            if column == COLUMNS - 1:
                return complex(-2.0, -0.125)
        elif profile == 2:
            if column == 0:
                return complex(-2.0, -0.125)
            if column == COLUMNS - 1:
                return complex(2.0, 0.125)
        return complex(((index % 31) - 15) / 16.0, ((index % 9) - 4) / 64.0)
    if profile == 1:
        if column == 0:
            return 4.0
        if column == COLUMNS - 1:
            return -4.0
    elif profile == 2:
        if column == 0:
            return -4.0
        if column == COLUMNS - 1:
            return 4.0
    return ((index % 63) - 31) / 8.0


def tensor_values(value: torch.Tensor, dtype: str) -> list[Any]:
    flat = value.reshape(-1).tolist()
    if dtype.startswith("complex"):
        return [[float(item.real), float(item.imag)] for item in flat]
    return flat


def make_sources(dtype: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create resident matrices and a contiguous cube view of the base."""
    tensor_dtype = TORCH_DTYPES[dtype]
    sources: dict[str, torch.Tensor] = {}
    for name, profile in (("base", 0), ("rows_head", 1), ("rows_tail", 2)):
        values = [source_value(dtype, index, profile) for index in range(ROWS * COLUMNS)]
        sources[name] = torch.tensor(values, dtype=tensor_dtype).reshape(ROWS, COLUMNS)
    sources["cube"] = sources["base"].reshape(CUBE_SHAPE)
    matrix, rows_head, rows_tail, cube = (
        sources["base"], sources["rows_head"], sources["rows_tail"], sources["cube"])
    if (len({matrix.data_ptr(), rows_head.data_ptr(), rows_tail.data_ptr()}) != 3 or
            cube.data_ptr() != matrix.data_ptr() or
            not all(value.is_contiguous() for value in sources.values())):
        raise RuntimeError("logsumexp gradient inputs must be resident with independent matrices and a base-sharing cube view")
    payload: dict[str, Any] = {
        "rows": ROWS,
        "columns": COLUMNS,
        "shape": [ROWS, COLUMNS],
        "values": tensor_values(matrix, dtype),
        "cube_shape": list(CUBE_SHAPE),
        "cube_values": tensor_values(cube, dtype),
        "rows_head_values": tensor_values(rows_head, dtype),
        "rows_tail_values": tensor_values(rows_tail, dtype),
    }
    if payload["cube_values"] != payload["values"]:
        raise RuntimeError("logsumexp gradient cube manifest must equal base matrix values")
    return sources, payload


def action_spec(name: str) -> tuple[bool, str, int | None]:
    mode, profile = name.split("_", 1)
    if mode not in ("flat", "keepdim") or profile not in PROFILES:
        raise ValueError(f"Malformed logsumexp gradient action: {name}")
    return mode == "keepdim", profile, PROFILE_AXES.get(profile)


def expected_shape(name: str) -> list[int]:
    keepdim, profile, axis = action_spec(name)
    shape = list(PROFILE_SHAPES[profile])
    if profile.startswith("whole_"):
        return [1] * len(shape) if keepdim else []
    assert axis is not None
    if keepdim:
        shape[axis] = 1
    else:
        del shape[axis]
    return shape


def seed_for(shape: tuple[int, ...], dtype: torch.dtype, complex_dtype: bool) -> torch.Tensor:
    count = math.prod(shape)
    real = [index % 7 - 3 + 0.25 for index in range(count)]
    values: list[float | complex] = [complex(value, -0.5) if complex_dtype else value for value in real]
    if any(not math.isfinite(value.real if isinstance(value, complex) else value) or
           (isinstance(value, complex) and not math.isfinite(value.imag)) or value == 0
           for value in values):
        raise RuntimeError("benchmark seeds must be finite and nonzero")
    return torch.tensor(values, dtype=dtype).reshape(shape)


def bind_actions(sources: dict[str, torch.Tensor], dtype: str
                 ) -> tuple[dict[str, Callable[[], Any]], dict[str, Any], dict[str, torch.Tensor]]:
    """Bind public torch.logsumexp calls and resident output-shaped seeds."""
    callables: dict[str, Callable[[], Any]] = {}
    seeds: dict[str, Any] = {}
    seed_tensors: dict[str, torch.Tensor] = {}
    for name in ACTIONS:
        keepdim, profile, axis = action_spec(name)
        base = sources[{
            "whole_matrix": "base", "whole_cube": "cube",
            "rows": "base", "rows_head": "rows_head", "rows_tail": "rows_tail",
            "columns": "base", "cube_axis0": "cube", "cube_axis1": "cube",
            "cube_axis2": "cube",
        }[profile]]
        shape = tuple(expected_shape(name))
        seed = seed_for(shape, TORCH_DTYPES[dtype], dtype.startswith("complex"))
        seed_tensors[name] = seed
        seeds[name] = {
            "dtype": dtype,
            "shape": list(seed.shape),
            "values": tensor_values(seed, dtype),
        }
        dimensions: int | tuple[int, ...] = (
            tuple(range(base.dim())) if axis is None else axis)
        callables[name] = lambda base=base, seed=seed, dimensions=dimensions, keepdim=keepdim: _train(
            base, seed, dimensions, keepdim)
    return callables, seeds, seed_tensors


def _train(base: torch.Tensor, seed: torch.Tensor,
           dimensions: int | tuple[int, ...], keepdim: bool
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make a fresh leaf and graph, then compute the seeded input gradient."""
    leaf = base.detach().requires_grad_(True)
    output = torch.logsumexp(leaf, dim=dimensions, keepdim=keepdim)
    gradient, = torch.autograd.grad(output, leaf, grad_outputs=seed)
    return output, gradient, leaf


def _storage_range(value: torch.Tensor) -> tuple[int, int]:
    storage = value.untyped_storage()
    start = int(storage.data_ptr())
    end = start + int(storage.nbytes())
    if start <= 0 or end <= start:
        raise ValueError("logsumexp gradient preflight encountered empty or invalid tensor storage")
    return start, end


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _storage_range(left)
    right_start, right_end = _storage_range(right)
    return left_start < right_end and right_start < left_end


def _json_number(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "+Inf"
    if value == -math.inf:
        return "-Inf"
    return value


def _torch_values(value: torch.Tensor, dtype: str) -> list[Any]:
    flat = value.detach().reshape(-1).tolist()
    if dtype.startswith("complex"):
        return [{"real": _json_number(float(item.real)),
                 "imag": _json_number(float(item.imag))} for item in flat]
    return [_json_number(float(item)) for item in flat]


def _validate_torch_run(name: str, result: Any, dtype: str,
                        sources: dict[str, torch.Tensor], seed: torch.Tensor) -> None:
    if type(result) is not tuple or len(result) != 3:
        raise ValueError(f"{name}: malformed PyTorch training result")
    output, gradient, leaf = result
    expected_output_dtype = TORCH_DTYPES[dtype]
    if (type(output) is not torch.Tensor or type(gradient) is not torch.Tensor or
            type(leaf) is not torch.Tensor or output.dtype != expected_output_dtype or
            gradient.dtype != expected_output_dtype or leaf.dtype != expected_output_dtype or
            output.device.type != "cpu" or gradient.device.type != "cpu" or
            leaf.device.type != "cpu" or list(output.shape) != expected_shape(name) or
            list(gradient.shape) != list(leaf.shape) or
            list(leaf.shape) not in (list(PROFILE_SHAPES[action_spec(name)[1]]),) or
            not output.is_contiguous() or not gradient.is_contiguous() or
            not leaf.requires_grad or not output.requires_grad or gradient.requires_grad):
        raise ValueError(f"{name}: PyTorch training dtype, device, shape, layout, or graph mismatch")
    if list(seed.shape) != expected_shape(name) or seed.dtype != expected_output_dtype:
        raise ValueError(f"{name}: PyTorch seed dtype or shape mismatch")
    for source in sources.values():
        if _overlaps(output, source) or _overlaps(gradient, source):
            raise ValueError(f"{name}: PyTorch output or input gradient aliases a resident input")
    if (_overlaps(output, gradient) or _overlaps(output, seed) or _overlaps(gradient, seed)):
        raise ValueError(f"{name}: PyTorch output or input gradient aliases its seed or peer output")


def _reference_observation(name: str, result: Any, repeated: Any, dtype: str,
                           sources: dict[str, torch.Tensor], seed: torch.Tensor) -> dict[str, Any]:
    _validate_torch_run(name, result, dtype, sources, seed)
    _validate_torch_run(name, repeated, dtype, sources, seed)
    output, gradient, _ = result
    repeated_output, repeated_gradient, _ = repeated
    if any(_overlaps(left, right) for left, right in (
            (output, repeated_output), (output, repeated_gradient),
            (gradient, repeated_output), (gradient, repeated_gradient))):
        raise ValueError(f"{name}: repeated PyTorch training outputs share storage")
    return {
        "input_dtype": dtype,
        "output_dtype": str(output.dtype).removeprefix("torch."),
        "gradient_dtype": str(gradient.dtype).removeprefix("torch."),
        "output_shape": list(output.shape),
        "gradient_shape": list(gradient.shape),
        "output_values": _torch_values(output, dtype),
        "gradient_values": _torch_values(gradient, dtype),
        "output_data_ptr": str(output.data_ptr()),
        "gradient_data_ptr": str(gradient.data_ptr()),
        "output_is_contiguous": output.is_contiguous(),
        "gradient_is_contiguous": gradient.is_contiguous(),
        "output_disjoint_from_inputs": all(not _overlaps(output, source) for source in sources.values()),
        "gradient_disjoint_from_inputs": all(not _overlaps(gradient, source) for source in sources.values()),
        "output_disjoint_from_seed": not _overlaps(output, seed),
        "gradient_disjoint_from_seed": not _overlaps(gradient, seed),
        "output_gradient_disjoint": not _overlaps(output, gradient),
        "output_distinct_from_repeat": not _overlaps(output, repeated_output),
        "gradient_distinct_from_repeat": not _overlaps(gradient, repeated_gradient),
    }


def _check_scalar(name: str, actual: Any, expected: float | str, tolerance: float) -> None:
    if isinstance(expected, str):
        if expected not in ("NaN", "+Inf", "-Inf") or actual != expected:
            raise ValueError(f"{name}: nonfinite category mismatch")
    elif math.isnan(expected):
        if actual != "NaN":
            raise ValueError(f"{name}: nonfinite NaN category mismatch")
    elif expected == math.inf:
        if actual != "+Inf":
            raise ValueError(f"{name}: nonfinite +Inf category mismatch")
    elif expected == -math.inf:
        if actual != "-Inf":
            raise ValueError(f"{name}: nonfinite -Inf category mismatch")
    elif (type(actual) not in (int, float) or not math.isfinite(actual) or
          not math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance)):
        raise ValueError(f"{name}: finite value mismatch beyond {tolerance:g}")
    elif expected == 0.0 and float(actual) == 0.0 and math.copysign(1.0, float(actual)) != math.copysign(1.0, expected):
        raise ValueError(f"{name}: signed-zero category mismatch")


def _check_values(name: str, actual: Any, expected: list[Any], dtype: str) -> None:
    if type(actual) is not list or len(actual) != len(expected):
        raise ValueError(f"{name}: value count does not match exact output shape")
    tolerance = 1e-6 if dtype in ("float32", "complex64") else 1e-12
    if dtype.startswith("complex"):
        for item, reference in zip(actual, expected, strict=True):
            if type(item) is not dict or set(item) != {"real", "imag"}:
                raise ValueError(f"{name}: malformed complex output value")
            _check_scalar(name, item["real"], reference["real"], tolerance)
            _check_scalar(name, item["imag"], reference["imag"], tolerance)
    else:
        for item, reference in zip(actual, expected, strict=True):
            _check_scalar(name, item, reference, tolerance)


def _validate_observation(name: str, observation: Any, reference: dict[str, Any], dtype: str) -> None:
    if type(observation) is not dict or set(observation) != OBSERVATION_FIELDS:
        raise ValueError(f"{name}: malformed logsumexp gradient observation fields")
    expected_values_shape = expected_shape(name)
    expected_gradient_shape = list(PROFILE_SHAPES[action_spec(name)[1]])
    if (observation["input_dtype"] != dtype or observation["output_dtype"] != dtype or
            observation["gradient_dtype"] != dtype or
            observation["output_shape"] != expected_values_shape or
            observation["gradient_shape"] != expected_gradient_shape or
            observation["output_shape"] != reference["output_shape"] or
            observation["gradient_shape"] != reference["gradient_shape"]):
        raise ValueError(f"{name}: dtype or shape does not match the exact request")
    for field in ("output_is_contiguous", "gradient_is_contiguous", "output_disjoint_from_inputs",
                  "gradient_disjoint_from_inputs", "output_disjoint_from_seed",
                  "gradient_disjoint_from_seed", "output_gradient_disjoint",
                  "output_distinct_from_repeat", "gradient_distinct_from_repeat"):
        if observation[field] is not True:
            raise ValueError(f"{name}: {field} must prove independent storage")
    for field in ("output_data_ptr", "gradient_data_ptr"):
        pointer = observation[field]
        if type(pointer) is not str or not pointer.isdecimal() or int(pointer) <= 0:
            raise ValueError(f"{name}: {field} must expose a nonzero tensor data pointer")
    _check_values(name + " output", observation["output_values"], reference["output_values"], dtype)
    _check_values(name + " gradient", observation["gradient_values"], reference["gradient_values"], dtype)


def _check_input_stability(sources: dict[str, torch.Tensor], payload: dict[str, Any], dtype: str) -> None:
    for source, field in (("base", "values"), ("rows_head", "rows_head_values"),
                          ("rows_tail", "rows_tail_values"), ("cube", "cube_values")):
        if tensor_values(sources[source], dtype) != payload[field]:
            raise RuntimeError(f"Resident input changed during logsumexp gradient benchmark: {source}")


def _check_timing_record(name: str, entry: Any, reference: dict[str, Any], dtype: str) -> None:
    if type(entry) is not dict or set(entry) != {"observation", "samples_ms", "median_ms", "p95_ms"}:
        raise ValueError(f"{name}: malformed timing record")
    _validate_observation(name, entry["observation"], reference, dtype)
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


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _retain_native_failure(args: argparse.Namespace, dtype: str, index: int, phase: str,
                           returncode: int | None, stdout: str, stderr: str, error: str) -> Path:
    path = args.output / f"failed_native_{dtype}_{index}.json"
    path.write_text(json.dumps({
        "input_dtype": dtype,
        "phase": phase,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _verify_sources(artifacts: dict[str, Path], source_hashes: dict[str, str]) -> None:
    changed = [label for label, path in artifacts.items()
               if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[label]]
    if changed:
        raise RuntimeError("Sources or native runner changed during measurement: " + ", ".join(changed))


def _invoke_native(args: argparse.Namespace, dtype: str, input_path: Path, input_hash: str,
                   artifacts: dict[str, Path], source_hashes: dict[str, str],
                   references: dict[str, dict[str, Any]], preflight: bool,
                   invocation_index: int) -> dict[str, Any]:
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != input_hash:
        raise RuntimeError(f"{dtype}: input manifest changed before native invocation")
    _verify_sources(artifacts, source_hashes)
    command = [str(args.runner), str(input_path.resolve())]
    phase = "preflight" if preflight else "timing"
    if preflight:
        command.append("--preflight")
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = _text(error.stdout), _text(error.stderr)
        path = _retain_native_failure(
            args, dtype, invocation_index, phase, None, stdout, stderr,
            f"native runner {phase} timed out")
        raise RuntimeError(f"{dtype}: native {phase} timeout evidence retained at {path}") from error
    except OSError as error:
        path = _retain_native_failure(args, dtype, invocation_index, phase, None, "", "", repr(error))
        raise RuntimeError(f"{dtype}: native {phase} invocation evidence retained at {path}") from error

    stem = args.output / f"native_{dtype}_{invocation_index}"
    stem.with_suffix(".stdout.txt").write_text(completed.stdout, encoding="utf-8")
    stem.with_suffix(".stderr.txt").write_text(completed.stderr, encoding="utf-8")
    try:
        if completed.returncode != 0:
            raise ValueError(f"native runner exited with status {completed.returncode}")
        report = json.loads(completed.stdout)
        if type(report) is not dict or set(report) != set(ACTIONS):
            raise ValueError("native action set differs from the strict manifest")
        stem.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in ACTIONS:
            if preflight:
                _validate_observation(name, report[name], references[name], dtype)
            else:
                _check_timing_record(name, report[name], references[name], dtype)
        return report
    except (KeyError, TypeError, ValueError, AssertionError) as error:
        path = _retain_native_failure(
            args, dtype, invocation_index, phase, completed.returncode,
            completed.stdout, completed.stderr, str(error))
        raise RuntimeError(f"{dtype}: invalid native {phase} report retained at {path}: {error}") from error


def _prepare_group(args: argparse.Namespace, dtype: str, artifacts: dict[str, Path],
                   source_hashes: dict[str, str], invocation_index: int) -> dict[str, Any]:
    sources, payload = make_sources(dtype)
    callables, seeds, seed_tensors = bind_actions(sources, dtype)
    manifest = {
        "input_dtype": dtype,
        "output_dtype": dtype,
        **payload,
        "profiles": list(PROFILES),
        "actions": list(ACTIONS),
        "seeds": seeds,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "iterations": ITERATIONS,
    }
    input_path = args.output / f"input_{dtype}.json"
    input_path.write_text(json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    references: dict[str, dict[str, Any]] = {}
    for name in ACTIONS:
        call = callables[name]
        result, repeated = call(), call()
        references[name] = _reference_observation(
            name, result, repeated, dtype, sources, seed_tensors[name])
    _check_input_stability(sources, payload, dtype)
    native_preflight = _invoke_native(
        args, dtype, input_path, input_hash, artifacts, source_hashes,
        references, True, invocation_index)
    _check_input_stability(sources, payload, dtype)
    return {
        "input_dtype": dtype,
        "sources": sources,
        "payload": payload,
        "callables": callables,
        "seeds": seeds,
        "references": references,
        "input_path": input_path,
        "input_hash": input_hash,
        "native_preflight": native_preflight,
    }


def _consume_torch(result: Any) -> None:
    """Observe both eager allocations without timing another operation."""
    _ = result[0].data_ptr(), result[1].data_ptr()


def _time_reference(callables: dict[str, Callable[[], Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in ACTIONS:
        call = callables[name]
        for _ in range(WARMUP):
            _consume_torch(call())
        samples = []
        for _ in range(REPEATS):
            start = time.perf_counter_ns()
            for _ in range(ITERATIONS):
                _consume_torch(call())
            samples.append((time.perf_counter_ns() - start) / 1e6 / ITERATIONS)
        ordered = sorted(samples)
        report[name] = {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "p95_ms": ordered[math.ceil(0.95 * REPEATS) - 1],
        }
    return report


def _time_group(args: argparse.Namespace, prepared: dict[str, Any], failures: list[str],
                artifacts: dict[str, Path], source_hashes: dict[str, str],
                invocation_index: int) -> dict[str, Any]:
    dtype = prepared["input_dtype"]
    rounds = []
    timing_failures: list[dict[str, Any]] = []
    for index in range(PAIRED_ROUNDS):
        order = "pytorch-first" if index % 2 == 0 else "arraymancer-first"
        if index % 2 == 0:
            pytorch_result = _time_reference(prepared["callables"])
            native_result = _invoke_native(
                args, dtype, prepared["input_path"], prepared["input_hash"], artifacts,
                source_hashes, prepared["references"], False, invocation_index + index)
        else:
            native_result = _invoke_native(
                args, dtype, prepared["input_path"], prepared["input_hash"], artifacts,
                source_hashes, prepared["references"], False, invocation_index + index)
            pytorch_result = _time_reference(prepared["callables"])
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
                    item = {
                        "round": index + 1, "action": name, "metric": metric,
                        "pytorch_ms": pytorch_result[name][metric],
                        "arraymancer_ms": native_result[name][metric],
                    }
                    timing_failures.append(item)
                    failures.append(f"{dtype}: round {index + 1} {name} {metric} did not win")
        _check_input_stability(prepared["sources"], prepared["payload"], dtype)
    if timing_failures:
        (args.output / f"failed_timing_{dtype}.json").write_text(
            json.dumps(timing_failures, indent=2) + "\n", encoding="utf-8")
    return {
        "input_dtype": dtype,
        "output_dtype": dtype,
        "actions": list(ACTIONS),
        "input_sha256": prepared["input_hash"],
        "correctness": "passed",
        "native_preflight": "passed",
        "rounds": rounds,
        "timing_failures": timing_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/logsumexp_gradient_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/logsumexp_gradient_benchmark"))
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
    nim_version = subprocess.run(["nim", "--version"], check=True,
                                 capture_output=True, text=True).stdout.strip()
    failures: list[str] = []
    prepared: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    invocation_counts = {dtype: 0 for dtype in DTYPES}

    # Complete every action for every dtype before entering any timed round.
    for dtype in DTYPES:
        try:
            item = _prepare_group(
                args, dtype, artifacts, hashes, invocation_counts[dtype])
            invocation_counts[dtype] += 1
            prepared.append(item)
            input_hashes[dtype] = item["input_hash"]
        except (RuntimeError, ValueError, TypeError, AssertionError) as error:
            failures.append(str(error))
            input_path = args.output / f"input_{dtype}.json"
            if input_path.is_file():
                input_hashes[dtype] = hashlib.sha256(input_path.read_bytes()).hexdigest()
            rounds_path = args.output / f"rounds_{dtype}.json"
            failed_rounds = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.is_file() else []
            failure_path = args.output / f"group_failure_{dtype}.json"
            failure_path.write_text(json.dumps({
                "input_dtype": dtype,
                "output_dtype": dtype,
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "error": str(error),
                "rounds": failed_rounds,
            }, indent=2) + "\n", encoding="utf-8")
            groups.append({
                "input_dtype": dtype,
                "output_dtype": dtype,
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "correctness": "failed",
                "error": str(error),
                "rounds": failed_rounds,
            })

    if not failures:
        for item in prepared:
            dtype = item["input_dtype"]
            try:
                groups.append(_time_group(
                    args, item, failures, artifacts, hashes, invocation_counts[dtype]))
                invocation_counts[dtype] += PAIRED_ROUNDS
            except (RuntimeError, ValueError, TypeError, AssertionError) as error:
                failures.append(str(error))
                rounds_path = args.output / f"rounds_{dtype}.json"
                failed_rounds = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.is_file() else []
                failure_path = args.output / f"group_failure_{dtype}.json"
                failure_path.write_text(json.dumps({
                    "input_dtype": dtype,
                    "output_dtype": dtype,
                    "input_sha256": item["input_hash"],
                    "actions": list(ACTIONS),
                    "error": str(error),
                    "rounds": failed_rounds,
                }, indent=2) + "\n", encoding="utf-8")
                groups.append({
                    "input_dtype": dtype,
                    "output_dtype": dtype,
                    "input_sha256": item["input_hash"],
                    "actions": list(ACTIONS),
                    "correctness": "failed",
                    "error": str(error),
                    "rounds": failed_rounds,
                })
    else:
        for item in prepared:
            dtype = item["input_dtype"]
            groups.append({
                "input_dtype": dtype,
                "output_dtype": dtype,
                "actions": list(ACTIONS),
                "input_sha256": item["input_hash"],
                "correctness": "preflight_passed_timing_skipped",
                "native_preflight": "passed",
                "timing_skipped": "at least one dtype/action failed global preflight",
                "rounds": [],
            })
        if len(prepared) != len(DTYPES):
            failures.append("No timing began because complete 4-dtype preflight did not pass")

    _verify_sources(artifacts, hashes)
    for dtype, digest in input_hashes.items():
        path = args.output / f"input_{dtype}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(f"Input manifest changed during measurement: {dtype}")
    action_count = len(groups) * len(ACTIONS)
    preflight_action_count = len(prepared) * len(ACTIONS)
    completed_action_count = sum(
        len(group["actions"]) for group in groups
        if group.get("correctness") == "passed" and len(group.get("rounds", [])) == PAIRED_ROUNDS)
    comparison_count = sum(
        len(group["actions"]) * len(group.get("rounds", [])) * 2 for group in groups)
    if (len(groups) != len(DTYPES) or action_count != 72 or preflight_action_count != 72 or
            completed_action_count != 72 or comparison_count != 432):
        failures.append("logsumexp gradient protocol requires 4 groups, 72 completed actions and 432 comparisons")
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
            "output_dtypes": {dtype: dtype for dtype in DTYPES},
            "profiles": list(PROFILES),
            "keepdim_values": list(KEEPDIMS),
            "actions_per_group": list(ACTIONS),
            "group_count": len(groups),
            "action_count": action_count,
            "preflight_action_count": preflight_action_count,
            "completed_action_count": completed_action_count,
            "warmup_calls_per_action": WARMUP,
            "samples_per_action_per_round": REPEATS,
            "allocating_calls_per_sample": ITERATIONS,
            "paired_rounds_per_group": PAIRED_ROUNDS,
            "comparisons": comparison_count,
            "fresh_graph_per_call": True,
            "seed": "i % 7 - 3 + 0.25; complex imaginary component -0.5; finite and nonzero",
            "input_storage": "three resident matrices; cube is a contiguous view of base",
            "preflight": "all dtype/action values, shapes, dtypes, pointer independence and repeat storage checked before timing",
            "timing_gate": "strict median and p95 win in every paired round",
            "response_schema": {
                "actions": list(ACTIONS),
                "entry_fields": ["observation", "samples_ms", "median_ms", "p95_ms"],
                "observation_fields": sorted(OBSERVATION_FIELDS),
                "complex_values": "objects with real and imag numeric values or exact NaN/+Inf/-Inf category strings",
                "tolerance": {"float32": 1e-6, "complex64": 1e-6,
                              "float64": 1e-12, "complex128": 1e-12},
            },
        },
        "source_sha256": {label: digest for label, digest in hashes.items()
                           if label != "native_runner"},
        "runner_sha256": hashes["native_runner"],
        "binary_sha256": hashes["native_runner"],
        "input_sha256": input_hashes,
        "commands": {
            "build_cwd": str(ROOT),
            "build_command": compile_command,
            "compiler_version": nim_version,
            "invoke_command_template": [str(args.runner), "<input-manifest.json>"],
            "preflight_command_template": [str(args.runner), "<input-manifest.json>", "--preflight"],
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
        "preflight_actions": preflight_action_count,
        "completed_actions": completed_action_count,
        "comparisons": comparison_count,
        "failures": failures,
        "passed": not failures,
        "results": str(result_path),
    }))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
