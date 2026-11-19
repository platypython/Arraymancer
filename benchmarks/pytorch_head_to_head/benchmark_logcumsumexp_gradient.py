#!/usr/bin/env python3
"""Benchmark fresh-graph CPU logcumsumexp gradients against pinned PyTorch.

Build the native runner from the repository root with:

  nim c -d:release --path:src -o:build/logcumsumexp_gradient_runner \
    benchmarks/pytorch_head_to_head/logcumsumexp_gradient_runner.nim

Run with the pinned CPython 3.13.18 and PyTorch 2.15.0 environment used by the
paired CPU benchmarks. Every timed call creates a new graph and consumes both
full allocations without adding a timed tensor operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
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
PROFILES = (
    "rows", "rows_head", "rows_tail", "columns",
    "cube_axis0", "cube_axis1", "cube_axis2", "vector",
)
ACTIONS = PROFILES
TORCH_VERSION = "2.15.0"
TORCH_GIT_REVISION = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
PYTHON_VERSION = (3, 13, 18)
ROWS, COLUMNS, CUBE_SHAPE = 128, 128, (16, 32, 32)
WARMUP, REPEATS, ITERATIONS, PAIRED_ROUNDS = 50, 100, 50, 3
ROOT = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/logcumsumexp_gradient_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_logcumsumexp_gradient.py")
SOURCE_PATHS = tuple(map(Path, (
    "src/arraymancer/autograd/gates_reduce.nim",
    "src/arraymancer/autograd/autograd_common.nim",
    "src/arraymancer/autograd/private/p_logcumsumexp_gradient.nim",
    "src/arraymancer/tensor/private/p_logcumsumexp.nim",
    "src/arraymancer/tensor/private/p_logsumexp.nim",
    "src/arraymancer/tensor/private/p_extrema.nim",
    "src/arraymancer/tensor/private/p_accelerate.nim",
    "src/arraymancer/tensor/aggregate.nim",
    "src/arraymancer/tensor/data_structure.nim",
    "src/arraymancer/tensor/init_cpu.nim",
    "src/arraymancer/tensor/backend/global_config.nim",
    "src/arraymancer/tensor/backend/openmp.nim",
    "src/arraymancer/laser/tensor/initialization.nim",
    "src/arraymancer/tensor.nim",
    "src/arraymancer.nim",
    "nim.cfg",
    "tests/parity/reference.json",
    str(RUNNER_SOURCE), str(SCRIPT_SOURCE),
)))
MANIFEST_FIELDS = (
    "input_dtype", "output_dtype", "rows", "columns", "shape", "values",
    "cube_shape", "cube_values", "vector_shape", "vector_values",
    "rows_head_values", "rows_tail_values", "profiles", "actions", "seeds",
    "warmup", "repeats", "iterations",
)
OBSERVATION_FIELDS = {
    "input_dtype", "output_dtype", "gradient_dtype", "output_shape",
    "gradient_shape", "output_values", "gradient_values", "output_data_ptr",
    "gradient_data_ptr", "output_is_contiguous", "gradient_is_contiguous",
    "output_disjoint_from_inputs", "gradient_disjoint_from_inputs",
    "output_disjoint_from_seeds", "gradient_disjoint_from_seeds",
    "output_gradient_disjoint", "output_distinct_from_repeat",
    "gradient_distinct_from_repeat", "output_repeated_values_match",
    "gradient_repeated_values_match",
}
PROFILE_SHAPES = {
    "rows": (ROWS, COLUMNS),
    "rows_head": (ROWS, COLUMNS),
    "rows_tail": (ROWS, COLUMNS),
    "columns": (ROWS, COLUMNS),
    "cube_axis0": CUBE_SHAPE,
    "cube_axis1": CUBE_SHAPE,
    "cube_axis2": CUBE_SHAPE,
    "vector": (ROWS * COLUMNS,),
}
PROFILE_AXES = {
    "rows": 1, "rows_head": 1, "rows_tail": 1, "columns": 0,
    "cube_axis0": 0, "cube_axis1": 1, "cube_axis2": 2, "vector": 0,
}
PROFILE_INPUTS = {
    "rows": "base", "rows_head": "rows_head", "rows_tail": "rows_tail",
    "columns": "base", "cube_axis0": "cube", "cube_axis1": "cube",
    "cube_axis2": "cube", "vector": "vector",
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
    return [float(item) for item in flat]


def make_sources(dtype: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create three resident matrices and base-sharing cube/vector views."""
    tensor_dtype = TORCH_DTYPES[dtype]
    sources: dict[str, torch.Tensor] = {}
    for name, profile in (("base", 0), ("rows_head", 1), ("rows_tail", 2)):
        values = [source_value(dtype, index, profile) for index in range(ROWS * COLUMNS)]
        sources[name] = torch.tensor(values, dtype=tensor_dtype).reshape(ROWS, COLUMNS)
    sources["cube"] = sources["base"].reshape(CUBE_SHAPE)
    sources["vector"] = sources["base"].reshape(ROWS * COLUMNS)
    matrix, rows_head, rows_tail, cube, vector = (
        sources["base"], sources["rows_head"], sources["rows_tail"],
        sources["cube"], sources["vector"])
    if (len({matrix.data_ptr(), rows_head.data_ptr(), rows_tail.data_ptr()}) != 3 or
            cube.data_ptr() != matrix.data_ptr() or vector.data_ptr() != matrix.data_ptr() or
            not all(value.is_contiguous() for value in sources.values())):
        raise RuntimeError("logcumsumexp gradient requires three resident matrices and base-sharing cube/vector views")
    payload: dict[str, Any] = {
        "rows": ROWS,
        "columns": COLUMNS,
        "shape": [ROWS, COLUMNS],
        "values": tensor_values(matrix, dtype),
        "cube_shape": list(CUBE_SHAPE),
        "cube_values": tensor_values(cube, dtype),
        "vector_shape": [ROWS * COLUMNS],
        "vector_values": tensor_values(vector, dtype),
        "rows_head_values": tensor_values(rows_head, dtype),
        "rows_tail_values": tensor_values(rows_tail, dtype),
    }
    if (payload["cube_values"] != payload["values"] or
            payload["vector_values"] != payload["values"]):
        raise RuntimeError("logcumsumexp gradient cube/vector manifests must equal base matrix values")
    return sources, payload


def expected_shape(name: str) -> list[int]:
    if name not in PROFILE_SHAPES:
        raise ValueError(f"Malformed logcumsumexp gradient action: {name}")
    return list(PROFILE_SHAPES[name])


def seed_for(shape: tuple[int, ...], dtype: torch.dtype,
             complex_dtype: bool) -> torch.Tensor:
    count = math.prod(shape)
    if complex_dtype:
        values: list[float | complex] = [
            complex(index % 5 - 2, (index % 3 - 1) / 4.0) for index in range(count)
        ]
    else:
        values = [float(index % 5 - 2) for index in range(count)]
    return torch.tensor(values, dtype=dtype).reshape(shape)


def _train(base: torch.Tensor, seed: torch.Tensor,
           axis: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make a fresh leaf and graph, then compute the seeded input gradient."""
    leaf = base.detach().requires_grad_(True)
    output = torch.logcumsumexp(leaf, dim=axis)
    gradient, = torch.autograd.grad(output, leaf, grad_outputs=seed)
    return output, gradient, leaf


def bind_actions(sources: dict[str, torch.Tensor], dtype: str
                 ) -> tuple[dict[str, Callable[[], Any]], dict[str, Any], dict[str, torch.Tensor]]:
    """Bind public torch scans and independently resident output-shaped seeds."""
    callables: dict[str, Callable[[], Any]] = {}
    seeds: dict[str, Any] = {}
    seed_tensors: dict[str, torch.Tensor] = {}
    for name in ACTIONS:
        source = sources[PROFILE_INPUTS[name]]
        seed = seed_for(tuple(expected_shape(name)), TORCH_DTYPES[dtype], dtype.startswith("complex"))
        if any(_overlaps(seed, tensor) for tensor in sources.values()):
            raise RuntimeError(f"{name}: resident seed aliases a source tensor")
        if any(_overlaps(seed, other) for other in seed_tensors.values()):
            raise RuntimeError(f"{name}: action seeds must own independent allocations")
        seed_tensors[name] = seed
        seeds[name] = {
            "dtype": dtype,
            "shape": list(seed.shape),
            "values": tensor_values(seed, dtype),
        }
        axis = PROFILE_AXES[name]
        callables[name] = lambda source=source, seed=seed, axis=axis: _train(source, seed, axis)
    return callables, seeds, seed_tensors


def _storage_range(value: torch.Tensor) -> tuple[int, int]:
    storage = value.untyped_storage()
    start = int(storage.data_ptr())
    end = start + int(storage.nbytes())
    if start <= 0 or end <= start:
        raise ValueError("logcumsumexp gradient preflight encountered empty or invalid tensor storage")
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


def _same_scalar(left: Any, right: Any) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return type(left) is str and type(right) is str and left == right
    if (type(left) not in (int, float) or type(right) not in (int, float) or
            float(left) != float(right)):
        return False
    if float(left) == 0.0:
        return math.copysign(1.0, float(left)) == math.copysign(1.0, float(right))
    return True


def _same_values(left: list[Any], right: list[Any], dtype: str) -> bool:
    if len(left) != len(right):
        return False
    if dtype.startswith("complex"):
        return all(
            type(a) is dict and type(b) is dict and set(a) == {"real", "imag"} and
            set(b) == {"real", "imag"} and
            _same_scalar(a["real"], b["real"]) and _same_scalar(a["imag"], b["imag"])
            for a, b in zip(left, right, strict=True)
        )
    return all(_same_scalar(a, b) for a, b in zip(left, right, strict=True))


def _validate_torch_run(name: str, result: Any, dtype: str,
                        sources: dict[str, torch.Tensor],
                        seed_tensors: dict[str, torch.Tensor]) -> None:
    if type(result) is not tuple or len(result) != 3:
        raise ValueError(f"{name}: malformed PyTorch training result")
    output, gradient, leaf = result
    expected_dtype = TORCH_DTYPES[dtype]
    input_shape = expected_shape(name)
    if (type(output) is not torch.Tensor or type(gradient) is not torch.Tensor or
            type(leaf) is not torch.Tensor or output.dtype != expected_dtype or
            gradient.dtype != expected_dtype or leaf.dtype != expected_dtype or
            output.device.type != "cpu" or gradient.device.type != "cpu" or
            leaf.device.type != "cpu" or list(output.shape) != input_shape or
            list(gradient.shape) != input_shape or list(leaf.shape) != input_shape or
            not output.is_contiguous() or not gradient.is_contiguous() or
            not leaf.requires_grad or not output.requires_grad or gradient.requires_grad):
        raise ValueError(f"{name}: PyTorch training dtype, device, shape, layout, or graph mismatch")
    for source in sources.values():
        if _overlaps(output, source) or _overlaps(gradient, source):
            raise ValueError(f"{name}: PyTorch output or input gradient aliases a resident input")
    for seed in seed_tensors.values():
        if _overlaps(output, seed) or _overlaps(gradient, seed):
            raise ValueError(f"{name}: PyTorch output or input gradient aliases a resident seed")
    if _overlaps(output, gradient):
        raise ValueError(f"{name}: PyTorch forward output and input gradient share storage")


def _reference_observation(name: str, result: Any, repeated: Any, dtype: str,
                           sources: dict[str, torch.Tensor],
                           seed_tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    _validate_torch_run(name, result, dtype, sources, seed_tensors)
    _validate_torch_run(name, repeated, dtype, sources, seed_tensors)
    output, gradient, _ = result
    repeated_output, repeated_gradient, _ = repeated
    ranges = (
        (output, repeated_output), (output, repeated_gradient),
        (gradient, repeated_output), (gradient, repeated_gradient),
    )
    if any(_overlaps(left, right) for left, right in ranges):
        raise ValueError(f"{name}: repeated PyTorch training outputs share storage")
    output_values = _torch_values(output, dtype)
    gradient_values = _torch_values(gradient, dtype)
    output_repeat_values = _torch_values(repeated_output, dtype)
    gradient_repeat_values = _torch_values(repeated_gradient, dtype)
    if (not _same_values(output_values, output_repeat_values, dtype) or
            not _same_values(gradient_values, gradient_repeat_values, dtype)):
        raise ValueError(f"{name}: repeated PyTorch values differ, including signed-zero or nonfinite categories")
    return {
        "input_dtype": dtype,
        "output_dtype": str(output.dtype).removeprefix("torch."),
        "gradient_dtype": str(gradient.dtype).removeprefix("torch."),
        "output_shape": list(output.shape),
        "gradient_shape": list(gradient.shape),
        "output_values": output_values,
        "gradient_values": gradient_values,
        "output_data_ptr": str(output.data_ptr()),
        "gradient_data_ptr": str(gradient.data_ptr()),
        "output_is_contiguous": output.is_contiguous(),
        "gradient_is_contiguous": gradient.is_contiguous(),
        "output_disjoint_from_inputs": all(not _overlaps(output, source) for source in sources.values()),
        "gradient_disjoint_from_inputs": all(not _overlaps(gradient, source) for source in sources.values()),
        "output_disjoint_from_seeds": all(not _overlaps(output, seed) for seed in seed_tensors.values()),
        "gradient_disjoint_from_seeds": all(not _overlaps(gradient, seed) for seed in seed_tensors.values()),
        "output_gradient_disjoint": not _overlaps(output, gradient),
        "output_distinct_from_repeat": not _overlaps(output, repeated_output),
        "gradient_distinct_from_repeat": not _overlaps(gradient, repeated_gradient),
        "output_repeated_values_match": True,
        "gradient_repeated_values_match": True,
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


def _validate_observation(name: str, observation: Any,
                          reference: dict[str, Any] | None, dtype: str) -> None:
    if type(observation) is not dict or set(observation) != OBSERVATION_FIELDS:
        raise ValueError(f"{name}: malformed logcumsumexp gradient observation fields")
    shape = expected_shape(name)
    if (observation["input_dtype"] != dtype or observation["output_dtype"] != dtype or
            observation["gradient_dtype"] != dtype or observation["output_shape"] != shape or
            observation["gradient_shape"] != shape):
        raise ValueError(f"{name}: dtype or shape does not match the exact request")
    for field in (
            "output_is_contiguous", "gradient_is_contiguous", "output_disjoint_from_inputs",
            "gradient_disjoint_from_inputs", "output_disjoint_from_seeds",
            "gradient_disjoint_from_seeds", "output_gradient_disjoint",
            "output_distinct_from_repeat", "gradient_distinct_from_repeat",
            "output_repeated_values_match", "gradient_repeated_values_match"):
        if observation[field] is not True:
            raise ValueError(f"{name}: {field} must prove independent storage and stable repeated values")
    for field in ("output_data_ptr", "gradient_data_ptr"):
        pointer = observation[field]
        if type(pointer) is not str or not pointer.isdecimal() or int(pointer) <= 0:
            raise ValueError(f"{name}: {field} must expose a nonzero tensor data pointer")
    if reference is not None:
        if (observation["output_shape"] != reference["output_shape"] or
                observation["gradient_shape"] != reference["gradient_shape"]):
            raise ValueError(f"{name}: native shape differs from the PyTorch reference")
        _check_values(name + " output", observation["output_values"], reference["output_values"], dtype)
        _check_values(name + " gradient", observation["gradient_values"], reference["gradient_values"], dtype)
    else:
        if (type(observation["output_values"]) is not list or
                len(observation["output_values"]) != math.prod(shape) or
                type(observation["gradient_values"]) is not list or
                len(observation["gradient_values"]) != math.prod(shape)):
            raise ValueError(f"{name}: native values do not match the exact output shape")


def _check_input_stability(sources: dict[str, torch.Tensor], payload: dict[str, Any],
                           dtype: str, seed_tensors: dict[str, torch.Tensor],
                           seeds: dict[str, Any]) -> None:
    for source, field in (("base", "values"), ("rows_head", "rows_head_values"),
                          ("rows_tail", "rows_tail_values"), ("cube", "cube_values"),
                          ("vector", "vector_values")):
        if tensor_values(sources[source], dtype) != payload[field]:
            raise RuntimeError(f"Resident logcumsumexp gradient input changed: {source}")
    for name, tensor in seed_tensors.items():
        if tensor_values(tensor, dtype) != seeds[name]["values"]:
            raise RuntimeError(f"Resident logcumsumexp gradient seed changed: {name}")


def _check_timing_record(name: str, entry: Any,
                         reference: dict[str, Any], dtype: str) -> None:
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
                           returncode: int | None, stdout: str, stderr: str,
                           error: str) -> Path:
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


def _verify_sources(artifacts: dict[str, Path], source_hashes: dict[str, str],
                    hash_checks: list[dict[str, Any]], stage: str) -> None:
    current = {label: hashlib.sha256(path.read_bytes()).hexdigest()
               for label, path in artifacts.items()}
    hash_checks.append({"stage": stage, "sha256": current})
    changed = [label for label, digest in current.items() if digest != source_hashes[label]]
    if changed:
        raise RuntimeError("Sources or native runner changed during measurement: " + ", ".join(changed))


def _verify_input(path: Path, expected_hash: str, hash_checks: list[dict[str, Any]],
                  stage: str) -> None:
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    hash_checks.append({"stage": stage, "path": str(path), "sha256": current})
    if current != expected_hash:
        raise RuntimeError(f"Input manifest changed during measurement: {path.name}")


def _invoke_native(args: argparse.Namespace, dtype: str, input_path: Path,
                   input_hash: str, artifacts: dict[str, Path],
                   source_hashes: dict[str, str],
                   references: dict[str, dict[str, Any]], preflight: bool,
                   invocation_index: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    phase = "preflight" if preflight else "timing"
    _verify_input(input_path, input_hash, args.input_hash_checks,
                  f"before_native_{phase}_{dtype}_{invocation_index}")
    _verify_sources(artifacts, source_hashes, args.hash_checks,
                    f"before_native_{phase}_{dtype}_{invocation_index}")
    command = [str(args.runner), str(input_path.resolve())]
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
    _verify_sources(artifacts, source_hashes, args.hash_checks,
                    f"after_native_{phase}_{dtype}_{invocation_index}")
    _verify_input(input_path, input_hash, args.input_hash_checks,
                  f"after_native_{phase}_{dtype}_{invocation_index}")
    try:
        if completed.returncode != 0:
            raise ValueError(f"native runner exited with status {completed.returncode}")
        report = json.loads(completed.stdout)
        if type(report) is not dict or set(report) != set(ACTIONS):
            raise ValueError("native action set differs from the strict manifest")
        stem.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        path = _retain_native_failure(
            args, dtype, invocation_index, phase, completed.returncode,
            completed.stdout, completed.stderr, str(error))
        raise RuntimeError(f"{dtype}: invalid native {phase} report retained at {path}: {error}") from error

    validation_errors: list[dict[str, str]] = []
    for name in ACTIONS:
        try:
            if preflight:
                _validate_observation(name, report[name], references.get(name), dtype)
            else:
                reference = references.get(name)
                if reference is None:
                    raise ValueError(f"{name}: missing PyTorch preflight reference")
                _check_timing_record(name, report[name], reference, dtype)
        except (KeyError, TypeError, ValueError, AssertionError) as error:
            validation_errors.append({"action": name, "error": str(error)})
    return report, validation_errors


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
    reference_failures: list[dict[str, str]] = []
    try:
        _verify_sources(artifacts, source_hashes, args.hash_checks,
                        f"before_pytorch_preflight_{dtype}")
    except RuntimeError as error:
        reference_failures.append({"action": "<source_hash>", "error": str(error)})
    try:
        _verify_input(input_path, input_hash, args.input_hash_checks,
                      f"before_pytorch_preflight_{dtype}")
    except RuntimeError as error:
        reference_failures.append({"action": "<input_hash>", "error": str(error)})
    try:
        _check_input_stability(sources, payload, dtype, seed_tensors, seeds)
    except RuntimeError as error:
        reference_failures.append({"action": "<resident_inputs>", "error": str(error)})
    for name in ACTIONS:
        try:
            result, repeated = callables[name](), callables[name]()
            references[name] = _reference_observation(
                name, result, repeated, dtype, sources, seed_tensors)
        except (RuntimeError, ValueError, TypeError, AssertionError) as error:
            reference_failures.append({"action": name, "error": str(error)})
    try:
        _check_input_stability(sources, payload, dtype, seed_tensors, seeds)
    except RuntimeError as error:
        reference_failures.append({"action": "<resident_inputs_after_pytorch>", "error": str(error)})
    try:
        _verify_input(input_path, input_hash, args.input_hash_checks,
                      f"after_pytorch_preflight_{dtype}")
    except RuntimeError as error:
        reference_failures.append({"action": "<input_hash_after_pytorch>", "error": str(error)})
    try:
        _verify_sources(artifacts, source_hashes, args.hash_checks,
                        f"after_pytorch_preflight_{dtype}")
    except RuntimeError as error:
        reference_failures.append({"action": "<source_hash_after_pytorch>", "error": str(error)})

    native_preflight: dict[str, Any] | None = None
    native_failures: list[dict[str, str]] = []
    try:
        native_preflight, native_failures = _invoke_native(
            args, dtype, input_path, input_hash, artifacts, source_hashes,
            references, True, invocation_index)
    except (RuntimeError, ValueError, TypeError, AssertionError) as error:
        native_failures.append({"action": "<native_preflight>", "error": str(error)})
    try:
        _check_input_stability(sources, payload, dtype, seed_tensors, seeds)
    except RuntimeError as error:
        native_failures.append({"action": "<resident_inputs_after_native>", "error": str(error)})

    validated = [name for name in ACTIONS
                 if name in references and not any(item["action"] == name for item in native_failures)]
    preflight = {
        "input_dtype": dtype,
        "input_sha256": input_hash,
        "actions_attempted": list(ACTIONS),
        "actions_validated": validated,
        "pytorch": references,
        "pytorch_failures": reference_failures,
        "arraymancer": native_preflight,
        "arraymancer_failures": native_failures,
    }
    (args.output / f"preflight_{dtype}.json").write_text(
        json.dumps(preflight, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    errors = [f"{dtype} PyTorch preflight {item['action']}: {item['error']}"
              for item in reference_failures]
    errors.extend(f"{dtype} native preflight {item['action']}: {item['error']}"
                  for item in native_failures)
    return {
        "input_dtype": dtype,
        "sources": sources,
        "payload": payload,
        "callables": callables,
        "seeds": seeds,
        "seed_tensors": seed_tensors,
        "references": references,
        "input_path": input_path,
        "input_hash": input_hash,
        "native_preflight": native_preflight,
        "preflight": preflight,
        "preflight_failures": errors,
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

    def time_pytorch(label: str) -> dict[str, Any]:
        _verify_sources(artifacts, source_hashes, args.hash_checks, f"before_{label}")
        _verify_input(prepared["input_path"], prepared["input_hash"],
                      args.input_hash_checks, f"before_{label}")
        _check_input_stability(
            prepared["sources"], prepared["payload"], dtype,
            prepared["seed_tensors"], prepared["seeds"])
        result = _time_reference(prepared["callables"])
        _check_input_stability(
            prepared["sources"], prepared["payload"], dtype,
            prepared["seed_tensors"], prepared["seeds"])
        _verify_input(prepared["input_path"], prepared["input_hash"],
                      args.input_hash_checks, f"after_{label}")
        _verify_sources(artifacts, source_hashes, args.hash_checks, f"after_{label}")
        return result

    for index in range(PAIRED_ROUNDS):
        order = "pytorch-first" if index % 2 == 0 else "arraymancer-first"
        if index % 2 == 0:
            pytorch_result = time_pytorch(f"pytorch_{dtype}_round_{index + 1}")
            native_result, native_errors = _invoke_native(
                args, dtype, prepared["input_path"], prepared["input_hash"], artifacts,
                source_hashes, prepared["references"], False, invocation_index + index)
        else:
            native_result, native_errors = _invoke_native(
                args, dtype, prepared["input_path"], prepared["input_hash"], artifacts,
                source_hashes, prepared["references"], False, invocation_index + index)
            pytorch_result = time_pytorch(f"pytorch_{dtype}_round_{index + 1}")
        rounds.append({
            "round": index + 1,
            "order": order,
            "pytorch": pytorch_result,
            "arraymancer": native_result,
        })
        (args.output / f"rounds_{dtype}.json").write_text(
            json.dumps(rounds, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for item in native_errors:
            failure = {"round": index + 1, **item}
            timing_failures.append(failure)
            failures.append(f"{dtype}: native timing {item['action']}: {item['error']}")
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
        _check_input_stability(
            prepared["sources"], prepared["payload"], dtype,
            prepared["seed_tensors"], prepared["seeds"])
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
        "preflight": prepared["preflight"],
        "rounds": rounds,
        "timing_failures": timing_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/logcumsumexp_gradient_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/logcumsumexp_gradient_benchmark"))
    args = parser.parse_args()
    args.hash_checks = []
    args.input_hash_checks = []
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

    # Complete every dtype/action preflight before beginning any timed round.
    for dtype in DTYPES:
        try:
            item = _prepare_group(
                args, dtype, artifacts, hashes, invocation_counts[dtype])
            invocation_counts[dtype] += 1
            prepared.append(item)
            input_hashes[dtype] = item["input_hash"]
            failures.extend(item["preflight_failures"])
        except (RuntimeError, ValueError, TypeError, AssertionError) as error:
            failures.append(str(error))
            input_path = args.output / f"input_{dtype}.json"
            if input_path.is_file():
                input_hashes[dtype] = hashlib.sha256(input_path.read_bytes()).hexdigest()
            preflight_path = args.output / f"preflight_{dtype}.json"
            failed_preflight = (json.loads(preflight_path.read_text(encoding="utf-8"))
                                if preflight_path.is_file() else None)
            failure_path = args.output / f"group_failure_{dtype}.json"
            failure_path.write_text(json.dumps({
                "input_dtype": dtype,
                "output_dtype": dtype,
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "error": str(error),
                "preflight": failed_preflight,
                "rounds": [],
            }, indent=2) + "\n", encoding="utf-8")
            groups.append({
                "input_dtype": dtype,
                "output_dtype": dtype,
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "correctness": "failed",
                "error": str(error),
                "preflight": failed_preflight,
                "rounds": [],
            })

    if not failures and len(prepared) == len(DTYPES):
        for item in prepared:
            dtype = item["input_dtype"]
            try:
                groups.append(_time_group(
                    args, item, failures, artifacts, hashes, invocation_counts[dtype]))
                invocation_counts[dtype] += PAIRED_ROUNDS
            except (RuntimeError, ValueError, TypeError, AssertionError) as error:
                failures.append(str(error))
                rounds_path = args.output / f"rounds_{dtype}.json"
                failed_rounds = (json.loads(rounds_path.read_text(encoding="utf-8"))
                                 if rounds_path.is_file() else [])
                failure_path = args.output / f"group_failure_{dtype}.json"
                failure_path.write_text(json.dumps({
                    "input_dtype": dtype,
                    "output_dtype": dtype,
                    "input_sha256": item["input_hash"],
                    "actions": list(ACTIONS),
                    "error": str(error),
                    "preflight": item["preflight"],
                    "rounds": failed_rounds,
                }, indent=2) + "\n", encoding="utf-8")
                groups.append({
                    "input_dtype": dtype,
                    "output_dtype": dtype,
                    "input_sha256": item["input_hash"],
                    "actions": list(ACTIONS),
                    "correctness": "failed",
                    "error": str(error),
                    "preflight": item["preflight"],
                    "rounds": failed_rounds,
                })
    else:
        for item in prepared:
            groups.append({
                "input_dtype": item["input_dtype"],
                "output_dtype": item["input_dtype"],
                "actions": list(ACTIONS),
                "input_sha256": item["input_hash"],
                "correctness": ("failed" if item["preflight_failures"] else
                                "preflight_passed_timing_skipped"),
                "preflight": item["preflight"],
                "native_preflight": "passed" if not item["preflight_failures"] else "failed",
                "timing_skipped": "at least one dtype/action failed global preflight",
                "rounds": [],
            })
        if len(prepared) != len(DTYPES):
            failures.append("No timing began because complete 4-dtype preflight did not pass")

    try:
        _verify_sources(artifacts, hashes, args.hash_checks, "after_all")
    except RuntimeError as error:
        failures.append(str(error))
        (args.output / "source_hash_failure.json").write_text(json.dumps({
            "initial_sha256": hashes,
            "current_sha256": {
                label: hashlib.sha256(path.read_bytes()).hexdigest()
                for label, path in artifacts.items()
            },
            "hash_checks": args.hash_checks,
            "error": str(error),
        }, indent=2) + "\n", encoding="utf-8")
    input_hashes_after: dict[str, str] = {}
    for dtype, digest in input_hashes.items():
        path = args.output / f"input_{dtype}.json"
        try:
            _verify_input(path, digest, args.input_hash_checks, f"after_all_{dtype}")
            input_hashes_after[dtype] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, RuntimeError) as error:
            failures.append(f"{dtype}: {error}")
            if path.is_file():
                input_hashes_after[dtype] = hashlib.sha256(path.read_bytes()).hexdigest()
    action_count = len(groups) * len(ACTIONS)
    preflight_action_count = sum(
        len(group.get("preflight", {}).get("actions_validated", [])) for group in groups)
    preflight_attempted_action_count = sum(
        len(group.get("preflight", {}).get("actions_attempted", [])) for group in groups)
    completed_action_count = sum(
        len(group["actions"]) for group in groups
        if group.get("correctness") == "passed" and len(group.get("rounds", [])) == PAIRED_ROUNDS)
    comparison_count = sum(
        len(group["actions"]) * len(group.get("rounds", [])) * 2 for group in groups)
    if (len(groups) != len(DTYPES) or action_count != 32 or
            preflight_attempted_action_count != 32 or
            (not failures and (preflight_action_count != 32 or
                               completed_action_count != 32 or comparison_count != 192))):
        failures.append("logcumsumexp gradient protocol requires 4 groups, 32 completed actions and 192 comparisons")
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
            "actions_per_group": list(ACTIONS),
            "group_count": len(groups),
            "action_count": action_count,
            "preflight_action_count": preflight_action_count,
            "preflight_attempted_action_count": preflight_attempted_action_count,
            "completed_action_count": completed_action_count,
            "warmup_calls_per_action": WARMUP,
            "samples_per_action_per_round": REPEATS,
            "allocating_calls_per_sample": ITERATIONS,
            "paired_rounds_per_group": PAIRED_ROUNDS,
            "comparisons": comparison_count,
            "fresh_graph_per_call": True,
            "seed": "real i % 5 - 2; complex (i % 5 - 2, (i % 3 - 1) / 4)",
            "input_storage": "three resident matrices; cube and vector are contiguous views of base",
            "preflight": "all dtype/action forward and gradient values, shape, dtype, contiguity, storage independence and repeated values checked before timing",
            "timed_consumer": "native forward and gradient pointers plus compiler memory barrier; PyTorch data_ptr for both allocations",
            "timing_gate": "strict median and p95 win in every paired round",
            "response_schema": {
                "actions": list(ACTIONS),
                "entry_fields": ["observation", "samples_ms", "median_ms", "p95_ms"],
                "observation_fields": sorted(OBSERVATION_FIELDS),
                "complex_values": "objects with real and imag numeric values or exact NaN/+Inf/-Inf category strings",
                "tolerance": {"float32": 1e-6, "complex64": 1e-6,
                              "float64": 1e-12, "complex128": 1e-12},
                "zeros": "signed-zero categories compared exactly",
            },
        },
        "source_sha256": {label: digest for label, digest in hashes.items()
                           if label != "native_runner"},
        "source_sha256_after": {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in artifacts.items() if label != "native_runner"
        },
        "runner_sha256": hashes["native_runner"],
        "runner_sha256_after": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
        "binary_sha256": hashes["native_runner"],
        "binary_sha256_after": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
        "input_sha256": input_hashes,
        "input_sha256_after": input_hashes_after,
        "commands": {
            "build_cwd": str(ROOT),
            "build_command": compile_command,
            "compiler_version": nim_version,
            "invoke_command_template": [str(args.runner), "<input-manifest.json>"],
            "preflight_command_template": [str(args.runner), "<input-manifest.json>", "--preflight"],
        },
        "source_hash_checks": args.hash_checks,
        "input_hash_checks": args.input_hash_checks,
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
        "preflight_attempted_actions": preflight_attempted_action_count,
        "completed_actions": completed_action_count,
        "comparisons": comparison_count,
        "failures": failures,
        "passed": not failures,
        "results": str(result_path),
    }))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
