#!/usr/bin/env python3
"""Benchmark allocating CPU logcumsumexp against pinned PyTorch 2.15.0.

Build from the repository root with:
  nim c -d:release --path:src --out:build/logcumsumexp_runner benchmarks/pytorch_head_to_head/logcumsumexp_runner.nim
Run all 32 mandatory dtype/profile actions with:
  build/parity-oracle/bin/python benchmarks/pytorch_head_to_head/benchmark_logcumsumexp.py
Each action gets 50 warmups and 100 samples of 50 allocating calls in three
alternating backend-order rounds. Every median and p95 must strictly beat
PyTorch in every round (192 comparisons). All dtype/action correctness and
allocation preflight finishes before any competitive timing starts.
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
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/logcumsumexp_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_logcumsumexp.py")
SOURCE_PATHS = tuple(map(Path, (
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
    "tests/parity/reference.json",
    "nim.cfg",
    str(RUNNER_SOURCE), str(SCRIPT_SOURCE),
)))
MANIFEST_FIELDS = (
    "input_dtype", "output_dtype", "rows", "columns", "shape", "values",
    "cube_shape", "cube_values", "vector_shape", "vector_values",
    "rows_head_values", "rows_tail_values", "profiles", "actions",
    "warmup", "repeats", "iterations",
)
OBSERVATION_FIELDS = {
    "input_dtype", "output_dtype", "output_shape", "output_values",
    "output_data_ptr", "output_is_contiguous", "output_disjoint_from_inputs",
    "output_distinct_from_repeat",
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
        raise RuntimeError("logcumsumexp requires three resident matrices and base-sharing cube/vector views")
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
    if payload["cube_values"] != payload["values"] or payload["vector_values"] != payload["values"]:
        raise RuntimeError("logcumsumexp cube/vector manifests must equal base matrix values")
    return sources, payload


def action_spec(name: str) -> tuple[str, int]:
    if name not in PROFILE_AXES:
        raise ValueError(f"Malformed logcumsumexp action: {name}")
    return name, PROFILE_AXES[name]


def expected_shape(name: str) -> list[int]:
    return list(PROFILE_SHAPES[action_spec(name)[0]])


def bind_actions(sources: dict[str, torch.Tensor]) -> dict[str, Callable[[], torch.Tensor]]:
    """Bind public torch.logcumsumexp calls to resident inputs and axes."""
    bound: dict[str, Callable[[], torch.Tensor]] = {}
    input_names = {
        "rows": "base", "rows_head": "rows_head", "rows_tail": "rows_tail",
        "columns": "base", "cube_axis0": "cube", "cube_axis1": "cube",
        "cube_axis2": "cube", "vector": "vector",
    }
    for name in ACTIONS:
        _, axis = action_spec(name)
        value = sources[input_names[name]]
        bound[name] = lambda value=value, axis=axis: torch.logcumsumexp(value, dim=axis)
    return bound


def _storage_range(value: torch.Tensor) -> tuple[int, int]:
    storage = value.untyped_storage()
    start = int(storage.data_ptr())
    end = start + int(storage.nbytes())
    if start <= 0 or end <= start:
        raise ValueError("logcumsumexp preflight encountered empty or invalid tensor storage")
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


def _torch_values(result: torch.Tensor, dtype: str) -> list[Any]:
    flat = result.detach().reshape(-1).tolist()
    if dtype.startswith("complex"):
        return [{"real": _json_number(float(item.real)),
                 "imag": _json_number(float(item.imag))} for item in flat]
    return [_json_number(float(item)) for item in flat]


def _same_scalar(left: Any, right: Any) -> bool:
    if type(left) is str or type(right) is str:
        return type(left) is str and type(right) is str and left == right
    if left == 0.0 and right == 0.0:
        return math.copysign(1.0, float(left)) == math.copysign(1.0, float(right))
    return left == right


def _same_values(left: list[Any], right: list[Any], dtype: str) -> bool:
    if len(left) != len(right):
        return False
    if dtype.startswith("complex"):
        return all(type(a) is dict and type(b) is dict and
                   _same_scalar(a["real"], b["real"]) and
                   _same_scalar(a["imag"], b["imag"])
                   for a, b in zip(left, right, strict=True))
    return all(_same_scalar(a, b) for a, b in zip(left, right, strict=True))


def _validate_torch_output(name: str, result: Any, dtype: str,
                           sources: dict[str, torch.Tensor]) -> None:
    expected_dtype = TORCH_DTYPES[dtype]
    if (type(result) is not torch.Tensor or result.dtype != expected_dtype or
            result.device.type != "cpu" or list(result.shape) != expected_shape(name) or
            not result.is_contiguous() or result.requires_grad):
        raise ValueError(f"{name}: PyTorch output dtype, device, shape, layout, or autograd state mismatch")
    if any(_overlaps(result, source) for source in sources.values()):
        raise ValueError(f"{name}: PyTorch output aliases a resident input")


def _reference_observation(name: str, result: torch.Tensor, repeated: torch.Tensor,
                           dtype: str, sources: dict[str, torch.Tensor]) -> dict[str, Any]:
    _validate_torch_output(name, result, dtype, sources)
    _validate_torch_output(name, repeated, dtype, sources)
    if _overlaps(result, repeated):
        raise ValueError(f"{name}: repeated PyTorch outputs alias each other")
    output_values = _torch_values(result, dtype)
    repeated_values = _torch_values(repeated, dtype)
    if not _same_values(output_values, repeated_values, dtype):
        raise ValueError(f"{name}: repeated PyTorch outputs differ, including nonfinite or signed-zero categories")
    return {
        "input_dtype": dtype,
        "output_dtype": str(result.dtype).removeprefix("torch."),
        "output_shape": list(result.shape),
        "output_values": output_values,
        "output_data_ptr": str(result.data_ptr()),
        "output_is_contiguous": result.is_contiguous(),
        "output_disjoint_from_inputs": all(not _overlaps(result, source) for source in sources.values()),
        "output_distinct_from_repeat": not _overlaps(result, repeated),
    }


def _expected_result(name: str, call: Callable[[], torch.Tensor], dtype: str,
                     sources: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    result, repeated = call(), call()
    observation = _reference_observation(name, result, repeated, dtype, sources)
    return result, observation


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
        raise ValueError(f"{name}: output value count does not match exact output shape")
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
                          reference: dict[str, Any], dtype: str) -> None:
    if type(observation) is not dict or set(observation) != OBSERVATION_FIELDS:
        raise ValueError(f"{name}: malformed logcumsumexp observation fields")
    shape = observation["output_shape"]
    if (observation["input_dtype"] != dtype or observation["output_dtype"] != dtype or
            type(shape) is not list or any(type(dimension) is not int for dimension in shape) or
            shape != expected_shape(name) or shape != reference["output_shape"] or
            observation["output_is_contiguous"] is not True or
            observation["output_disjoint_from_inputs"] is not True or
            observation["output_distinct_from_repeat"] is not True):
        raise ValueError(f"{name}: dtype, shape, layout, or independent-allocation mismatch")
    pointer = observation["output_data_ptr"]
    if type(pointer) is not str or not pointer.isdecimal() or int(pointer) <= 0:
        raise ValueError(f"{name}: output_data_ptr must expose a nonzero tensor data pointer")
    _check_values(name, observation["output_values"], reference["output_values"], dtype)


def _check_input_stability(sources: dict[str, torch.Tensor], payload: dict[str, Any], dtype: str) -> None:
    for source, field in (("base", "values"), ("rows_head", "rows_head_values"),
                          ("rows_tail", "rows_tail_values"), ("cube", "cube_values"),
                          ("vector", "vector_values")):
        if tensor_values(sources[source], dtype) != payload[field]:
            raise RuntimeError(f"Resident input changed during logcumsumexp benchmark: {source}")


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


def _verify_sources(artifacts: dict[str, Path], source_hashes: dict[str, str],
                    hash_checks: list[dict[str, Any]] | None = None,
                    stage: str = "") -> None:
    current = {label: hashlib.sha256(path.read_bytes()).hexdigest()
               for label, path in artifacts.items()}
    if hash_checks is not None:
        hash_checks.append({"stage": stage, "sha256": current})
    changed = [label for label, digest in current.items() if digest != source_hashes[label]]
    if changed:
        raise RuntimeError("Sources or native runner changed during measurement: " + ", ".join(changed))


def _invoke_native(args: argparse.Namespace, dtype: str, input_path: Path, input_hash: str,
                   artifacts: dict[str, Path], source_hashes: dict[str, str],
                   references: dict[str, dict[str, Any]], preflight: bool,
                   invocation_index: int) -> dict[str, Any]:
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != input_hash:
        raise RuntimeError(f"{dtype}: input manifest changed before native invocation")
    phase = "preflight" if preflight else "timing"
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
    try:
        if completed.returncode != 0:
            raise ValueError(f"native runner exited with status {completed.returncode}")
        _verify_sources(artifacts, source_hashes, args.hash_checks,
                        f"after_native_{phase}_{dtype}_{invocation_index}")
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != input_hash:
            raise ValueError(f"{dtype}: input manifest changed during native invocation")
        report = json.loads(completed.stdout)
        if type(report) is not dict or set(report) != set(ACTIONS):
            raise ValueError("native action set differs from the strict manifest")
        stem.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in ACTIONS:
            if preflight:
                if name in references:
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
    callables = bind_actions(sources)
    manifest = {
        "input_dtype": dtype,
        "output_dtype": dtype,
        **payload,
        "profiles": list(PROFILES),
        "actions": list(ACTIONS),
        "warmup": WARMUP,
        "repeats": REPEATS,
        "iterations": ITERATIONS,
    }
    input_path = args.output / f"input_{dtype}.json"
    input_path.write_text(json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    references: dict[str, dict[str, Any]] = {}
    reference_failures = []
    _verify_sources(artifacts, source_hashes, args.hash_checks,
                    f"before_pytorch_preflight_{dtype}")
    for name in ACTIONS:
        try:
            _, references[name] = _expected_result(name, callables[name], dtype, sources)
        except (RuntimeError, ValueError, TypeError, AssertionError) as error:
            reference_failures.append({"action": name, "error": str(error)})
    _verify_sources(artifacts, source_hashes, args.hash_checks,
                    f"after_pytorch_preflight_{dtype}")
    _check_input_stability(sources, payload, dtype)
    preflight_path = args.output / f"preflight_{dtype}.json"
    native_preflight = None
    native_failure = None
    try:
        native_preflight = _invoke_native(
            args, dtype, input_path, input_hash, artifacts, source_hashes,
            references, True, invocation_index)
    except (RuntimeError, ValueError, TypeError, AssertionError) as error:
        native_failure = str(error)
    _check_input_stability(sources, payload, dtype)
    preflight = {
        "input_dtype": dtype,
        "input_sha256": input_hash,
        "actions_attempted": list(ACTIONS),
        "actions_validated": list(ACTIONS) if not reference_failures and not native_failure else [],
        "pytorch": references,
        "pytorch_failures": reference_failures,
        "arraymancer": native_preflight,
        "arraymancer_failure": native_failure,
    }
    preflight_path.write_text(json.dumps(preflight, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    if reference_failures or native_failure:
        errors = [f"{item['action']}: {item['error']}" for item in reference_failures]
        if native_failure:
            errors.append(native_failure)
        raise RuntimeError(f"{dtype}: correctness preflight failed: " + "; ".join(errors))
    return {
        "input_dtype": dtype,
        "sources": sources,
        "payload": payload,
        "callables": callables,
        "references": references,
        "input_path": input_path,
        "input_hash": input_hash,
        "native_preflight": native_preflight,
        "preflight": preflight,
    }


def _consume_torch(result: torch.Tensor) -> None:
    """Observe the eager allocation without timing an additional operation."""
    _ = result.data_ptr()


def _time_reference(callables: dict[str, Callable[[], torch.Tensor]]) -> dict[str, Any]:
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
        _check_input_stability(prepared["sources"], prepared["payload"], dtype)
        result = _time_reference(prepared["callables"])
        _check_input_stability(prepared["sources"], prepared["payload"], dtype)
        _verify_sources(artifacts, source_hashes, args.hash_checks, f"after_{label}")
        return result

    for index in range(PAIRED_ROUNDS):
        order = "pytorch-first" if index % 2 == 0 else "arraymancer-first"
        if index % 2 == 0:
            pytorch_result = time_pytorch(f"pytorch_{dtype}_round_{index + 1}")
            native_result = _invoke_native(
                args, dtype, prepared["input_path"], prepared["input_hash"], artifacts,
                source_hashes, prepared["references"], False, invocation_index + index)
        else:
            native_result = _invoke_native(
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
        "preflight": prepared["preflight"],
        "rounds": rounds,
        "timing_failures": timing_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/logcumsumexp_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/logcumsumexp_benchmark"))
    args = parser.parse_args()
    args.hash_checks = []
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

    # Finish every dtype/action output, shape, dtype, and storage preflight
    # before beginning any competitive timing.
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
            preflight_path = args.output / f"preflight_{dtype}.json"
            failed_preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else None
            rounds_path = args.output / f"rounds_{dtype}.json"
            failed_rounds = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.is_file() else []
            failure_path = args.output / f"group_failure_{dtype}.json"
            failure_path.write_text(json.dumps({
                "input_dtype": dtype,
                "output_dtype": dtype,
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "error": str(error),
                "preflight": failed_preflight,
                "rounds": failed_rounds,
            }, indent=2) + "\n", encoding="utf-8")
            groups.append({
                "input_dtype": dtype,
                "output_dtype": dtype,
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "correctness": "failed",
                "error": str(error),
                "preflight": failed_preflight,
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
    for dtype, digest in input_hashes.items():
        path = args.output / f"input_{dtype}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(f"Input manifest changed during measurement: {dtype}")
    action_count = len(groups) * len(ACTIONS)
    preflight_reports = [
        json.loads((args.output / f"preflight_{dtype}.json").read_text(encoding="utf-8"))
        for dtype in DTYPES if (args.output / f"preflight_{dtype}.json").is_file()
    ]
    preflight_action_count = sum(len(item.get("actions_validated", [])) for item in preflight_reports)
    preflight_attempted_action_count = sum(len(item.get("actions_attempted", [])) for item in preflight_reports)
    completed_action_count = sum(
        len(group["actions"]) for group in groups
        if group.get("correctness") == "passed" and len(group.get("rounds", [])) == PAIRED_ROUNDS)
    comparison_count = sum(
        len(group["actions"]) * len(group.get("rounds", [])) * 2 for group in groups)
    if (len(groups) != len(DTYPES) or action_count != 32 or preflight_action_count != 32 or
            completed_action_count != 32 or comparison_count != 192):
        failures.append("logcumsumexp protocol requires 4 groups, 32 completed actions and 192 comparisons")
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
            "preflight": "all dtype/action values, shapes, dtypes, contiguity and storage independence checked before timing",
            "input_storage": "three resident matrices; cube and vector are contiguous views of base",
            "timed_consumer": "native output data pointer plus compiler memory barrier; PyTorch output data_ptr only",
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
        "commands": {
            "build_cwd": str(ROOT),
            "build_command": compile_command,
            "compiler_version": nim_version,
            "invoke_command_template": [str(args.runner), "<input-manifest.json>"],
            "preflight_command_template": [str(args.runner), "<input-manifest.json>", "--preflight"],
        },
        "source_hash_checks": args.hash_checks,
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
