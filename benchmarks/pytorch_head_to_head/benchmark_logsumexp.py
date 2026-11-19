#!/usr/bin/env python3
"""Benchmark allocating CPU logsumexp against pinned PyTorch 2.15.0.

Build from the repository root with:
  nim c -d:release --path:src --out:build/logsumexp_runner benchmarks/pytorch_head_to_head/logsumexp_runner.nim
Run all 234 actions with:
  build/parity-oracle/bin/python benchmarks/pytorch_head_to_head/benchmark_logsumexp.py
Each action gets 50 warmups and 100 samples of 50 allocating calls in three
alternating backend-order rounds. Every median and p95 must strictly beat
PyTorch in every round (1,404 comparisons). All actions are preflighted before
timing starts; inputs are resident and reused, and every result is consumed.
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

DTYPES = (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float32", "float64", "complex64", "complex128",
)
TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
    "uint8": torch.uint8, "uint16": torch.uint16, "uint32": torch.uint32, "uint64": torch.uint64,
    "float32": torch.float32, "float64": torch.float64,
    "complex64": torch.complex64, "complex128": torch.complex128,
}
OUTPUT_DTYPES = {
    dtype: ("float32" if dtype == "bool" or dtype.startswith(("int", "uint")) else dtype)
    for dtype in DTYPES
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
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/logsumexp_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_logsumexp.py")
SOURCE_PATHS = tuple(map(Path, (
    "src/arraymancer/tensor/private/p_logsumexp.nim",
    "src/arraymancer/tensor/private/p_extrema.nim",
    "src/arraymancer/tensor/private/p_accelerate.nim",
    "src/arraymancer/tensor/data_structure.nim",
    "src/arraymancer/tensor/init_cpu.nim",
    "src/arraymancer/tensor/aggregate.nim",
    "src/arraymancer.nim",
    "nim.cfg",
    str(RUNNER_SOURCE), str(SCRIPT_SOURCE),
)))
MANIFEST_FIELDS = (
    "input_dtype", "output_dtype", "rows", "columns", "shape", "values",
    "cube_shape", "cube_values", "rows_head_values", "rows_tail_values",
    "profiles", "actions", "warmup", "repeats", "iterations",
)
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


def source_value(dtype: str, index: int, profile: int) -> bool | int | float | complex:
    column = index % COLUMNS
    if dtype == "bool":
        if profile == 1:
            return column != COLUMNS - 1
        if profile == 2:
            return column == COLUMNS - 1
        return index % 3 != 0
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
    if dtype.startswith("float"):
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
    if dtype.startswith("uint"):
        if profile == 1:
            if column == 0:
                return 64
            if column == COLUMNS - 1:
                return 0
        elif profile == 2:
            if column == 0:
                return 0
            if column == COLUMNS - 1:
                return 64
        return index % 63 + 1
    if profile == 1:
        if column == 0:
            return 64
        if column == COLUMNS - 1:
            return -64
    elif profile == 2:
        if column == 0:
            return -64
        if column == COLUMNS - 1:
            return 64
    return index % 63 - 31


def source_tensor_values(value: torch.Tensor, dtype: str) -> list[Any]:
    flat = value.reshape(-1).tolist()
    if dtype.startswith("complex"):
        return [[float(item.real), float(item.imag)] for item in flat]
    return flat


def make_sources(dtype: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create three independent resident matrices and a base-sharing cube view."""
    tensor_dtype = TORCH_DTYPES[dtype]
    sources: dict[str, torch.Tensor] = {}
    for name, profile in (("base", 0), ("rows_head", 1), ("rows_tail", 2)):
        values = [source_value(dtype, index, profile) for index in range(ROWS * COLUMNS)]
        matrix = torch.tensor(values, dtype=tensor_dtype).reshape(ROWS, COLUMNS)
        sources[name] = matrix
    sources["cube"] = sources["base"].reshape(CUBE_SHAPE)
    matrix, rows_head, rows_tail, cube = (
        sources["base"], sources["rows_head"], sources["rows_tail"], sources["cube"])
    if (len({matrix.data_ptr(), rows_head.data_ptr(), rows_tail.data_ptr()}) != 3 or
            cube.data_ptr() != matrix.data_ptr() or
            not all(value.is_contiguous() for value in sources.values())):
        raise RuntimeError("logsumexp requires independent resident matrices and a contiguous base-sharing cube view")
    payload: dict[str, Any] = {
        "rows": ROWS,
        "columns": COLUMNS,
        "shape": [ROWS, COLUMNS],
        "values": source_tensor_values(matrix, dtype),
        "cube_shape": list(CUBE_SHAPE),
        "cube_values": source_tensor_values(cube, dtype),
        "rows_head_values": source_tensor_values(rows_head, dtype),
        "rows_tail_values": source_tensor_values(rows_tail, dtype),
    }
    if payload["cube_values"] != payload["values"]:
        raise RuntimeError("logsumexp cube manifest must equal base matrix values")
    return sources, payload


def action_spec(name: str) -> tuple[bool, str, int | None]:
    mode, profile = name.split("_", 1)
    if mode not in ("flat", "keepdim") or profile not in PROFILES:
        raise ValueError(f"Malformed logsumexp action: {name}")
    return mode == "keepdim", profile, PROFILE_AXES.get(profile)


def bind_actions(sources: dict[str, torch.Tensor]) -> dict[str, Callable[[], torch.Tensor]]:
    """Bind allocating torch.logsumexp calls, inputs, dimensions and keepdim."""
    bound: dict[str, Callable[[], torch.Tensor]] = {}
    for name in ACTIONS:
        keepdim, profile, axis = action_spec(name)
        value = sources[{
            "whole_matrix": "base", "whole_cube": "cube",
            "rows": "base", "rows_head": "rows_head", "rows_tail": "rows_tail",
            "columns": "base", "cube_axis0": "cube", "cube_axis1": "cube",
            "cube_axis2": "cube",
        }[profile]]
        if axis is None:
            dimensions = tuple(range(value.dim()))
            bound[name] = lambda value=value, dimensions=dimensions, keepdim=keepdim: torch.logsumexp(
                value, dim=dimensions, keepdim=keepdim)
        else:
            bound[name] = lambda value=value, axis=axis, keepdim=keepdim: torch.logsumexp(
                value, dim=axis, keepdim=keepdim)
    return bound


def _storage_range(value: torch.Tensor) -> tuple[int, int]:
    storage = value.untyped_storage()
    start = int(storage.data_ptr())
    end = start + int(storage.nbytes())
    if start <= 0 or end <= start:
        raise ValueError("logsumexp preflight encountered empty or invalid tensor storage")
    return start, end


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _storage_range(left)
    right_start, right_end = _storage_range(right)
    return left_start < right_end and right_start < left_end


def _expected_shape(name: str) -> list[int]:
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


def _validate_torch_output(name: str, result: Any, dtype: str,
                           sources: dict[str, torch.Tensor]) -> None:
    expected_dtype = TORCH_DTYPES[OUTPUT_DTYPES[dtype]]
    if (type(result) is not torch.Tensor or result.dtype != expected_dtype or
            result.device.type != "cpu" or list(result.shape) != _expected_shape(name) or
            not result.is_contiguous() or result.requires_grad):
        raise ValueError(f"{name}: PyTorch output dtype, device, shape, layout, or autograd state mismatch")
    if any(_overlaps(result, source) for source in sources.values()):
        raise ValueError(f"{name}: PyTorch output aliases a resident input")


def _torch_values(result: torch.Tensor, output_dtype: str) -> list[Any]:
    flat = result.reshape(-1).tolist()
    if output_dtype.startswith("complex"):
        return [{"real": float(item.real), "imag": float(item.imag)} for item in flat]
    return flat


def _retain_native_failure(args: argparse.Namespace, dtype: str, index: int,
                           phase: str, returncode: int | None,
                           stdout: str, stderr: str, error: str) -> Path:
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


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _close(actual: float, expected: float, tolerance: float) -> bool:
    return math.isfinite(actual) and math.isfinite(expected) and math.isclose(
        actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def _check_json_values(name: str, actual: Any, expected: list[Any], output_dtype: str) -> None:
    if type(actual) is not list or len(actual) != len(expected):
        raise ValueError(f"{name}: output value count does not match exact output shape")
    tolerance = 1e-6 if output_dtype in ("float32", "complex64") else 1e-12
    if output_dtype.startswith("complex"):
        for item, reference in zip(actual, expected, strict=True):
            if (type(item) is not dict or set(item) != {"real", "imag"} or
                    any(type(item[key]) not in (int, float) or not math.isfinite(item[key])
                        for key in ("real", "imag")) or
                    not _close(float(item["real"]), float(reference["real"]), tolerance) or
                    not _close(float(item["imag"]), float(reference["imag"]), tolerance)):
                raise ValueError(f"{name}: native complex output does not match PyTorch")
    else:
        for item, reference in zip(actual, expected, strict=True):
            if (type(item) not in (int, float) or not math.isfinite(item) or
                    not _close(float(item), float(reference), tolerance)):
                raise ValueError(f"{name}: native output does not match PyTorch within {tolerance:g}")


def _validate_observation(name: str, observation: Any, reference: torch.Tensor,
                          dtype: str) -> None:
    fields = {
        "input_dtype", "output_dtype", "output_shape", "output_values",
        "output_is_contiguous", "output_disjoint_from_inputs", "output_distinct_from_repeat",
    }
    if type(observation) is not dict or set(observation) != fields:
        raise ValueError(f"{name}: malformed logsumexp observation fields")
    output_dtype = OUTPUT_DTYPES[dtype]
    shape = observation["output_shape"]
    if (type(shape) is not list or any(type(dimension) is not int for dimension in shape) or
            observation["input_dtype"] != dtype or observation["output_dtype"] != output_dtype or
            shape != _expected_shape(name) or observation["output_is_contiguous"] is not True or
            observation["output_disjoint_from_inputs"] is not True or
            observation["output_distinct_from_repeat"] is not True):
        raise ValueError(f"{name}: dtype, shape, layout, or independent-allocation mismatch")
    _check_json_values(name, observation["output_values"], _torch_values(reference, output_dtype), output_dtype)


def _expected_result(name: str, call: Callable[[], torch.Tensor], dtype: str,
                     sources: dict[str, torch.Tensor]) -> torch.Tensor:
    result, repeated = call(), call()
    _validate_torch_output(name, result, dtype, sources)
    _validate_torch_output(name, repeated, dtype, sources)
    if _overlaps(result, repeated):
        raise ValueError(f"{name}: repeated PyTorch outputs alias each other")
    return result


def _check_input_stability(sources: dict[str, torch.Tensor], payload: dict[str, Any], dtype: str) -> None:
    for source, field in (("base", "values"), ("rows_head", "rows_head_values"),
                          ("rows_tail", "rows_tail_values"), ("cube", "cube_values")):
        if source_tensor_values(sources[source], dtype) != payload[field]:
            raise RuntimeError(f"Resident input changed during logsumexp benchmark: {source}")


def _check_timing_record(name: str, entry: Any) -> None:
    if type(entry) is not dict or set(entry) != {"observation", "samples_ms", "median_ms", "p95_ms"}:
        raise ValueError(f"{name}: malformed timing record")
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


def _verify_sources(artifacts: dict[str, Path], source_hashes: dict[str, str]) -> None:
    changed = [label for label, path in artifacts.items()
               if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[label]]
    if changed:
        raise RuntimeError("Sources or native runner changed during measurement: " + ", ".join(changed))


def _invoke_native(args: argparse.Namespace, dtype: str, input_path: Path, input_hash: str,
                   artifacts: dict[str, Path], source_hashes: dict[str, str],
                   references: dict[str, torch.Tensor], preflight: bool,
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
        stem.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if type(report) is not dict or set(report) != set(ACTIONS):
            raise ValueError("native action set differs from strict manifest")
        for name in ACTIONS:
            if preflight:
                _validate_observation(name, report[name], references[name], dtype)
            else:
                _check_timing_record(name, report[name])
                _validate_observation(name, report[name]["observation"], references[name], dtype)
        return report
    except (KeyError, TypeError, ValueError, AssertionError) as error:
        path = _retain_native_failure(
            args, dtype, invocation_index, phase, completed.returncode,
            completed.stdout, completed.stderr, str(error))
        raise RuntimeError(f"{dtype}: invalid native {phase} report retained at {path}: {error}") from error


def _prepare_group(args: argparse.Namespace, dtype: str,
                   artifacts: dict[str, Path], source_hashes: dict[str, str],
                   invocation_counts: dict[str, int]) -> dict[str, Any]:
    sources, payload = make_sources(dtype)
    callables = bind_actions(sources)
    manifest = {
        "input_dtype": dtype,
        "output_dtype": OUTPUT_DTYPES[dtype],
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
    references = {
        name: _expected_result(name, callables[name], dtype, sources)
        for name in ACTIONS
    }
    _check_input_stability(sources, payload, dtype)
    invocation_index = invocation_counts[dtype]
    invocation_counts[dtype] += 1
    native_preflight = _invoke_native(
        args, dtype, input_path, input_hash, artifacts, source_hashes,
        references, True, invocation_index)
    _check_input_stability(sources, payload, dtype)
    return {
        "input_dtype": dtype,
        "output_dtype": OUTPUT_DTYPES[dtype],
        "sources": sources,
        "payload": payload,
        "callables": callables,
        "references": references,
        "input_path": input_path,
        "input_hash": input_hash,
        "native_preflight": native_preflight,
    }


def _consume_torch(result: torch.Tensor) -> None:
    """Expose the eager CPU allocation without timing an additional reduction."""
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
                invocation_counts: dict[str, int]) -> dict[str, Any]:
    dtype = prepared["input_dtype"]
    rounds = []
    for index in range(PAIRED_ROUNDS):
        invocation_index = invocation_counts[dtype]
        invocation_counts[dtype] += 1
        order = "pytorch-first" if index % 2 == 0 else "arraymancer-first"
        if index % 2 == 0:
            pytorch_result = _time_reference(prepared["callables"])
            native_result = _invoke_native(
                args, dtype, prepared["input_path"], prepared["input_hash"], artifacts,
                source_hashes, prepared["references"], False, invocation_index)
        else:
            native_result = _invoke_native(
                args, dtype, prepared["input_path"], prepared["input_hash"], artifacts,
                source_hashes, prepared["references"], False, invocation_index)
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
                    failures.append(f"{dtype}: round {index + 1} {name} {metric} did not win")
        _check_input_stability(prepared["sources"], prepared["payload"], dtype)
    return {
        "input_dtype": dtype,
        "output_dtype": OUTPUT_DTYPES[dtype],
        "actions": list(ACTIONS),
        "input_sha256": prepared["input_hash"],
        "correctness": "passed",
        "native_preflight": "passed",
        "rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/logsumexp_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/logsumexp_benchmark"))
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

    # Finish dtype, shape, value, output-dtype and allocation preflight for all
    # 234 actions before beginning any competitive timing.
    for dtype in DTYPES:
        try:
            item = _prepare_group(args, dtype, artifacts, hashes, invocation_counts)
            prepared.append(item)
            input_hashes[dtype] = item["input_hash"]
        except (RuntimeError, ValueError, TypeError) as error:
            failures.append(str(error))
            input_path = args.output / f"input_{dtype}.json"
            if input_path.is_file():
                input_hashes[dtype] = hashlib.sha256(input_path.read_bytes()).hexdigest()
            rounds_path = args.output / f"rounds_{dtype}.json"
            failed_rounds = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.is_file() else []
            failure_path = args.output / f"group_failure_{dtype}.json"
            failure_path.write_text(json.dumps({
                "input_dtype": dtype,
                "output_dtype": OUTPUT_DTYPES[dtype],
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "error": str(error),
                "rounds": failed_rounds,
            }, indent=2) + "\n", encoding="utf-8")
            groups.append({
                "input_dtype": dtype,
                "output_dtype": OUTPUT_DTYPES[dtype],
                "input_sha256": input_hashes.get(dtype),
                "actions": list(ACTIONS),
                "correctness": "failed",
                "error": str(error),
                "rounds": failed_rounds,
            })

    if not failures:
        for item in prepared:
            try:
                groups.append(_time_group(args, item, failures, artifacts, hashes, invocation_counts))
            except (RuntimeError, ValueError, TypeError) as error:
                dtype = item["input_dtype"]
                failures.append(str(error))
                rounds_path = args.output / f"rounds_{dtype}.json"
                failed_rounds = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.is_file() else []
                failure_path = args.output / f"group_failure_{dtype}.json"
                failure_path.write_text(json.dumps({
                    "input_dtype": dtype,
                    "output_dtype": OUTPUT_DTYPES[dtype],
                    "input_sha256": item["input_hash"],
                    "actions": list(ACTIONS),
                    "error": str(error),
                    "rounds": failed_rounds,
                }, indent=2) + "\n", encoding="utf-8")
                groups.append({
                    "input_dtype": dtype,
                    "output_dtype": OUTPUT_DTYPES[dtype],
                    "input_sha256": item["input_hash"],
                    "actions": list(ACTIONS),
                    "correctness": "failed",
                    "error": str(error),
                    "rounds": failed_rounds,
                })
    else:
        prepared_dtypes = {item["input_dtype"] for item in prepared}
        for item in prepared:
            dtype = item["input_dtype"]
            groups.append({
                "input_dtype": dtype,
                "output_dtype": OUTPUT_DTYPES[dtype],
                "actions": list(ACTIONS),
                "input_sha256": item["input_hash"],
                "correctness": "preflight_passed_timing_skipped",
                "native_preflight": "passed",
                "timing_skipped": "at least one dtype failed global preflight",
                "rounds": [],
            })
        if len(prepared_dtypes) != len(DTYPES):
            failures.append("No timing began because complete 13-dtype preflight did not pass")

    _verify_sources(artifacts, hashes)
    for dtype, digest in input_hashes.items():
        path = args.output / f"input_{dtype}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(f"Input manifest changed during measurement: {dtype}")
    action_count = len(groups) * len(ACTIONS)
    completed_action_count = sum(
        len(group["actions"]) for group in groups
        if group.get("correctness") == "passed" and len(group.get("rounds", [])) == PAIRED_ROUNDS)
    comparison_count = sum(
        len(group["actions"]) * len(group.get("rounds", [])) * 2 for group in groups)
    if (len(groups) != len(DTYPES) or action_count != 234 or
            completed_action_count != 234 or comparison_count != 1404):
        failures.append("logsumexp protocol requires 13 groups, 234 completed actions and 1,404 comparisons")
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
            "output_dtypes": OUTPUT_DTYPES,
            "profiles": list(PROFILES),
            "keepdim_values": list(KEEPDIMS),
            "actions_per_group": list(ACTIONS),
            "group_count": len(groups),
            "action_count": action_count,
            "completed_action_count": completed_action_count,
            "warmup_calls_per_action": WARMUP,
            "samples_per_action_per_round": REPEATS,
            "allocating_calls_per_sample": ITERATIONS,
            "paired_rounds_per_group": PAIRED_ROUNDS,
            "comparisons": comparison_count,
            "preflight": "all dtype/action outputs checked before competitive timing",
            "request_schema": {
                "fields": list(MANIFEST_FIELDS),
                "integer_values": "exact JSON integers, cast to declared source dtype",
                "boolean_values": "JSON booleans",
                "complex_values": "[real, imag] pairs cast to declared complex source dtype",
                "cube_values": "required and equal to base values",
            },
            "response_schema": {
                "actions": list(ACTIONS),
                "entry_fields": ["observation", "samples_ms", "median_ms", "p95_ms"],
                "observation_fields": [
                    "input_dtype", "output_dtype", "output_shape", "output_values",
                    "output_is_contiguous", "output_disjoint_from_inputs",
                    "output_distinct_from_repeat",
                ],
                "complex_output_values": "objects with finite real and imag numbers",
                "finite_tolerance": {"float32": 1e-6, "complex64": 1e-6,
                                     "float64": 1e-12, "complex128": 1e-12},
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
        "completed_actions": completed_action_count,
        "comparisons": comparison_count,
        "failures": failures,
        "passed": not failures,
        "results": str(result_path),
    }))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
