#!/usr/bin/env python3
"""Benchmark allocating CPU cumulative min/max against pinned PyTorch 2.15.0.

Build from the repository root with:
  nim c -d:ssl --cc:clang --mm:orc -d:release --path:src --out:build/cumulative_extrema_runner benchmarks/pytorch_head_to_head/cumulative_extrema_runner.nim
Run the full mandatory matrix with:
  build/parity-oracle/bin/python benchmarks/pytorch_head_to_head/benchmark_cumulative_extrema.py
The 244 allocating actions receive 50 warmups and 100 samples of 500 calls in
three alternating backend-order rounds (1,464 strict median/p95 comparisons).
Every dtype/action and all input/output allocation checks finish before any
competitive timing starts. No trial, smoke, or partial run is canonical proof.
"""
from __future__ import annotations

import argparse
import gzip
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

DTYPES = (
    "bool", "uint8", "int8", "int16", "int32", "int64", "float32", "float64",
    "uint16", "uint32", "uint64", "complex64", "complex128",
)
TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
OPERATIONS = ("cummin", "cummax")
NONEMPTY_DTYPES = DTYPES[:8]
NONEMPTY_PROFILES = (
    "rows", "rows_head", "rows_tail", "columns",
    "cube_axis0", "cube_axis1", "cube_axis2", "vector",
    "transpose", "offset", "reverse", "broadcast",
)
SPECIAL_PROFILES = ("scalar", "empty")
PROFILES = NONEMPTY_PROFILES + SPECIAL_PROFILES
TORCH_VERSION = "2.15.0"
TORCH_GIT_REVISION = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
PYTHON_VERSION = (3, 13, 18)
ROWS, COLUMNS, CUBE_SHAPE = 128, 128, (16, 32, 32)
VECTOR_SHAPE = (ROWS * COLUMNS,)
OFFSET_SHAPE = (ROWS, COLUMNS + 2)
EMPTY_SHAPE = (2, 0, 3)
# Both backends use millisecond-scale batches of amortized allocating calls.
WARMUP, REPEATS, ITERATIONS, PAIRED_ROUNDS = 50, 100, 500, 3
ROOT = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/cumulative_extrema_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_cumulative_extrema.py")
SOURCE_PATHS = tuple(map(Path, (
    "src/arraymancer/laser/compiler_optim_hints.nim",
    "src/arraymancer/tensor/private/p_cumulative_extrema.nim",
    "src/arraymancer/tensor/private/p_extrema.nim",
    "src/arraymancer/tensor/aggregate.nim",
    "src/arraymancer/tensor/data_structure.nim",
    "src/arraymancer/tensor/private/p_accessors.nim",
    "src/arraymancer/tensor/private/p_accessors_macros_read.nim",
    "src/arraymancer/tensor/private/p_accessors_macros_write.nim",
    "src/arraymancer/tensor/private/p_shapeshifting.nim",
    "src/arraymancer/tensor/private/p_empty_tensors.nim",
    "src/arraymancer/tensor/private/p_checks.nim",
    "src/arraymancer/tensor/shapeshifting.nim",
    "src/arraymancer/tensor/selectors.nim",
    "src/arraymancer/tensor/accessors.nim",
    "src/arraymancer/tensor/init_cpu.nim",
    "src/arraymancer/tensor/backend/global_config.nim",
    "src/arraymancer/laser/tensor/initialization.nim",
    "src/arraymancer/laser/tensor/datatypes.nim",
    "src/arraymancer/laser/private/memory.nim",
    "src/arraymancer/tensor.nim",
    "src/arraymancer.nim",
    "tests/parity/reference.json",
    "nim.cfg",
    str(RUNNER_SOURCE), str(SCRIPT_SOURCE),
)))
MANIFEST_FIELDS = (
    "input_dtype", "values_dtype", "indices_dtype", "rows", "columns", "shape",
    "cube_shape", "vector_shape", "offset_backing_shape", "scalar_shape", "scalar_value",
    "empty_shape", "values", "rows_head_values", "rows_tail_values",
    "offset_backing_values", "profiles", "actions", "profile_value_counts",
    "profile_layouts", "warmup", "repeats", "iterations",
)
OBSERVATION_FIELDS = {
    "input_dtype", "values_dtype", "indices_dtype", "values_shape", "indices_shape",
    "values", "indices", "values_bits", "values_data_ptr", "indices_data_ptr",
    "values_contiguous", "indices_contiguous", "values_disjoint_from_inputs",
    "indices_disjoint_from_inputs", "values_indices_independent",
    "peer_independent", "repeat_independent",
}
PROFILE_AXES = {
    "rows": 1, "rows_head": 1, "rows_tail": 1, "columns": 0,
    "cube_axis0": 0, "cube_axis1": 1, "cube_axis2": 2, "vector": 0,
    "transpose": 1, "offset": 1, "reverse": 1, "broadcast": 0,
    "scalar": 0, "empty": 1,
}
PROFILE_SHAPES = {
    "rows": (ROWS, COLUMNS), "rows_head": (ROWS, COLUMNS),
    "rows_tail": (ROWS, COLUMNS), "columns": (ROWS, COLUMNS),
    "cube_axis0": CUBE_SHAPE, "cube_axis1": CUBE_SHAPE, "cube_axis2": CUBE_SHAPE,
    "vector": VECTOR_SHAPE, "transpose": (COLUMNS, ROWS),
    "offset": (ROWS, COLUMNS), "reverse": (ROWS, COLUMNS),
    "broadcast": (ROWS, COLUMNS), "scalar": (), "empty": EMPTY_SHAPE,
}
PROFILE_TORCH_STRIDES = {
    "rows": (COLUMNS, 1), "rows_head": (COLUMNS, 1),
    "rows_tail": (COLUMNS, 1), "columns": (COLUMNS, 1),
    "cube_axis0": (CUBE_SHAPE[1] * CUBE_SHAPE[2], CUBE_SHAPE[2], 1),
    "cube_axis1": (CUBE_SHAPE[1] * CUBE_SHAPE[2], CUBE_SHAPE[2], 1),
    "cube_axis2": (CUBE_SHAPE[1] * CUBE_SHAPE[2], CUBE_SHAPE[2], 1),
    "vector": (1,), "transpose": (1, ROWS), "offset": (COLUMNS + 2, 1),
    "reverse": (COLUMNS, 1), "broadcast": (0, 1), "scalar": (),
}
PROFILE_TORCH_OFFSETS = {
    "rows": 0, "rows_head": 0, "rows_tail": 0, "columns": 0,
    "cube_axis0": 0, "cube_axis1": 0, "cube_axis2": 0, "vector": 0,
    "transpose": 0, "offset": 1, "reverse": 0, "broadcast": 0,
    "scalar": 0, "empty": 0,
}
PROFILE_NIM_STRIDES = {
    "rows": (COLUMNS, 1), "rows_head": (COLUMNS, 1),
    "rows_tail": (COLUMNS, 1), "columns": (COLUMNS, 1),
    "cube_axis0": (CUBE_SHAPE[1] * CUBE_SHAPE[2], CUBE_SHAPE[2], 1),
    "cube_axis1": (CUBE_SHAPE[1] * CUBE_SHAPE[2], CUBE_SHAPE[2], 1),
    "cube_axis2": (CUBE_SHAPE[1] * CUBE_SHAPE[2], CUBE_SHAPE[2], 1),
    "vector": (1,), "transpose": (1, ROWS), "offset": (COLUMNS + 2, 1),
    "reverse": (-COLUMNS, -1), "broadcast": (0, 1), "scalar": (),
    "empty": (0, 3, 1),
}
PROFILE_NIM_OFFSETS = {
    "rows": 0, "rows_head": 0, "rows_tail": 0, "columns": 0,
    "cube_axis0": 0, "cube_axis1": 0, "cube_axis2": 0, "vector": 0,
    "transpose": 0, "offset": 1, "reverse": ROWS * COLUMNS - 1,
    "broadcast": 0, "scalar": 0, "empty": 0,
}
PROFILE_TORCH_OWNERS = {
    "rows": "base", "rows_head": "rows_head", "rows_tail": "rows_tail",
    "columns": "base", "cube_axis0": "base", "cube_axis1": "base",
    "cube_axis2": "base", "vector": "base", "transpose": "base",
    "offset": "offset_backing", "reverse": "reverse", "broadcast": "base",
    "scalar": "scalar", "empty": "empty",
}
PROFILE_NIM_OWNERS = {
    "rows": "base", "rows_head": "rows_head", "rows_tail": "rows_tail",
    "columns": "base", "cube_axis0": "base", "cube_axis1": "base",
    "cube_axis2": "base", "vector": "base", "transpose": "base",
    "offset": "offset_backing", "reverse": "base", "broadcast": "base",
    "scalar": "scalar", "empty": "empty",
}
PROFILE_SOURCE_NAMES = {
    "rows": "base", "rows_head": "rows_head", "rows_tail": "rows_tail",
    "columns": "base", "cube_axis0": "cube", "cube_axis1": "cube",
    "cube_axis2": "cube", "vector": "vector", "transpose": "transpose",
    "offset": "offset", "reverse": "reverse", "broadcast": "broadcast",
    "scalar": "scalar", "empty": "empty",
}


def actions_for(dtype: str) -> tuple[str, ...]:
    active_profiles = PROFILES if dtype in NONEMPTY_DTYPES else SPECIAL_PROFILES
    return tuple(f"{operation}_{profile}" for operation in OPERATIONS
                 for profile in active_profiles)


ACTIONS_BY_DTYPE = {dtype: actions_for(dtype) for dtype in DTYPES}
ACTION_COUNT = sum(map(len, ACTIONS_BY_DTYPE.values()))


def _cast_input(dtype: str, value: int | float | complex) -> Any:
    if dtype == "bool":
        return bool(value)
    if dtype.startswith("complex"):
        assert isinstance(value, complex)
        return value
    assert isinstance(value, (int, float))
    if dtype.startswith("float"):
        return float(value)
    return int(value)


def _source_scalar(dtype: str, index: int, profile: str) -> Any:
    column = index % COLUMNS
    if profile == "rows_head":
        if column == 0:
            return _cast_input(dtype, 63)
        if column == COLUMNS - 1:
            return _cast_input(dtype, 0)
    elif profile == "rows_tail":
        if column == 0:
            return _cast_input(dtype, 0)
        if column == COLUMNS - 1:
            return _cast_input(dtype, 63)
    return _cast_input(dtype, index % 31)


def _scalar_source(dtype: str) -> Any:
    if dtype == "bool":
        return True
    if dtype.startswith("complex"):
        return complex(2.0, 0.25)
    return _cast_input(dtype, 3)


def _tensor_payload(tensor: torch.Tensor, dtype: str) -> list[Any]:
    items = tensor.detach().reshape(-1).tolist()
    if dtype.startswith("complex"):
        return [[float(item.real), float(item.imag)] for item in items]
    if dtype == "bool":
        return [bool(item) for item in items]
    if dtype.startswith("float"):
        return [float(item) for item in items]
    return [int(item) for item in items]


def _expected_torch_owner(name: str) -> str:
    return PROFILE_TORCH_OWNERS[name]


def make_sources(dtype: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build all required inputs before timing and attest resident layouts."""
    tensor_dtype = TORCH_DTYPES[dtype]
    special_only = dtype.startswith("complex")
    if special_only:
        scalar = torch.tensor(_scalar_source(dtype), dtype=tensor_dtype)
        empty = torch.empty(EMPTY_SHAPE, dtype=tensor_dtype)
        sources = {"scalar": scalar, "empty": empty}
        values_payload: list[Any] = []
        rows_head_payload: list[Any] = []
        rows_tail_payload: list[Any] = []
        offset_payload: list[Any] = []
    else:
        base = torch.tensor([_source_scalar(dtype, index, "rows")
                             for index in range(ROWS * COLUMNS)], dtype=tensor_dtype).reshape(ROWS, COLUMNS)
        rows_head = torch.tensor([_source_scalar(dtype, index, "rows_head")
                                  for index in range(ROWS * COLUMNS)], dtype=tensor_dtype).reshape(ROWS, COLUMNS)
        rows_tail = torch.tensor([_source_scalar(dtype, index, "rows_tail")
                                  for index in range(ROWS * COLUMNS)], dtype=tensor_dtype).reshape(ROWS, COLUMNS)
        offset_backing = torch.tensor([_cast_input(dtype, index % 31)
                                       for index in range(ROWS * (COLUMNS + 2))],
                                      dtype=tensor_dtype).reshape(OFFSET_SHAPE)
        scalar = torch.tensor(_scalar_source(dtype), dtype=tensor_dtype)
        empty = torch.empty(EMPTY_SHAPE, dtype=tensor_dtype)
        sources = {
            "base": base,
            "rows_head": rows_head,
            "rows_tail": rows_tail,
            "cube": base.reshape(CUBE_SHAPE),
            "vector": base.reshape(VECTOR_SHAPE),
            "transpose": base.transpose(0, 1),
            "offset_backing": offset_backing,
            "offset": offset_backing[:, 1:COLUMNS + 1],
            "reverse": torch.flip(base, dims=(0, 1)),
            "broadcast": base[:1, :].expand(ROWS, COLUMNS),
            "scalar": scalar,
            "empty": empty,
        }
        if (list(sources["reverse"].stride()) != list(PROFILE_TORCH_STRIDES["reverse"]) or
                sources["reverse"].untyped_storage().data_ptr() == base.untyped_storage().data_ptr() or
                list(sources["broadcast"].stride()) != [0, 1] or
                sources["offset"].storage_offset() != 1 or
                list(sources["transpose"].stride()) != [1, ROWS] or
                sources["empty"].numel() != 0):
            raise RuntimeError("PyTorch profile layout or storage ownership mismatch")
        root_names = ("base", "rows_head", "rows_tail", "offset_backing", "scalar", "empty")
        for index, left_name in enumerate(root_names):
            for right_name in root_names[index + 1:]:
                left, right = sources[left_name], sources[right_name]
                if (left.numel() and right.numel() and
                        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()):
                    raise RuntimeError(f"Resident inputs {left_name} and {right_name} must own distinct storage")
        if (sources["cube"].untyped_storage().data_ptr() != base.untyped_storage().data_ptr() or
                sources["vector"].untyped_storage().data_ptr() != base.untyped_storage().data_ptr() or
                sources["transpose"].untyped_storage().data_ptr() != base.untyped_storage().data_ptr() or
                sources["broadcast"].untyped_storage().data_ptr() != base.untyped_storage().data_ptr() or
                sources["offset"].untyped_storage().data_ptr() != offset_backing.untyped_storage().data_ptr()):
            raise RuntimeError("PyTorch resident views do not share their specified backing storage")
        values_payload = _tensor_payload(base, dtype)
        rows_head_payload = _tensor_payload(rows_head, dtype)
        rows_tail_payload = _tensor_payload(rows_tail, dtype)
        offset_payload = _tensor_payload(offset_backing, dtype)
    profile_layouts: dict[str, Any] = {}
    for profile in PROFILES:
        source_name = PROFILE_SOURCE_NAMES[profile]
        expected_shape = list(PROFILE_SHAPES[profile])
        if source_name in sources:
            tensor = sources[source_name]
            torch_strides = list(tensor.stride())
            torch_storage_offset = tensor.storage_offset()
            expected_contiguous = profile not in ("transpose", "offset", "broadcast")
            if (list(tensor.shape) != expected_shape or tensor.dtype != tensor_dtype or
                    tensor.device.type != "cpu" or tensor.is_contiguous() != expected_contiguous or
                    (profile != "empty" and torch_strides != list(PROFILE_TORCH_STRIDES[profile])) or
                    torch_storage_offset != PROFILE_TORCH_OFFSETS[profile]):
                raise RuntimeError(f"PyTorch {profile} input shape, dtype, or layout mismatch")
        else:
            torch_strides = list(PROFILE_TORCH_STRIDES[profile])
            torch_storage_offset = PROFILE_TORCH_OFFSETS[profile]
        profile_layouts[profile] = {
            "shape": expected_shape,
            "axis": PROFILE_AXES[profile],
            "torch_strides": torch_strides,
            "torch_storage_offset": torch_storage_offset,
            "torch_storage_owner": _expected_torch_owner(profile),
            "nim_strides": list(PROFILE_NIM_STRIDES[profile]),
            "nim_storage_offset": PROFILE_NIM_OFFSETS[profile],
            "nim_storage_owner": PROFILE_NIM_OWNERS[profile],
        }
    payload = {
        "rows": ROWS,
        "columns": COLUMNS,
        "shape": [ROWS, COLUMNS],
        "cube_shape": list(CUBE_SHAPE),
        "vector_shape": list(VECTOR_SHAPE),
        "offset_backing_shape": list(OFFSET_SHAPE),
        "scalar_shape": [],
        "scalar_value": _tensor_payload(scalar, dtype)[0],
        "empty_shape": list(EMPTY_SHAPE),
        "values": values_payload,
        "rows_head_values": rows_head_payload,
        "rows_tail_values": rows_tail_payload,
        "offset_backing_values": offset_payload,
        "profiles": list(PROFILES),
        "profile_value_counts": {
            profile: math.prod(PROFILE_SHAPES[profile]) for profile in PROFILES
        },
        "profile_layouts": profile_layouts,
    }
    expected_payload_lengths = (
        0 if special_only else ROWS * COLUMNS,
        0 if special_only else ROWS * COLUMNS,
        0 if special_only else ROWS * COLUMNS,
        0 if special_only else ROWS * (COLUMNS + 2),
    )
    actual_payload_lengths = tuple(
        len(payload[field]) for field in
        ("values", "rows_head_values", "rows_tail_values", "offset_backing_values")
    )
    if actual_payload_lengths != expected_payload_lengths:
        raise RuntimeError("Manifest input value counts do not match the dtype's resident profiles")
    return sources, payload



def bind_actions(sources: dict[str, torch.Tensor], dtype: str) -> dict[str, Callable[[], Any]]:
    bound: dict[str, Callable[[], Any]] = {}
    for name in ACTIONS_BY_DTYPE[dtype]:
        operation, profile, axis, _ = action_spec_for_dtype(name, dtype)
        value = sources[PROFILE_SOURCE_NAMES[profile]]
        call = getattr(torch, operation)
        bound[name] = lambda value=value, axis=axis, call=call: call(value, dim=axis)
    return bound


def action_spec_for_dtype(name: str, dtype: str) -> tuple[str, str, int, str]:
    if name not in ACTIONS_BY_DTYPE[dtype]:
        raise ValueError(f"Unknown action for {dtype}: {name}")
    operation, profile = name.split("_", 1)
    if operation not in OPERATIONS or profile not in PROFILES:
        raise ValueError(f"Malformed cumulative-extrema action: {name}")
    return operation, profile, PROFILE_AXES[profile], profile


def _storage_range(value: torch.Tensor) -> tuple[int, int] | None:
    if value.numel() == 0:
        return None
    storage = value.untyped_storage()
    start = int(storage.data_ptr())
    end = start + int(storage.nbytes())
    if start <= 0 or end <= start:
        raise ValueError("Cumulative-extrema preflight encountered invalid tensor storage")
    return start, end


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_range, right_range = _storage_range(left), _storage_range(right)
    if left_range is None or right_range is None:
        return False
    return left_range[0] < right_range[1] and right_range[0] < left_range[1]


def _json_number(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "+Inf"
    if value == -math.inf:
        return "-Inf"
    return value


def _torch_output_values(value: torch.Tensor, dtype: str) -> list[Any]:
    items = value.detach().reshape(-1).tolist()
    if dtype.startswith("complex"):
        return [{"real": _json_number(float(item.real)),
                 "imag": _json_number(float(item.imag))} for item in items]
    if dtype.startswith("float"):
        return [_json_number(float(item)) for item in items]
    if dtype == "bool":
        return [bool(item) for item in items]
    return [int(item) for item in items]


def _torch_value_bits(value: torch.Tensor) -> list[str]:
    flat = value.detach().contiguous().reshape(-1)
    if value.is_complex():
        components = torch.view_as_real(flat)
        raw = components.reshape(-1).view(torch.uint8).tolist()
        component_bytes = components.element_size()
        words = [
            int.from_bytes(bytes(raw[index:index + component_bytes]), byteorder=sys.byteorder).to_bytes(
                component_bytes, byteorder="big").hex()
            for index in range(0, len(raw), component_bytes)
        ]
        return [words[index] + words[index + 1] for index in range(0, len(words), 2)]
    raw = flat.view(torch.uint8).tolist()
    item_bytes = value.element_size()
    return [int.from_bytes(bytes(raw[index:index + item_bytes]), byteorder=sys.byteorder).to_bytes(
        item_bytes, byteorder="big").hex()
        for index in range(0, len(raw), item_bytes)]


def _expected_shape(profile: str) -> list[int]:
    return list(PROFILE_SHAPES[profile])


def _validate_torch_pair(name: str, pair: Any, dtype: str,
                         sources: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if type(pair) not in (tuple, torch.return_types.cummin, torch.return_types.cummax):
        raise ValueError(f"{name}: PyTorch must return its public values/indices pair")
    values, indices = pair
    profile = name.split("_", 1)[1]
    shape = _expected_shape(profile)
    if (type(values) is not torch.Tensor or type(indices) is not torch.Tensor or
            values.dtype != TORCH_DTYPES[dtype] or indices.dtype != torch.int64 or
            values.device.type != "cpu" or indices.device.type != "cpu" or
            list(values.shape) != shape or list(indices.shape) != shape or
            not values.is_contiguous() or not indices.is_contiguous() or
            values.requires_grad or indices.requires_grad):
        raise ValueError(f"{name}: PyTorch values/indices dtype, device, shape, or layout mismatch")
    if any(_overlaps(output, source) for output in (values, indices) for source in sources.values()):
        raise ValueError(f"{name}: PyTorch output aliases a resident input")
    if _overlaps(values, indices):
        raise ValueError(f"{name}: PyTorch values and indices must own distinct allocations")
    return values, indices


def _all_pair_outputs(pair: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return pair.values, pair.indices


def _pairs_independent(left: Any, right: Any) -> bool:
    return all(not _overlaps(a, b) for a in _all_pair_outputs(left) for b in _all_pair_outputs(right))


def _torch_observation(name: str, first: Any, peer: Any, repeated: Any,
                       dtype: str, sources: dict[str, torch.Tensor]) -> dict[str, Any]:
    first_values, first_indices = _validate_torch_pair(name, first, dtype, sources)
    peer_values, peer_indices = _validate_torch_pair(name, peer, dtype, sources)
    repeated_values, repeated_indices = _validate_torch_pair(name, repeated, dtype, sources)
    if not _pairs_independent(first, peer) or not _pairs_independent(first, repeated) or not _pairs_independent(peer, repeated):
        raise ValueError(f"{name}: PyTorch peer/repeat outputs alias another allocation")
    expected_values = _torch_output_values(first_values, dtype)
    peer_values_list = _torch_output_values(peer_values, dtype)
    repeated_values_list = _torch_output_values(repeated_values, dtype)
    expected_indices = [int(item) for item in first_indices.reshape(-1).tolist()]
    peer_indices_list = [int(item) for item in peer_indices.reshape(-1).tolist()]
    repeated_indices_list = [int(item) for item in repeated_indices.reshape(-1).tolist()]
    value_bits = _torch_value_bits(first_values)
    if (expected_values != peer_values_list or expected_values != repeated_values_list or
            value_bits != _torch_value_bits(peer_values) or value_bits != _torch_value_bits(repeated_values) or
            expected_indices != peer_indices_list or expected_indices != repeated_indices_list):
        raise ValueError(f"{name}: PyTorch peer/repeat values or indices differ exactly")
    return {
        "input_dtype": dtype,
        "values_dtype": dtype,
        "indices_dtype": "int64",
        "values_shape": list(first_values.shape),
        "indices_shape": list(first_indices.shape),
        "values": expected_values,
        "indices": expected_indices,
        "values_bits": value_bits,
        "values_data_ptr": str(first_values.data_ptr()),
        "indices_data_ptr": str(first_indices.data_ptr()),
        "values_contiguous": bool(first_values.is_contiguous()),
        "indices_contiguous": bool(first_indices.is_contiguous()),
        "values_disjoint_from_inputs": True,
        "indices_disjoint_from_inputs": True,
        "values_indices_independent": True,
        "peer_independent": True,
        "repeat_independent": True,
    }




def _exact_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return (actual.keys() == expected.keys() and
                all(_exact_json_equal(actual[key], expected[key]) for key in expected))
    if type(actual) is list:
        return (len(actual) == len(expected) and
                all(_exact_json_equal(a, b) for a, b in zip(actual, expected, strict=True)))
    return actual == expected


def _validate_observation(name: str, observation: Any, dtype: str,
                          reference: dict[str, Any] | None) -> None:
    if type(observation) is not dict or set(observation) != OBSERVATION_FIELDS:
        raise ValueError(f"{name}: malformed native cumulative-extrema observation")
    profile = name.split("_", 1)[1]
    shape = _expected_shape(profile)
    if (observation["input_dtype"] != dtype or observation["values_dtype"] != dtype or
            observation["indices_dtype"] != "int64" or
            type(observation["values_shape"]) is not list or observation["values_shape"] != shape or
            type(observation["indices_shape"]) is not list or observation["indices_shape"] != shape or
            observation["values_contiguous"] is not True or
            observation["indices_contiguous"] is not True or
            observation["values_disjoint_from_inputs"] is not True or
            observation["indices_disjoint_from_inputs"] is not True or
            observation["values_indices_independent"] is not True or
            observation["peer_independent"] is not True or
            observation["repeat_independent"] is not True):
        raise ValueError(f"{name}: native dtype, shape, contiguity, or allocation independence mismatch")
    for pointer_name in ("values_data_ptr", "indices_data_ptr"):
        pointer = observation[pointer_name]
        if type(pointer) is not str or not pointer.isdecimal():
            raise ValueError(f"{name}: {pointer_name} must be a decimal storage address")
        if math.prod(shape) > 0 and int(pointer) <= 0:
            raise ValueError(f"{name}: nonempty {pointer_name} must be nonzero")
    value_count = math.prod(shape)
    values, indices, bits = observation["values"], observation["indices"], observation["values_bits"]
    if (type(values) is not list or len(values) != value_count or
            type(indices) is not list or len(indices) != value_count or
            any(type(item) is not int for item in indices) or
            type(bits) is not list or len(bits) != value_count or
            any(type(item) is not str or not item for item in bits)):
        raise ValueError(f"{name}: native values, indices, or exact-bit counts mismatch output shape")
    if dtype.startswith("complex"):
        if any(type(item) is not dict or set(item) != {"real", "imag"} for item in values):
            raise ValueError(f"{name}: malformed complex values")
    elif dtype.startswith("float"):
        if any(type(item) not in (int, float, str) for item in values):
            raise ValueError(f"{name}: malformed floating values")
    elif dtype == "bool":
        if any(type(item) is not bool for item in values):
            raise ValueError(f"{name}: malformed boolean values")
    elif any(type(item) is not int for item in values):
        raise ValueError(f"{name}: malformed integer values")
    if reference is not None:
        for field in ("values", "indices", "values_bits"):
            if not _exact_json_equal(observation[field], reference[field]):
                raise ValueError(f"{name}: native {field} differ exactly from pinned PyTorch")


def _check_input_stability(sources: dict[str, torch.Tensor], payload: dict[str, Any], dtype: str) -> None:
    for name, field in (("base", "values"), ("rows_head", "rows_head_values"),
                        ("rows_tail", "rows_tail_values"),
                        ("offset_backing", "offset_backing_values")):
        if name in sources:
            if _tensor_payload(sources[name], dtype) != payload[field]:
                raise RuntimeError(f"Resident cumulative-extrema input changed during measurement: {name}")
        elif payload[field] != []:
            raise RuntimeError(f"Missing resident cumulative-extrema input: {name}")
    if "scalar" not in sources or _tensor_payload(sources["scalar"], dtype)[0] != payload["scalar_value"]:
        raise RuntimeError("Resident scalar cumulative-extrema input changed during measurement")
    if sources["empty"].shape != EMPTY_SHAPE or sources["empty"].numel() != 0:
        raise RuntimeError("Resident empty cumulative-extrema input shape or contents changed")
    if "reverse" in sources:
        base_values = payload["values"]
        reverse_values = [
            base_values[row * COLUMNS + column]
            for row in range(ROWS - 1, -1, -1)
            for column in range(COLUMNS - 1, -1, -1)
        ]
        if _tensor_payload(sources["reverse"], dtype) != reverse_values:
            raise RuntimeError("Resident reverse cumulative-extrema input changed during measurement")


def _progress_values(entry: Any, name: str) -> dict[str, int]:
    if type(entry) is not dict or "progress" not in entry or type(entry["progress"]) is not dict:
        raise ValueError(f"{name}: missing native action progress")
    progress = entry["progress"]
    required = {
        "preflight_calls_attempted", "preflight_calls_completed",
        "warmup_calls_attempted", "warmup_calls_completed",
        "sample_calls_attempted", "sample_calls_completed", "samples_completed",
    }
    if set(progress) != required or any(type(progress[key]) is not int or progress[key] < 0 for key in required):
        raise ValueError(f"{name}: malformed native action progress")
    return progress


def _timing_progress(entry: Any, expected: dict[str, int], name: str) -> None:
    progress = _progress_values(entry, name)
    for key, value in expected.items():
        if progress[key] != value:
            raise ValueError(f"{name}: native progress {key}={progress[key]}, expected {value}")

def _check_timing_record(name: str, entry: Any, dtype: str,
                         reference: dict[str, Any]) -> None:
    if type(entry) is not dict or set(entry) != {
            "observation", "samples_ms", "median_ms", "p95_ms", "progress"}:
        raise ValueError(f"{name}: malformed native timing record")
    _validate_observation(name, entry["observation"], dtype, reference)
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
    _timing_progress(entry, {
        "preflight_calls_attempted": 3, "preflight_calls_completed": 3,
        "warmup_calls_attempted": WARMUP, "warmup_calls_completed": WARMUP,
        "sample_calls_attempted": REPEATS * ITERATIONS,
        "sample_calls_completed": REPEATS * ITERATIONS, "samples_completed": REPEATS,
    }, name)


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _retain_native_failure(args: argparse.Namespace, dtype: str, invocation: str,
                           phase: str, returncode: int | None, stdout: str,
                           stderr: str, error: str) -> Path:
    path = args.output / f"failed_native_{dtype}_{invocation}.json"
    path.write_text(json.dumps({
        "input_dtype": dtype,
        "invocation": invocation,
        "phase": phase,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _snapshot_source_hashes(artifacts: dict[str, Path]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for label, path in artifacts.items():
        try:
            hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            hashes[label] = None
    return hashes


def _verify_sources(artifacts: dict[str, Path], source_hashes: dict[str, str],
                    hash_checks: list[dict[str, Any]], stage: str) -> None:
    current = _snapshot_source_hashes(artifacts)
    hash_checks.append({"stage": stage, "sha256": current})
    changed = [label for label, digest in current.items() if digest != source_hashes[label]]
    if changed:
        raise RuntimeError("Sources or native binary changed during measurement: " + ", ".join(changed))

def _verify_input(input_path: Path, input_hash: str, checks: list[dict[str, Any]], stage: str) -> None:
    current = hashlib.sha256(input_path.read_bytes()).hexdigest()
    checks.append({"stage": stage, "path": str(input_path), "sha256": current})
    if current != input_hash:
        raise RuntimeError(f"Input manifest changed during measurement: {input_path.name}")


def _invoke_native(args: argparse.Namespace, dtype: str, input_path: Path,
                   input_hash: str, artifacts: dict[str, Path], source_hashes: dict[str, str],
                   references: dict[str, dict[str, Any]], preflight: bool,
                   invocation: str) -> tuple[dict[str, Any] | None, list[str]]:
    phase = "preflight" if preflight else "timing"
    errors: list[str] = []
    command = [str(args.runner), str(input_path.resolve())]
    if preflight:
        command.append("--preflight")
    launch_attempted = False
    try:
        _verify_input(input_path, input_hash, args.input_hash_checks,
                      f"before_native_{phase}_{dtype}_{invocation}")
        _verify_sources(artifacts, source_hashes, args.hash_checks,
                        f"before_native_{phase}_{dtype}_{invocation}")
        launch_attempted = True
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = _text(error.stdout), _text(error.stderr)
        try:
            _verify_sources(artifacts, source_hashes, args.hash_checks,
                            f"after_native_{phase}_timeout_{dtype}_{invocation}")
            _verify_input(input_path, input_hash, args.input_hash_checks,
                          f"after_native_{phase}_timeout_{dtype}_{invocation}")
        except (RuntimeError, OSError) as attestation_error:
            errors.append(str(attestation_error))
        path = _retain_native_failure(args, dtype, invocation, phase, None, stdout, stderr,
                                      "native runner timed out")
        errors.append(f"{dtype}: native {phase} timeout evidence retained at {path}")
        return None, errors
    except (OSError, RuntimeError, ValueError) as error:
        if launch_attempted:
            try:
                _verify_sources(artifacts, source_hashes, args.hash_checks,
                                f"after_native_{phase}_launch_failure_{dtype}_{invocation}")
                _verify_input(input_path, input_hash, args.input_hash_checks,
                              f"after_native_{phase}_launch_failure_{dtype}_{invocation}")
            except (RuntimeError, OSError) as attestation_error:
                errors.append(str(attestation_error))
        path = _retain_native_failure(args, dtype, invocation, phase, None, "", "", repr(error))
        errors.append(f"{dtype}: native {phase} invocation evidence retained at {path}")
        return None, errors
    stem = args.output / f"native_{dtype}_{invocation}"
    try:
        _verify_sources(artifacts, source_hashes, args.hash_checks,
                        f"after_native_{phase}_{dtype}_{invocation}")
        _verify_input(input_path, input_hash, args.input_hash_checks,
                      f"after_native_{phase}_{dtype}_{invocation}")
    except (RuntimeError, OSError) as error:
        errors.append(str(error))
    stem.with_suffix(".stdout.txt").write_text(completed.stdout, encoding="utf-8")
    stem.with_suffix(".stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        path = _retain_native_failure(args, dtype, invocation, phase, completed.returncode,
                                      completed.stdout, completed.stderr,
                                      f"native runner exited with status {completed.returncode}")
        errors.append(f"{dtype}: native {phase} failure evidence retained at {path}")
        return None, errors
    try:
        report = json.loads(completed.stdout)
        if type(report) is not dict or set(report) != {"actions", "progress"}:
            raise ValueError("native report must contain exactly actions and progress")
        if type(report["actions"]) is not dict or set(report["actions"]) != set(ACTIONS_BY_DTYPE[dtype]):
            raise ValueError("native action set differs from the strict manifest")
        stem.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        progress_by_action: dict[str, dict[str, int]] = {}
        for name in ACTIONS_BY_DTYPE[dtype]:
            entry = report["actions"][name]
            if type(entry) is not dict:
                errors.append(f"{name}: native action result is not an object")
                continue
            if "failure" in entry:
                errors.append(f"{name}: native action failed: {entry['failure']}")
                if set(entry) != {"failure", "progress"}:
                    errors.append(f"{name}: malformed native failure fields")
                try:
                    progress_by_action[name] = _progress_values(entry, name)
                except ValueError as progress_error:
                    errors.append(str(progress_error))
                continue
            try:
                if "observation" not in entry:
                    raise ValueError("missing observation")
                _validate_observation(name, entry["observation"], dtype, references.get(name))
                if preflight:
                    if set(entry) != {"observation", "progress"}:
                        raise ValueError("malformed preflight action fields")
                    _timing_progress(entry, {
                        "preflight_calls_attempted": 3, "preflight_calls_completed": 3,
                        "warmup_calls_attempted": 0, "warmup_calls_completed": 0,
                        "sample_calls_attempted": 0, "sample_calls_completed": 0,
                        "samples_completed": 0,
                    }, name)
                else:
                    reference = references.get(name)
                    if reference is None:
                        raise ValueError("missing pinned PyTorch reference observation")
                    _check_timing_record(name, entry, dtype, reference)
                progress_by_action[name] = _progress_values(entry, name)
            except (KeyError, TypeError, ValueError, AssertionError) as error:
                errors.append(f"{name}: invalid native {phase} evidence: {error}")
        progress = report["progress"]
        progress_fields = {
            "preflight_calls_attempted", "preflight_calls_completed",
            "warmup_calls_attempted", "warmup_calls_completed",
            "sample_calls_attempted", "sample_calls_completed", "samples_completed",
        }
        root_fields = progress_fields | {"actions_attempted", "actions_completed"}
        if (type(progress) is not dict or set(progress) != root_fields or
                any(type(progress[field]) is not int or progress[field] < 0 for field in root_fields)):
            errors.append("native report progress fields are malformed")
        else:
            if progress["actions_attempted"] != len(ACTIONS_BY_DTYPE[dtype]):
                errors.append("native report action attempt count mismatch")
            if progress["actions_completed"] != sum(
                    "failure" not in report["actions"][name] for name in ACTIONS_BY_DTYPE[dtype]):
                errors.append("native report action completion count mismatch")
            if len(progress_by_action) == len(ACTIONS_BY_DTYPE[dtype]):
                expected_totals = {
                    field: sum(item[field] for item in progress_by_action.values())
                    for field in progress_fields
                }
                if any(progress[field] != expected_totals[field] for field in progress_fields):
                    errors.append("native report aggregate completion counts mismatch action evidence")
        return report, errors
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        path = _retain_native_failure(args, dtype, invocation, phase, completed.returncode,
                                      completed.stdout, completed.stderr, str(error))
        errors.append(f"{dtype}: invalid native {phase} report retained at {path}: {error}")
        return None, errors


def _manifest(dtype: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_dtype": dtype,
        "values_dtype": dtype,
        "indices_dtype": "int64",
        **payload,
        "actions": list(ACTIONS_BY_DTYPE[dtype]),
        "warmup": WARMUP,
        "repeats": REPEATS,
        "iterations": ITERATIONS,
    }


def _prepare_group(args: argparse.Namespace, dtype: str, artifacts: dict[str, Path],
                   source_hashes: dict[str, str], invocation_counts: dict[str, int]) -> dict[str, Any]:
    input_path = args.output / f"input_{dtype}.json"
    preflight_path = args.output / f"preflight_{dtype}.json"
    sources, payload = make_sources(dtype)
    callables = bind_actions(sources, dtype)
    manifest = _manifest(dtype, payload)
    if set(manifest) != set(MANIFEST_FIELDS):
        raise RuntimeError(f"{dtype}: benchmark manifest fields differ from the strict schema")
    input_path.write_text(json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    references: dict[str, dict[str, Any]] = {}
    pytorch_action_progress: dict[str, dict[str, int]] = {}
    reference_failures: list[dict[str, str]] = []
    try:
        _verify_sources(artifacts, source_hashes, args.hash_checks, f"before_pytorch_preflight_{dtype}")
        _verify_input(input_path, input_hash, args.input_hash_checks, f"before_pytorch_preflight_{dtype}")
        for name in ACTIONS_BY_DTYPE[dtype]:
            state = {"calls_attempted": 0, "calls_completed": 0}
            pairs: list[Any] = []
            try:
                for _ in range(3):
                    state["calls_attempted"] += 1
                    pairs.append(callables[name]())
                    state["calls_completed"] += 1
                references[name] = _torch_observation(
                    name, pairs[0], pairs[1], pairs[2], dtype, sources)
            except (RuntimeError, ValueError, TypeError, AssertionError, NotImplementedError) as error:
                reference_failures.append({"action": name, "error": str(error)})
            pytorch_action_progress[name] = state
        _check_input_stability(sources, payload, dtype)
    except (RuntimeError, OSError, ValueError) as error:
        reference_failures.append({"action": "<group>", "error": str(error)})
    try:
        _verify_sources(artifacts, source_hashes, args.hash_checks, f"after_pytorch_preflight_{dtype}")
        _verify_input(input_path, input_hash, args.input_hash_checks, f"after_pytorch_preflight_{dtype}")
    except (RuntimeError, OSError) as error:
        reference_failures.append({"action": "<attestation>", "error": str(error)})
    invocation = f"preflight_{invocation_counts[dtype]}"
    invocation_counts[dtype] += 1
    native_report, native_failures = _invoke_native(
        args, dtype, input_path, input_hash, artifacts, source_hashes,
        references, True, invocation)
    native_entries = native_report["actions"] if native_report is not None else {}
    preflight_actions: dict[str, Any] = {}
    for name in ACTIONS_BY_DTYPE[dtype]:
        preflight_actions[name] = {
            "pytorch": references.get(name),
            "pytorch_progress": pytorch_action_progress.get(name),
            "arraymancer": native_entries.get(name),
        }
    errors = [f"{item['action']}: {item['error']}" for item in reference_failures]
    errors.extend(native_failures)
    if native_report is None:
        errors.append("native preflight produced no parseable action report")
    preflight = {
        "input_dtype": dtype,
        "input_sha256": input_hash,
        "actions_attempted": list(ACTIONS_BY_DTYPE[dtype]),
        "actions_validated": [
            name for name in ACTIONS_BY_DTYPE[dtype]
            if name in references and name in native_entries and
            not any(f"{name}:" in error for error in native_failures)
        ],
        "pytorch": references,
        "pytorch_failures": reference_failures,
        "pytorch_action_progress": pytorch_action_progress,
        "pytorch_preflight_calls_attempted": sum(
            item["calls_attempted"] for item in pytorch_action_progress.values()),
        "pytorch_preflight_calls_completed": sum(
            item["calls_completed"] for item in pytorch_action_progress.values()),
        "arraymancer": native_report,
        "arraymancer_failures": native_failures,
        "action_evidence": preflight_actions,
        "passed": not errors and len(references) == len(ACTIONS_BY_DTYPE[dtype]) and
                  native_report is not None and not native_failures,
    }
    preflight_path.write_text(json.dumps(preflight, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "input_dtype": dtype,
        "sources": sources,
        "payload": payload,
        "callables": callables,
        "references": references,
        "input_path": input_path,
        "input_hash": input_hash,
        "preflight": preflight,
        "preflight_failures": errors,
        "preflight_passed": preflight["passed"],
        "native_preflight": native_report,
    }


def _consume_torch(pair: Any) -> None:
    """Consume both fresh output allocations without measuring a reduction."""
    _ = pair.values.data_ptr(), pair.indices.data_ptr()


def _time_torch(prepared: dict[str, Any], artifacts: dict[str, Path],
                source_hashes: dict[str, str], args: argparse.Namespace,
                round_number: int) -> dict[str, Any]:
    dtype = prepared["input_dtype"]
    _verify_sources(artifacts, source_hashes, args.hash_checks, f"before_pytorch_{dtype}_round_{round_number}")
    _verify_input(prepared["input_path"], prepared["input_hash"], args.input_hash_checks,
                  f"before_pytorch_{dtype}_round_{round_number}")
    _check_input_stability(prepared["sources"], prepared["payload"], dtype)
    report: dict[str, Any] = {}
    progress: dict[str, Any] = {}
    for name in ACTIONS_BY_DTYPE[dtype]:
        state = {"warmup_calls_attempted": 0, "warmup_calls_completed": 0,
                 "sample_calls_attempted": 0, "sample_calls_completed": 0,
                 "samples_completed": 0}
        try:
            call = prepared["callables"][name]
            for _ in range(WARMUP):
                state["warmup_calls_attempted"] += 1
                _consume_torch(call())
                state["warmup_calls_completed"] += 1
            samples: list[float] = []
            for _ in range(REPEATS):
                start = time.perf_counter_ns()
                for _ in range(ITERATIONS):
                    state["sample_calls_attempted"] += 1
                    _consume_torch(call())
                    state["sample_calls_completed"] += 1
                samples.append((time.perf_counter_ns() - start) / 1e6 / ITERATIONS)
                state["samples_completed"] += 1
            ordered = sorted(samples)
            report[name] = {
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p95_ms": ordered[math.ceil(0.95 * REPEATS) - 1],
            }
        except (RuntimeError, ValueError, TypeError, AssertionError, NotImplementedError) as error:
            report[name] = {"failure": str(error)}
        progress[name] = state
    try:
        _check_input_stability(prepared["sources"], prepared["payload"], dtype)
        _verify_input(prepared["input_path"], prepared["input_hash"], args.input_hash_checks,
                      f"after_pytorch_{dtype}_round_{round_number}")
        _verify_sources(artifacts, source_hashes, args.hash_checks,
                        f"after_pytorch_{dtype}_round_{round_number}")
    except (RuntimeError, OSError, ValueError) as error:
        report["<attestation>"] = {"failure": str(error)}
    return {"actions": report, "progress": progress}


def _round_results(args: argparse.Namespace, prepared: dict[str, Any],
                   artifacts: dict[str, Path], source_hashes: dict[str, str],
                   invocation_counts: dict[str, int], round_number: int) -> dict[str, Any]:
    dtype = prepared["input_dtype"]
    order = "pytorch-first" if round_number % 2 == 1 else "arraymancer-first"
    pytorch_result: dict[str, Any] | None = None
    native_result: dict[str, Any] | None = None
    backend_errors: dict[str, list[str]] = {"pytorch": [], "arraymancer": []}
    backend_order = ("pytorch", "arraymancer") if order == "pytorch-first" else ("arraymancer", "pytorch")
    for backend in backend_order:
        if backend == "pytorch":
            try:
                pytorch_result = _time_torch(prepared, artifacts, source_hashes, args, round_number)
                for name, entry in pytorch_result["actions"].items():
                    if "failure" in entry:
                        backend_errors["pytorch"].append(f"{name}: {entry['failure']}")
            except (RuntimeError, OSError, ValueError, TypeError, AssertionError, NotImplementedError) as error:
                backend_errors["pytorch"].append(str(error))
        else:
            native_invocation = f"timing_{dtype}_round_{round_number}_{invocation_counts[dtype]}"
            invocation_counts[dtype] += 1
            try:
                native_result, native_failures = _invoke_native(
                    args, dtype, prepared["input_path"], prepared["input_hash"], artifacts,
                    source_hashes, prepared["references"], False, native_invocation)
                backend_errors["arraymancer"].extend(native_failures)
            except (RuntimeError, OSError, ValueError, TypeError, AssertionError) as error:
                backend_errors["arraymancer"].append(str(error))
    return {
        "round": round_number,
        "order": order,
        "pytorch": pytorch_result,
        "arraymancer": native_result,
        "backend_errors": backend_errors,
    }


def _compare_round(dtype: str, round_entry: dict[str, Any], failures: list[str]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    torch_report = round_entry.get("pytorch")
    native_report = round_entry.get("arraymancer")
    torch_actions = torch_report.get("actions", {}) if type(torch_report) is dict else {}
    native_actions = native_report.get("actions", {}) if type(native_report) is dict else {}
    for name in ACTIONS_BY_DTYPE[dtype]:
        torch_entry, native_entry = torch_actions.get(name), native_actions.get(name)
        for metric in ("median_ms", "p95_ms"):
            record: dict[str, Any] = {
                "input_dtype": dtype,
                "round": round_entry["round"], "action": name, "metric": metric,
                "status": "unavailable",
            }
            if (type(torch_entry) is dict and type(native_entry) is dict and
                    metric in torch_entry and metric in native_entry):
                torch_value, native_value = torch_entry[metric], native_entry[metric]
                record.update({"pytorch_ms": torch_value, "arraymancer_ms": native_value})
                if native_value < torch_value:
                    record["status"] = "win"
                else:
                    record["status"] = "loss"
                    failures.append(f"{dtype}: round {round_entry['round']} {name} {metric} did not strictly win")
            else:
                failures.append(f"{dtype}: round {round_entry['round']} {name} {metric} comparison unavailable")
            comparisons.append(record)
    return comparisons


def _time_group(args: argparse.Namespace, prepared: dict[str, Any],
                artifacts: dict[str, Path], source_hashes: dict[str, str],
                invocation_counts: dict[str, int], failures: list[str]) -> dict[str, Any]:
    dtype = prepared["input_dtype"]
    rounds: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []
    rounds_path = args.output / f"rounds_{dtype}.json"
    rounds_path.write_text("[]\n", encoding="utf-8")
    for round_number in range(1, PAIRED_ROUNDS + 1):
        try:
            item = _round_results(args, prepared, artifacts, source_hashes,
                                  invocation_counts, round_number)
        except (RuntimeError, OSError, ValueError, TypeError, AssertionError) as error:
            item = {
                "round": round_number,
                "order": "pytorch-first" if round_number % 2 == 1 else "arraymancer-first",
                "pytorch": None,
                "arraymancer": None,
                "backend_errors": {"pytorch": [], "arraymancer": [str(error)]},
            }
        rounds.append(item)
        comparison_records.extend(_compare_round(dtype, item, failures))
        rounds_path.write_text(
            json.dumps(rounds, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for round_entry in rounds:
        for backend, entries in round_entry["backend_errors"].items():
            failures.extend(f"{dtype} round {round_entry['round']} {backend}: {error}" for error in entries)
    completed_actions = sum(
        all(type(round_entry.get(backend)) is dict and
            all(type(round_entry[backend].get("actions", {}).get(name)) is dict and
                "failure" not in round_entry[backend]["actions"].get(name, {}) and
                "samples_ms" in round_entry[backend]["actions"][name]
                for round_entry in rounds)
            for backend in ("pytorch", "arraymancer"))
        for name in ACTIONS_BY_DTYPE[dtype]
    )
    timing_failures = [item for item in comparison_records if item["status"] != "win"]
    comparison_completed_count = sum(item["status"] != "unavailable" for item in comparison_records)
    has_backend_errors = any(entries for round_entry in rounds
                             for entries in round_entry["backend_errors"].values())
    group = {
        "input_dtype": dtype,
        "values_dtype": dtype,
        "indices_dtype": "int64",
        "actions": list(ACTIONS_BY_DTYPE[dtype]),
        "input_sha256": prepared["input_hash"],
        "correctness": "passed",
        "native_preflight": "passed",
        "preflight": prepared["preflight"],
        "rounds": rounds,
        "comparison_records": comparison_records,
        "completed_action_count": completed_actions,
        "comparison_attempted_count": len(comparison_records),
        "comparison_completed_count": comparison_completed_count,
        "comparison_unavailable_count": len(comparison_records) - comparison_completed_count,
        "timing": "passed" if (
            completed_actions == len(ACTIONS_BY_DTYPE[dtype]) and
            not has_backend_errors and not timing_failures) else "failed",
        "timing_failures": timing_failures,
    }
    if timing_failures or has_backend_errors:
        (args.output / f"failed_timing_{dtype}.json").write_text(
            json.dumps({
                "timing_failures": timing_failures,
                "round_backend_errors": [
                    {"round": item["round"], "backend_errors": item["backend_errors"]}
                    for item in rounds if any(item["backend_errors"].values())
                ],
                "comparison_attempted_count": len(comparison_records),
                "comparison_completed_count": comparison_completed_count,
            }, indent=2) + "\n", encoding="utf-8")
    return group


def _record_group_failure(args: argparse.Namespace, dtype: str, input_hash: str | None,
                          actions: tuple[str, ...], error: str,
                          preflight: dict[str, Any] | None,
                          rounds: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rounds = rounds or []
    comparison_records = [
        record for round_entry in rounds
        for record in _compare_round(dtype, round_entry, [])
    ]
    completed_actions = sum(
        len(rounds) == PAIRED_ROUNDS and
        all(type(round_entry.get(backend)) is dict and
            type(round_entry[backend].get("actions", {}).get(name)) is dict and
            "failure" not in round_entry[backend]["actions"][name] and
            "samples_ms" in round_entry[backend]["actions"][name]
            for round_entry in rounds for backend in ("pytorch", "arraymancer"))
        for name in actions
    )
    completed_comparisons = sum(record["status"] != "unavailable" for record in comparison_records)
    path = args.output / f"group_failure_{dtype}.json"
    path.write_text(json.dumps({
        "input_dtype": dtype,
        "values_dtype": dtype,
        "indices_dtype": "int64",
        "input_sha256": input_hash,
        "actions": list(actions),
        "error": error,
        "preflight": preflight,
        "rounds": rounds,
        "comparison_records": comparison_records,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "input_dtype": dtype,
        "values_dtype": dtype,
        "indices_dtype": "int64",
        "input_sha256": input_hash,
        "actions": list(actions),
        "correctness": "failed",
        "error": error,
        "preflight": preflight,
        "rounds": rounds,
        "completed_action_count": completed_actions,
        "comparison_records": comparison_records,
        "comparison_attempted_count": len(comparison_records),
        "comparison_completed_count": completed_comparisons,
        "comparison_unavailable_count": len(comparison_records) - completed_comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/cumulative_extrema_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/cumulative_extrema_benchmark"))
    args = parser.parse_args()
    args.hash_checks = []
    args.input_hash_checks = []
    args.runner = (ROOT / args.runner).resolve() if not args.runner.is_absolute() else args.runner
    args.output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    if platform.python_implementation() != "CPython" or sys.version_info[:3] != PYTHON_VERSION:
        raise RuntimeError("Requires pinned CPython 3.13.18")
    if (torch.__version__.split("+")[0] != TORCH_VERSION or
            torch.version.git_version != TORCH_GIT_REVISION):
        raise RuntimeError("Requires pinned PyTorch 2.15.0 wheel revision " + TORCH_GIT_REVISION)
    if not args.runner.is_file():
        raise RuntimeError(f"Native runner not found: {args.runner}; build it using the docstring command")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise RuntimeError("Output directory must be empty; choose a fresh --output to preserve prior evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = {str(path): ROOT / path for path in SOURCE_PATHS}
    artifacts["native_runner"] = args.runner
    for label, path in artifacts.items():
        if not path.is_file():
            raise RuntimeError(f"Required attested source/binary is missing: {label} ({path})")
    source_hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest()
                     for label, path in artifacts.items()}
    compile_command = [
        "nim", "c", "-d:ssl", "--cc:clang", "--mm:orc", "-d:release", "--path:src", f"--out:{args.runner}",
        str(ROOT / RUNNER_SOURCE),
    ]
    nim_version = subprocess.run(["nim", "--version"], check=True,
                                 capture_output=True, text=True).stdout.strip()
    failures: list[str] = []
    prepared_groups: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    invocation_counts = {dtype: 0 for dtype in DTYPES}
    preflight_attempted_action_count = 0
    preflight_validated_action_count = 0

    # All 13 dtype groups and every action are preflighted before any timing.
    for dtype in DTYPES:
        actions = ACTIONS_BY_DTYPE[dtype]
        preflight_attempted_action_count += len(actions)
        try:
            prepared = _prepare_group(args, dtype, artifacts, source_hashes, invocation_counts)
            prepared_groups.append(prepared)
            input_hashes[dtype] = prepared["input_hash"]
            preflight_validated_action_count += len(prepared["preflight"]["actions_validated"])
            if prepared["preflight_failures"] or not prepared["preflight_passed"]:
                failure = "; ".join(prepared["preflight_failures"] or ["preflight validation failed"])
                failures.append(f"{dtype}: correctness preflight failed: {failure}")
                old_rounds: list[dict[str, Any]] = []
                groups.append(_record_group_failure(
                    args, dtype, prepared["input_hash"], actions, failure,
                    prepared["preflight"], old_rounds))
        except (RuntimeError, OSError, ValueError, TypeError, AssertionError, NotImplementedError) as error:
            failures.append(f"{dtype}: preflight setup failed: {error}")
            input_path = args.output / f"input_{dtype}.json"
            digest = hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path.is_file() else None
            input_hashes[dtype] = digest if digest is not None else ""
            preflight_path = args.output / f"preflight_{dtype}.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else None
            groups.append(_record_group_failure(args, dtype, digest, actions, str(error), preflight))

    # If any single action failed preflight, the global timing schedule is skipped.
    if not failures and len(prepared_groups) == len(DTYPES):
        for prepared in prepared_groups:
            dtype = prepared["input_dtype"]
            try:
                group = _time_group(args, prepared, artifacts, source_hashes,
                                    invocation_counts, failures)
                groups.append(group)
                if group["completed_action_count"] != len(ACTIONS_BY_DTYPE[dtype]):
                    failures.append(f"{dtype}: not every action completed all three timing rounds")
            except (RuntimeError, OSError, ValueError, TypeError, AssertionError) as error:
                failures.append(f"{dtype}: timing group failed: {error}")
                rounds_path = args.output / f"rounds_{dtype}.json"
                saved_rounds = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.is_file() else []
                groups.append(_record_group_failure(
                    args, dtype, prepared["input_hash"], ACTIONS_BY_DTYPE[dtype],
                    str(error), prepared["preflight"], saved_rounds))
    else:
        for prepared in prepared_groups:
            dtype = prepared["input_dtype"]
            if not any(group.get("input_dtype") == dtype for group in groups):
                groups.append({
                    "input_dtype": dtype,
                    "values_dtype": dtype,
                    "indices_dtype": "int64",
                    "input_sha256": prepared["input_hash"],
                    "actions": list(ACTIONS_BY_DTYPE[dtype]),
                    "correctness": "preflight_passed_timing_skipped",
                    "native_preflight": "passed",
                    "timing_skipped": "at least one dtype/action failed global preflight",
                    "preflight": prepared["preflight"],
                    "rounds": [],
                    "completed_action_count": 0,
                    "comparison_records": [],
                    "comparison_attempted_count": 0,
                    "comparison_completed_count": 0,
                    "comparison_unavailable_count": 0,
                })

    # Any failed group not represented yet (for example a source setup error) is retained.
    present = {group.get("input_dtype") for group in groups}
    for dtype in DTYPES:
        if dtype not in present:
            preflight_path = args.output / f"preflight_{dtype}.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else None
            groups.append(_record_group_failure(
                args, dtype, input_hashes.get(dtype), ACTIONS_BY_DTYPE[dtype],
                "group was not otherwise represented after an earlier failure", preflight))
    groups.sort(key=lambda group: DTYPES.index(group["input_dtype"]))

    for dtype, digest in input_hashes.items():
        path = args.output / f"input_{dtype}.json"
        if digest:
            try:
                _verify_input(path, digest, args.input_hash_checks, f"final_{dtype}")
            except (RuntimeError, OSError) as error:
                failures.append(str(error))
    try:
        _verify_sources(artifacts, source_hashes, args.hash_checks, "after_all")
    except (RuntimeError, OSError) as error:
        failures.append(str(error))
        (args.output / "source_hash_failure.json").write_text(json.dumps({
            "initial_sha256": source_hashes,
            "current_sha256": _snapshot_source_hashes(artifacts),
            "source_hash_checks": args.hash_checks,
            "input_hash_checks": args.input_hash_checks,
            "error": str(error),
        }, indent=2) + "\n", encoding="utf-8")

    action_count = ACTION_COUNT
    completed_action_count = sum(group.get("completed_action_count", 0) for group in groups)
    comparison_records = [record for group in groups for record in group.get("comparison_records", [])]
    comparison_attempted_count = sum(group.get("comparison_attempted_count", 0) for group in groups)
    comparison_count = sum(record["status"] != "unavailable" for record in comparison_records)
    comparison_unavailable_count = comparison_attempted_count - comparison_count
    expected_comparison_count = action_count * PAIRED_ROUNDS * 2
    if (len(groups) != len(DTYPES) or action_count != 244 or
            preflight_attempted_action_count != 244 or
            (not failures and preflight_validated_action_count != 244) or
            completed_action_count != action_count or
            comparison_attempted_count != expected_comparison_count or
            comparison_count != expected_comparison_count):
        failures.append(
            f"cumulative-extrema protocol requires 13 groups, 244 actions, "
            f"244 complete preflights, 244 completed actions and 1,464 strict comparisons "
            f"(observed groups={len(groups)}, preflight_attempted={preflight_attempted_action_count}, "
            f"preflight_validated={preflight_validated_action_count}, completed={completed_action_count}, "
            f"comparison_attempted={comparison_attempted_count}, comparison_completed={comparison_count}, "
            f"comparison_unavailable={comparison_unavailable_count})")
    if any(record["status"] != "win" for record in comparison_records):
        failures.append("At least one cumulative-extrema median or p95 comparison did not strictly win")

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
            "operations": list(OPERATIONS),
            "profiles": list(PROFILES),
            "actions_per_dtype": {dtype: list(ACTIONS_BY_DTYPE[dtype]) for dtype in DTYPES},
            "group_count": len(groups),
            "action_count": action_count,
            "preflight_attempted_action_count": preflight_attempted_action_count,
            "preflight_validated_action_count": preflight_validated_action_count,
            "completed_action_count": completed_action_count,
            "warmup_calls_per_action": WARMUP,
            "samples_per_action_per_round": REPEATS,
            "allocating_calls_per_sample": ITERATIONS,
            "paired_rounds_per_group": PAIRED_ROUNDS,
            "comparisons": comparison_count,
            "comparison_attempted_count": comparison_attempted_count,
            "comparison_completed_count": comparison_count,
            "comparison_unavailable_count": comparison_unavailable_count,
            "expected_comparisons": expected_comparison_count,
            "input_profiles": {
                "base_shape": [ROWS, COLUMNS], "cube_shape": list(CUBE_SHAPE),
                "vector_shape": list(VECTOR_SHAPE), "offset_backing_shape": list(OFFSET_SHAPE),
                "scalar_shape": [], "empty_shape": list(EMPTY_SHAPE),
                "source_values": "base i%31; head/tail boundary values 63 and 0; bool truth conversion",
                "reverse": "PyTorch resident torch.flip materialization; native genuine negative strides; identical logical values",
                "broadcast": "first base row expanded to 128x128 with zero row stride",
                "offset": "view[:,1:129] of a resident 128x130 backing",
                "complex_and_wide_unsigned": "scalar and empty only",
            },
            "preflight": "all 13 dtype groups and all 244 actions finish exact values/indices, shape, dtype, contiguity and input/peer/repeat storage checks before timing",
            "timed_consumer": "both values and indices allocations consumed by PyTorch data_ptr reads and a native compiler barrier",
            "timing_gate": "strict median and p95 win for every action in every alternating round",
            "exact_comparison": "full JSON values, exact int64 indices, and fixed-width lowercase raw value bit words; signed zero and NaN payload bits are not normalized",
            "empty_independence": "storage pointer independence is vacuous for zero-element outputs",
        },
        "source_sha256": {label: digest for label, digest in source_hashes.items()
                           if label != "native_runner"},
        "source_sha256_after": {
            label: digest for label, digest in _snapshot_source_hashes(artifacts).items()
            if label != "native_runner"
        },
        "runner_sha256": source_hashes["native_runner"],
        "runner_sha256_after": _snapshot_source_hashes({"native_runner": args.runner})["native_runner"],
        "binary_sha256": source_hashes["native_runner"],
        "binary_sha256_after": _snapshot_source_hashes({"native_runner": args.runner})["native_runner"],
        "input_sha256": input_hashes,
        "source_hash_checks": args.hash_checks,
        "input_hash_checks": args.input_hash_checks,
        "commands": {
            "build_cwd": str(ROOT),
            "build_command": compile_command,
            "compiler_version": nim_version,
            "invoke_command_template": [str(args.runner), "<input-manifest.json>"],
            "preflight_command_template": [str(args.runner), "<input-manifest.json>", "--preflight"],
        },
        "groups": groups,
        "comparison_records": comparison_records,
        "failures": failures,
        "passed": not failures,
    }
    result_path = args.output / "results.json.gz"
    with gzip.open(result_path, "wt", encoding="utf-8", compresslevel=1) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "groups": len(groups),
        "actions": action_count,
        "preflight_attempted_actions": preflight_attempted_action_count,
        "preflight_validated_actions": preflight_validated_action_count,
        "completed_actions": completed_action_count,
        "comparison_attempted": comparison_attempted_count,
        "comparison_unavailable": comparison_unavailable_count,
        "comparisons": comparison_count,
        "expected_comparisons": expected_comparison_count,
        "failures": failures,
        "passed": not failures,
        "results": str(result_path),
    }))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
