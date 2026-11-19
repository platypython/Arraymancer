#!/usr/bin/env python3
"""Check and time CPU reduction and scan dtype actions against pinned PyTorch.

Build the native runner from the repository root with:
  nim c -d:release --path:src --out:build/reduction_dtype_runner benchmarks/pytorch_head_to_head/reduction_dtype_runner.nim
Then run the floating-output benchmark (the default; optional --output and --runner):
  python benchmarks/pytorch_head_to_head/benchmark_reduction_dtype.py
Run just the six supported integer/bool outputs with:
  python benchmarks/pytorch_head_to_head/benchmark_reduction_dtype.py --output-dtypes uint8 int8 int16 int32 int64 bool
Build the default complex scan mode with:
  nim c -d:release -d:complexScans --path:src --out:build/reduction_dtype_runner benchmarks/pytorch_head_to_head/reduction_dtype_runner.nim
Then run --complex-scans: two matching complex input/output groups, eight actions
and 48 comparisons under the same timing protocol.
Build and run explicit complex scan output-dtype coverage with:
  nim c -d:release -d:complexScanDtypes --path:src --out:build/reduction_dtype_runner benchmarks/pytorch_head_to_head/reduction_dtype_runner.nim
  python benchmarks/pytorch_head_to_head/benchmark_reduction_dtype.py --complex-scan-dtypes
This mode covers 40 groups, 160 scan actions and 960 median/p95 comparisons.
Build and run the complex mean benchmark:
  nim c -d:release -d:complexMeans --path:src --out:build/reduction_dtype_runner benchmarks/pytorch_head_to_head/reduction_dtype_runner.nim
  python benchmarks/pytorch_head_to_head/benchmark_reduction_dtype.py --complex-means
This mode covers 30 groups, 96 explicit/default mean actions and 576 timing
comparisons. Matching complex input/output groups include the three default
overload actions in addition to the three explicit-dtype actions.
The 22 floating input/output groups each contain thirteen actions, for 286 actions
and 1,716 median/p95 comparisons across three paired rounds. Integer outputs have
ten supported sum/product/scan actions; bool outputs have six sum/product actions,
for 616 actions and 3,696 comparisons across all eleven input dtypes. Each paired
round alternates backend order; all 100 samples of 50 allocating calls are
retained. Inputs stay resident and explicit output-dtype casting is part of each
timed reduction or scan call.

Integer and bool outputs are validated exactly, including dtype, shape and value
count. Float outputs retain the existing tolerance checks. Every action must win
both median and p95 in every paired round.
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

INPUT_DTYPES = (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float32", "float64",
)
INTEGER_DTYPES = frozenset(INPUT_DTYPES[:9])
COMPLEX_DTYPES = ("complex64", "complex128")
COMPLEX_SCAN_CASES = (
    "cumsum_rows", "cumsum_columns", "cumprod_rows", "cumprod_columns",
)
COMPLEX_SUM_CASES = ("sum", "sum_rows", "sum_columns")
COMPLEX_SUM_DEFAULT_CASES = ("sum_default", "sum_rows_default", "sum_columns_default")
COMPLEX_MEAN_CASES = ("mean", "mean_rows", "mean_columns")
COMPLEX_MEAN_DEFAULT_CASES = (
    "mean_default", "mean_rows_default", "mean_columns_default",
)
OUTPUT_DTYPES = ("float32", "float64")
INTEGER_OUTPUT_DTYPES = ("uint8", "int8", "int16", "int32", "int64")
BOOL_OUTPUT_DTYPES = ("bool",)
SELECTABLE_OUTPUT_DTYPES = OUTPUT_DTYPES + INTEGER_OUTPUT_DTYPES + BOOL_OUTPUT_DTYPES
CASES = (
    "mean", "mean_rows", "mean_columns", "sum", "sum_rows", "sum_columns",
    "prod", "prod_rows", "prod_columns",
    "cumsum_rows", "cumsum_columns", "cumprod_rows", "cumprod_columns",
)
INTEGER_CASES = (
    "sum", "sum_rows", "sum_columns", "prod", "prod_rows", "prod_columns",
    "cumsum_rows", "cumsum_columns", "cumprod_rows", "cumprod_columns",
)
BOOL_CASES = ("sum", "sum_rows", "sum_columns", "prod", "prod_rows", "prod_columns")

def actions_for_output(output_dtype: str) -> tuple[str, ...]:
    if output_dtype == "bool":
        return BOOL_CASES
    if output_dtype in INTEGER_OUTPUT_DTYPES:
        return INTEGER_CASES
    if output_dtype in OUTPUT_DTYPES:
        return CASES
    raise ValueError(f"Unsupported reduction output dtype: {output_dtype}")

TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool, "int8": torch.int8, "int16": torch.int16, "int32": torch.int32,
    "int64": torch.int64, "uint8": torch.uint8, "uint16": torch.uint16,
    "uint32": torch.uint32, "uint64": torch.uint64,
    "float32": torch.float32, "float64": torch.float64,
    "complex64": torch.complex64, "complex128": torch.complex128,
}
TORCH_VERSION = "2.15.0"
TORCH_GIT_REVISION = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
PYTHON_VERSION = (3, 13, 18)
ROWS, COLUMNS = 128, 128
WARMUP, REPEATS, ITERATIONS = 50, 100, 50
ROOT = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/reduction_dtype_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_reduction_dtype.py")
SOURCE_PATHS = (
    Path("src/arraymancer/tensor/aggregate.nim"),
    Path("src/arraymancer/tensor/private/p_accelerate.nim"),
    Path("src/arraymancer/tensor/private/p_complex.nim"),
    Path("src/arraymancer/laser/compiler_optim_hints.nim"),
    RUNNER_SOURCE,
    SCRIPT_SOURCE,
)


def operation(
    name: str, value: torch.Tensor, output_dtype: str, *, explicit_scan_dtype: bool = False,
) -> Callable[[], torch.Tensor]:
    """Resolve an allocating reduction or scan; keep casting inside the action."""
    if output_dtype in COMPLEX_DTYPES and name in COMPLEX_SUM_CASES + COMPLEX_SUM_DEFAULT_CASES:
        is_default = name in COMPLEX_SUM_DEFAULT_CASES
        action = name.removesuffix("_default")
        reducer = value.sum
        if action == "sum":
            if is_default:
                return lambda: reducer()
            dtype = TORCH_DTYPES[output_dtype]
            return lambda: reducer(dtype=dtype)
        axis = -1 if action == "sum_rows" else -2
        if is_default:
            return lambda: reducer(dim=axis, keepdim=True)
        dtype = TORCH_DTYPES[output_dtype]
        return lambda: reducer(dim=axis, keepdim=True, dtype=dtype)
    if output_dtype in COMPLEX_DTYPES and (
        name in COMPLEX_MEAN_CASES or name in COMPLEX_MEAN_DEFAULT_CASES
    ):
        is_default = name in COMPLEX_MEAN_DEFAULT_CASES
        action = name.removesuffix("_default")
        reducer = value.mean
        if action == "mean":
            if is_default:
                return lambda: reducer()
            return lambda: reducer(dtype=TORCH_DTYPES[output_dtype])
        axis = -1 if action == "mean_rows" else -2
        if is_default:
            return lambda: reducer(dim=axis, keepdim=True)
        return lambda: reducer(
            dim=axis, keepdim=True, dtype=TORCH_DTYPES[output_dtype],
        )
    if output_dtype in COMPLEX_DTYPES:
        axis = -1 if name.endswith("_rows") else -2
        scanner = value.cumsum if name.startswith("cumsum") else value.cumprod
        if explicit_scan_dtype:
            dtype = TORCH_DTYPES[output_dtype]
            return lambda: scanner(dim=axis, dtype=dtype)
        return lambda: scanner(dim=axis)
    dtype = TORCH_DTYPES[output_dtype]
    if explicit_scan_dtype and name.startswith(("cumsum", "cumprod")):
        scanner = value.cumsum if name.startswith("cumsum") else value.cumprod
        axis = -1 if name.endswith("_rows") else -2
        return lambda: scanner(dim=axis, dtype=dtype)
    if name in ("mean", "sum", "prod"):
        if name == "mean":
            reducer = value.mean
        elif name == "sum":
            reducer = value.sum
        else:
            reducer = value.prod
        return lambda: reducer(dtype=dtype)
    if name in ("cumsum_rows", "cumprod_rows"):
        scanner = value.cumsum if name == "cumsum_rows" else value.cumprod
        return lambda: scanner(dim=-1, dtype=dtype)
    if name in ("cumsum_columns", "cumprod_columns"):
        scanner = value.cumsum if name == "cumsum_columns" else value.cumprod
        return lambda: scanner(dim=-2, dtype=dtype)
    if name in ("mean_rows", "sum_rows", "prod_rows"):
        axis = -1
    elif name in ("mean_columns", "sum_columns", "prod_columns"):
        axis = -2
    else:
        raise ValueError(f"Unknown reduction dtype action: {name}")
    if name.startswith("mean"):
        reducer = value.mean
    elif name.startswith("sum"):
        reducer = value.sum
    else:
        reducer = value.prod
    return lambda: reducer(dim=axis, keepdim=True, dtype=dtype)


def make_source(dtype: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Create one resident, already typed source and its native manifest values."""
    if dtype in COMPLEX_DTYPES:
        indices = torch.arange(ROWS * COLUMNS, dtype=torch.float64)
        real = 1 + ((indices % 17) - 8) * 0.0001
        imag = ((indices % 2) * 2 - 1) * 0.0002
        source = torch.complex(real, imag).reshape(ROWS, COLUMNS)
        source = source.to(dtype=TORCH_DTYPES[dtype])
        # Serialize the actual component values after rounding to the source dtype.
        values = [[item.real, item.imag] for item in source.flatten().tolist()]
        return source, {"values": values}
    if dtype in INTEGER_DTYPES:
        words = [index % 5 - 2 for index in range(ROWS * COLUMNS)]
        source = torch.tensor(words, dtype=torch.int64).reshape(ROWS, COLUMNS)
        return source.to(dtype=TORCH_DTYPES[dtype]), {"words": words}

    source = (1 + ((torch.arange(ROWS * COLUMNS, dtype=torch.float64) % 17) - 8)
              * 0.0001).reshape(ROWS, COLUMNS)
    source = source.to(dtype=TORCH_DTYPES[dtype])
    # Serialize the actual rounded source values so native and Torch see identical inputs.
    return source, {"values": source.flatten().tolist()}



def complex_mean_actions(input_dtype: str, output_dtype: str) -> tuple[str, ...]:
    """Return explicit mean actions plus true default overloads for matching complex types."""
    if input_dtype == output_dtype and input_dtype in COMPLEX_DTYPES:
        return COMPLEX_MEAN_CASES + COMPLEX_MEAN_DEFAULT_CASES
    return COMPLEX_MEAN_CASES


def complex_sum_actions(input_dtype: str, output_dtype: str) -> tuple[str, ...]:
    """Return explicit sum actions plus valid matching-complex default overloads."""
    if input_dtype == output_dtype and input_dtype in COMPLEX_DTYPES:
        return COMPLEX_SUM_CASES + COMPLEX_SUM_DEFAULT_CASES
    return COMPLEX_SUM_CASES

def run_group(
    args: argparse.Namespace,
    input_dtype: str,
    output_dtype: str,
    source: torch.Tensor,
    source_payload: dict[str, Any],
    input_hashes: dict[str, str],
    failures: list[str],
) -> dict[str, Any]:
    if args.complex_sums:
        action_names = complex_sum_actions(input_dtype, output_dtype)
    elif args.complex_means:
        action_names = complex_mean_actions(input_dtype, output_dtype)
    elif args.complex_scan_dtypes or output_dtype in COMPLEX_DTYPES:
        action_names = COMPLEX_SCAN_CASES
    else:
        action_names = actions_for_output(output_dtype)
    group = f"{input_dtype}_to_{output_dtype}"
    manifest: dict[str, Any] = {
        "dtype": input_dtype,
        "output_dtype": output_dtype,
        "rows": ROWS,
        "columns": COLUMNS,
        **source_payload,
        "actions": list(action_names),
        "warmup": WARMUP,
        "repeats": REPEATS,
        "iterations": ITERATIONS,
    }
    input_path = args.output / f"input_{group}.json"
    input_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    input_hashes[group] = input_hash

    callables = {
        name: operation(name, source, output_dtype, explicit_scan_dtype=args.complex_scan_dtypes)
        for name in action_names
    }
    expected = {name: call() for name, call in callables.items()}
    native_command = [str(args.runner), str(input_path.resolve())]
    tolerance = 5e-6 if output_dtype == "float32" else 1e-12
    native_runs = 0

    def native() -> dict[str, Any]:
        """Preserve runner timings and validate dtype, shape and numerical results."""
        nonlocal native_runs
        completed = subprocess.run(
            native_command, check=True, capture_output=True, text=True, timeout=120,
        )
        report: dict[str, Any] = json.loads(completed.stdout)
        raw_path = args.output / f"native_{group}_{native_runs}.json"
        raw_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        native_runs += 1
        for name in action_names:
            actual = report[name]["observation"]
            if actual.get("input_dtype") != input_dtype:
                raise RuntimeError(f"{group}/{name}: input dtype mismatch")
            reference = expected[name]
            if actual["shape"] != list(reference.shape):
                raise RuntimeError(f"{group}/{name}: shape mismatch")
            expected_dtype = str(reference.dtype).removeprefix("torch.")
            if actual.get("dtype") != expected_dtype:
                raise RuntimeError(f"{group}/{name}: dtype mismatch")
            raw_values = actual["values"]
            if len(raw_values) != reference.numel():
                raise RuntimeError(f"{group}/{name}: value count mismatch")
            expected_values = reference.flatten().tolist()
            if output_dtype in COMPLEX_DTYPES:
                if any(not isinstance(value, dict) or set(value) != {"real", "imag"}
                       for value in raw_values):
                    raise RuntimeError(f"{group}/{name}: complex values must contain real and imag")
                actual_components = []
                for value in raw_values:
                    components = (value["real"], value["imag"])
                    if any(
                        (type(component) not in (int, float) and
                         component not in ("NaN", "+Inf", "-Inf"))
                        for component in components
                    ):
                        raise RuntimeError(f"{group}/{name}: invalid complex component")
                    actual_components.append([float(component) for component in components])
                actual_values = torch.view_as_complex(torch.tensor(
                    actual_components, dtype=torch.float64,
                ).contiguous()).to(dtype=reference.dtype).reshape(actual["shape"])
                complex_tolerance = 5e-6 if output_dtype == "complex64" else 1e-12
                torch.testing.assert_close(
                    actual_values, reference, atol=complex_tolerance,
                    rtol=complex_tolerance, equal_nan=True,
                    msg=lambda message: f"{group}/{name}: {message}",
                )
            elif output_dtype == "bool":
                if any(type(value) is not bool for value in raw_values):
                    raise RuntimeError(f"{group}/{name}: bool values must be JSON booleans")
                if raw_values != expected_values:
                    raise RuntimeError(f"{group}/{name}: exact bool values mismatch")
            elif output_dtype in INTEGER_OUTPUT_DTYPES:
                if any(not isinstance(value, str) for value in raw_values):
                    raise RuntimeError(f"{group}/{name}: integer values must be exact decimal strings")
                actual_values = [int(value, 10) for value in raw_values]
                if actual_values != expected_values:
                    raise RuntimeError(f"{group}/{name}: exact integer values mismatch")
            else:
                actual_values = torch.tensor(
                    [float(value) for value in raw_values], dtype=reference.dtype,
                ).reshape(actual["shape"])
                torch.testing.assert_close(
                    actual_values, reference, atol=tolerance, rtol=tolerance, equal_nan=True,
                    msg=lambda message: f"{group}/{name}: {message}",
                )
        return report

    def reference() -> dict[str, Any]:
        """Match native warmup, allocating calls and per-sample timing boundaries."""
        report: dict[str, Any] = {}
        for name, call in callables.items():
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
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
            }
        return report

    native()  # Validate every native action before timing competition.
    rounds: list[dict[str, Any]] = []
    for index in range(3):
        order = "torch-first" if index % 2 == 0 else "arraymancer-first"
        if index % 2 == 0:
            torch_result, native_result = reference(), native()
        else:
            native_result, torch_result = native(), reference()
        rounds.append({
            "round": index + 1,
            "order": order,
            "pytorch": torch_result,
            "arraymancer": native_result,
        })
        (args.output / f"rounds_{group}.json").write_text(
            json.dumps(rounds, indent=2) + "\n", encoding="utf-8",
        )
        for name in action_names:
            for metric in ("median_ms", "p95_ms"):
                if native_result[name][metric] >= torch_result[name][metric]:
                    failures.append(
                        f"{group}: round {index + 1} {name} {metric} did not win"
                    )

    return {
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "actions": list(action_names),
        "input_sha256": input_hash,
        "correctness": "passed",
        "rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/reduction_dtype_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/reduction_dtype_benchmark"))
    parser.add_argument(
        "--output-dtypes", nargs="+", choices=SELECTABLE_OUTPUT_DTYPES,
        default=None,
        help="Output dtypes to benchmark (default: float32 float64)",
    )
    parser.add_argument(
        "--complex-scans", action="store_true",
        help="Benchmark complex64 and complex128 scans using default native scan overloads",
    )
    parser.add_argument(
        "--complex-means", action="store_true",
        help="Benchmark explicit and default complex mean overloads",
    )
    parser.add_argument(
        "--complex-scan-dtypes", action="store_true",
        help="Benchmark explicit complex scan output dtypes across real and complex inputs",
    )
    parser.add_argument(
        "--complex-sums", action="store_true",
        help="Benchmark allocating complex-output and complex-input sum dtype overloads",
    )
    args = parser.parse_args()
    explicit_output_dtypes = args.output_dtypes is not None
    if args.complex_sums and explicit_output_dtypes:
        parser.error("--complex-sums cannot be combined with --output-dtypes")
    if args.complex_sums and (
        args.complex_scans or args.complex_means or args.complex_scan_dtypes
    ):
        parser.error("--complex-sums cannot be combined with other complex modes")
    if args.complex_scans and explicit_output_dtypes and args.output_dtypes != list(OUTPUT_DTYPES):
        parser.error("--complex-scans cannot be combined with --output-dtypes")
    if args.complex_scan_dtypes and args.complex_scans:
        parser.error("--complex-scan-dtypes cannot be combined with --complex-scans")
    if args.complex_scan_dtypes and explicit_output_dtypes:
        parser.error("--complex-scan-dtypes cannot be combined with --output-dtypes")
    if args.complex_means and (args.complex_scans or args.complex_scan_dtypes):
        parser.error("--complex-means cannot be combined with complex scan modes")
    if args.complex_means and explicit_output_dtypes:
        parser.error("--complex-means cannot be combined with --output-dtypes")
    if args.output_dtypes is None:
        args.output_dtypes = list(OUTPUT_DTYPES)
    if len(set(args.output_dtypes)) != len(args.output_dtypes):
        parser.error("--output-dtypes must not contain duplicates")
    args.runner = (ROOT / args.runner).resolve() if not args.runner.is_absolute() else args.runner
    args.output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output

    if sys.version_info[:3] != PYTHON_VERSION:
        raise RuntimeError("Requires pinned CPython 3.13.18")
    if (torch.__version__.split("+")[0] != TORCH_VERSION or
            torch.version.git_version != TORCH_GIT_REVISION):
        raise RuntimeError(
            "Requires pinned PyTorch 2.15.0 wheel revision " + TORCH_GIT_REVISION
        )
    if not args.runner.is_file():
        raise RuntimeError(f"Native runner not found: {args.runner}; build it using the docstring command")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)

    source_paths = {str(path): ROOT / path for path in SOURCE_PATHS}
    artifact_paths = {**source_paths, "native_runner": args.runner}
    artifacts_sha256 = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in artifact_paths.items()
    }
    compile_command = [
        "nim", "c", "-d:release", "--path:src", f"--out:{args.runner}",
    ]
    if args.complex_sums:
        compile_command.append("-d:complexSums")
    if args.complex_means:
        compile_command.append("-d:complexMeans")
    elif args.complex_scans:
        compile_command.append("-d:complexScans")
    elif args.complex_scan_dtypes:
        compile_command.append("-d:complexScanDtypes")
    compile_command.append(str(RUNNER_SOURCE))

    failures: list[str] = []
    input_hashes: dict[str, str] = {}
    groups: list[dict[str, Any]] = []
    if args.complex_sums:
        input_dtypes = INPUT_DTYPES + COMPLEX_DTYPES
        output_dtypes = COMPLEX_DTYPES + OUTPUT_DTYPES + INTEGER_OUTPUT_DTYPES + BOOL_OUTPUT_DTYPES
        for input_dtype in input_dtypes:
            source, payload = make_source(input_dtype)
            selected_outputs = (
                COMPLEX_DTYPES if input_dtype not in COMPLEX_DTYPES else output_dtypes
            )
            for output_dtype in selected_outputs:
                groups.append(run_group(
                    args, input_dtype, output_dtype, source, payload, input_hashes, failures,
                ))
    elif args.complex_means:
        input_dtypes = INPUT_DTYPES + COMPLEX_DTYPES
        output_dtypes = COMPLEX_DTYPES + OUTPUT_DTYPES
        for input_dtype in input_dtypes:
            source, payload = make_source(input_dtype)
            selected_outputs = (
                COMPLEX_DTYPES if input_dtype not in COMPLEX_DTYPES
                else COMPLEX_DTYPES + OUTPUT_DTYPES
            )
            for output_dtype in selected_outputs:
                groups.append(run_group(
                    args, input_dtype, output_dtype, source, payload, input_hashes, failures,
                ))
    elif args.complex_scan_dtypes:
        input_dtypes = INPUT_DTYPES + COMPLEX_DTYPES
        output_dtypes = COMPLEX_DTYPES + OUTPUT_DTYPES + INTEGER_OUTPUT_DTYPES
        for input_dtype in input_dtypes:
            source, payload = make_source(input_dtype)
            selected_outputs = (
                COMPLEX_DTYPES + OUTPUT_DTYPES + INTEGER_OUTPUT_DTYPES
                if input_dtype in COMPLEX_DTYPES else COMPLEX_DTYPES
            )
            for output_dtype in selected_outputs:
                groups.append(run_group(
                    args, input_dtype, output_dtype, source, payload, input_hashes, failures,
                ))
    else:
        input_dtypes = COMPLEX_DTYPES if args.complex_scans else INPUT_DTYPES
        output_dtypes = COMPLEX_DTYPES if args.complex_scans else args.output_dtypes
        for input_dtype in input_dtypes:
            source, payload = make_source(input_dtype)
            for output_dtype in ([input_dtype] if args.complex_scans else output_dtypes):
                groups.append(run_group(
                    args, input_dtype, output_dtype, source, payload, input_hashes, failures,
                ))

    changed_artifacts = [
        label for label, path in artifact_paths.items()
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifacts_sha256[label]
    ]
    if changed_artifacts:
        raise RuntimeError("Sources or native runner changed during measurement: " +
                           ", ".join(changed_artifacts))
    for group, digest in input_hashes.items():
        input_path = args.output / f"input_{group}.json"
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Input manifest changed during measurement: {group}")

    action_count = sum(len(group["actions"]) for group in groups)
    comparison_count = action_count * 3 * 2
    if args.complex_means and (
        len(groups) != 30 or action_count != 96 or comparison_count != 576
    ):
        failures.append(
            "complex mean protocol expected 30 groups, 96 actions, and 576 comparisons"
        )
    if args.complex_sums and (
        len(groups) != 42 or action_count != 132 or comparison_count != 792
    ):
        failures.append(
            "complex sum protocol expected 42 groups, 132 actions, and 792 comparisons"
        )
    if args.complex_means or args.complex_sums:
        actions_by_output_dtype = {
            output_dtype: list(dict.fromkeys(
                action
                for group in groups if group["output_dtype"] == output_dtype
                for action in group["actions"]
            ))
            for output_dtype in output_dtypes
        }
    else:
        actions_by_output_dtype = {
            output_dtype: list(COMPLEX_SCAN_CASES if args.complex_scan_dtypes or
                               output_dtype in COMPLEX_DTYPES
                               else actions_for_output(output_dtype))
            for output_dtype in output_dtypes
        }
    protocol: dict[str, Any] = {
        "input_dtypes": list(input_dtypes),
        "output_dtypes": list(output_dtypes),
        "actions_per_output_dtype": actions_by_output_dtype,
        "group_count": len(groups),
        "action_count": action_count,
        "warmup_calls_per_action": WARMUP,
        "samples_per_action_per_round": REPEATS,
        "allocating_calls_per_sample": ITERATIONS,
        "paired_rounds_per_group": 3,
        "comparisons": comparison_count,
    }
    if args.complex_means or args.complex_sums:
        protocol["actions_per_group"] = {
            f'{group["input_dtype"]}_to_{group["output_dtype"]}': list(group["actions"])
            for group in groups
        }
        protocol["actions_per_group_count"] = {
            f'{group["input_dtype"]}_to_{group["output_dtype"]}': len(group["actions"])
            for group in groups
        }
    elif args.complex_scans or args.complex_scan_dtypes:
        protocol["actions_per_group"] = list(COMPLEX_SCAN_CASES)
    elif all(dtype in OUTPUT_DTYPES for dtype in args.output_dtypes):
        protocol["actions_per_group"] = list(CASES)
    report = {
        "torch": torch.__version__,
        "torch_git_revision": torch.version.git_version,
        "python": sys.version,
        "backend": "cpu",
        "protocol": protocol,
        "source_sha256": {
            label: digest for label, digest in artifacts_sha256.items()
            if label != "native_runner"
        },
        "runner_sha256": artifacts_sha256["native_runner"],
        "input_sha256": input_hashes,
        "commands": {
            "build_cwd": str(ROOT),
            "compile": compile_command,
            "invoke_example": [str(args.runner), "<input-manifest.json>"],
        },
        "groups": groups,
        "failures": failures,
        "passed": not failures,
    }
    results_path = args.output / "results.json"
    results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "groups": len(groups),
        "actions": action_count,
        "comparisons": comparison_count,
        "failures": failures,
        "passed": not failures,
        "results": str(results_path),
    }))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
