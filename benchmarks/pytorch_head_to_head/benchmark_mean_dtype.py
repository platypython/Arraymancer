#!/usr/bin/env python3
"""Check and time all 66 CPU mean dtype actions against pinned PyTorch.

Build the native runner from the repository root with:
  nim c -d:release --path:src --out:build/mean_dtype_runner benchmarks/pytorch_head_to_head/mean_dtype_runner.nim
Then run the full benchmark (no filters; optional --output and --runner):
  python benchmarks/pytorch_head_to_head/benchmark_mean_dtype.py
The 22 input/output dtype groups each contain mean, mean_rows and mean_columns.
Each paired round alternates backend order; all 100 samples of 50 allocating calls
are retained. The aggregate gate requires every median and p95 to win in all
three paired rounds. Inputs stay resident and explicit output-dtype casting is
part of each timed mean call.
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
OUTPUT_DTYPES = ("float32", "float64")
CASES = ("mean", "mean_rows", "mean_columns")
TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool, "int8": torch.int8, "int16": torch.int16, "int32": torch.int32,
    "int64": torch.int64, "uint8": torch.uint8, "uint16": torch.uint16,
    "uint32": torch.uint32, "uint64": torch.uint64,
    "float32": torch.float32, "float64": torch.float64,
}
TORCH_VERSION = "2.15.0"
TORCH_GIT_REVISION = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
PYTHON_VERSION = (3, 13, 18)
ROWS, COLUMNS = 128, 128
WARMUP, REPEATS, ITERATIONS = 50, 100, 50
ROOT = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/mean_dtype_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_mean_dtype.py")
SOURCE_PATHS = (
    Path("src/arraymancer/tensor/aggregate.nim"),
    Path("src/arraymancer/tensor/private/p_accelerate.nim"),
    Path("src/arraymancer/laser/compiler_optim_hints.nim"),
    RUNNER_SOURCE,
    SCRIPT_SOURCE,
)


def operation(name: str, value: torch.Tensor, output_dtype: str) -> Callable[[], torch.Tensor]:
    """Resolve an allocating mean call; keep requested casting inside the action."""
    dtype = TORCH_DTYPES[output_dtype]
    if name == "mean":
        return lambda: value.mean(dtype=dtype)
    axis = -1 if name == "mean_rows" else -2
    return lambda: value.mean(dim=axis, keepdim=True, dtype=dtype)


def make_source(dtype: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Create one resident, already typed source and its native manifest values."""
    if dtype in INTEGER_DTYPES:
        words = [-2 if index % 4 == 0 else -1 if index % 4 == 1 else
                 1 if index % 4 == 2 else 2 for index in range(ROWS * COLUMNS)]
        source = torch.tensor(words, dtype=torch.int64).reshape(ROWS, COLUMNS)
        return source.to(dtype=TORCH_DTYPES[dtype]), {"words": words}

    source = (1 + ((torch.arange(ROWS * COLUMNS, dtype=torch.float64) % 17) - 8)
              * 0.0001).reshape(ROWS, COLUMNS)
    source = source.to(dtype=TORCH_DTYPES[dtype])
    # Serialize the actual rounded source values so native and Torch see identical inputs.
    return source, {"values": source.flatten().tolist()}


def run_group(
    args: argparse.Namespace,
    input_dtype: str,
    output_dtype: str,
    source: torch.Tensor,
    source_payload: dict[str, Any],
    input_hashes: dict[str, str],
    failures: list[str],
) -> dict[str, Any]:
    group = f"{input_dtype}_to_{output_dtype}"
    manifest: dict[str, Any] = {
        "dtype": input_dtype,
        "output_dtype": output_dtype,
        "rows": ROWS,
        "columns": COLUMNS,
        **source_payload,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "iterations": ITERATIONS,
    }
    input_path = args.output / f"input_{group}.json"
    input_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    input_hashes[group] = input_hash

    callables = {name: operation(name, source, output_dtype) for name in CASES}
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
        for name in CASES:
            actual = report[name]["observation"]
            if actual.get("input_dtype") != input_dtype:
                raise RuntimeError(f"{group}/{name}: input dtype mismatch")
            reference = expected[name]
            if actual["shape"] != list(reference.shape):
                raise RuntimeError(f"{group}/{name}: shape mismatch")
            expected_dtype = str(reference.dtype).removeprefix("torch.")
            if actual.get("dtype") != expected_dtype:
                raise RuntimeError(f"{group}/{name}: dtype mismatch")
            actual_values = torch.tensor(actual["values"], dtype=reference.dtype)
            if actual_values.numel() != reference.numel():
                raise RuntimeError(f"{group}/{name}: value count mismatch")
            actual_values = actual_values.reshape(actual["shape"])
            torch.testing.assert_close(
                actual_values, reference, atol=tolerance, rtol=tolerance,
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

    native()  # Validate all three native actions before timing competition.
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
        for name in CASES:
            for metric in ("median_ms", "p95_ms"):
                if native_result[name][metric] >= torch_result[name][metric]:
                    failures.append(
                        f"{group}: round {index + 1} {name} {metric} did not win"
                    )

    return {
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "input_sha256": input_hash,
        "correctness": "passed",
        "rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/mean_dtype_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/mean_dtype_benchmark"))
    args = parser.parse_args()
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
        str(RUNNER_SOURCE),
    ]

    failures: list[str] = []
    input_hashes: dict[str, str] = {}
    groups: list[dict[str, Any]] = []
    for input_dtype in INPUT_DTYPES:
        source, payload = make_source(input_dtype)
        for output_dtype in OUTPUT_DTYPES:
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

    comparison_count = len(groups) * len(CASES) * 3 * 2
    report = {
        "torch": torch.__version__,
        "torch_git_revision": torch.version.git_version,
        "python": sys.version,
        "backend": "cpu",
        "protocol": {
            "input_dtypes": list(INPUT_DTYPES),
            "output_dtypes": list(OUTPUT_DTYPES),
            "actions_per_group": list(CASES),
            "group_count": len(groups),
            "action_count": len(groups) * len(CASES),
            "warmup_calls_per_action": WARMUP,
            "samples_per_action_per_round": REPEATS,
            "allocating_calls_per_sample": ITERATIONS,
            "paired_rounds_per_group": 3,
            "comparisons": comparison_count,
        },
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
        "actions": len(groups) * len(CASES),
        "comparisons": comparison_count,
        "failures": failures,
        "passed": not failures,
        "results": str(results_path),
    }))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
