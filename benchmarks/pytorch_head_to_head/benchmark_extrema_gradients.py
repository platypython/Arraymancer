#!/usr/bin/env python3
"""Benchmark allocating CPU amin/amax forward plus seeded backward against pinned PyTorch.

Build from the repository root with:
  nim c -d:release --path:src --out:build/extrema_gradient_runner benchmarks/pytorch_head_to_head/extrema_gradient_runner.nim
Run the 40 mandatory actions with:
  python benchmarks/pytorch_head_to_head/benchmark_extrema_gradients.py
Each action gets 50 warmups and 100 samples of 50 allocating training calls,
in three alternating backend-order pairs. All medians and p95s must strictly
beat PyTorch in every pair. Inputs and matching seeds are resident and shared.
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

DTYPES = ("float32", "float64")
TORCH_DTYPES = {"float32": torch.float32, "float64": torch.float64}
OPS = ("amin", "amax")
KEEPDIMS = (False, True)
CASES = ("whole", "rows", "columns", "axes_0_2", "axes_2_1")
ACTIONS = tuple(f"{op}_{'keepdim' if keepdim else 'flat'}_{case}"
                for op in OPS for keepdim in KEEPDIMS for case in CASES)
TORCH_VERSION = "2.15.0"
TORCH_GIT_REVISION = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
PYTHON_VERSION = (3, 13, 18)
ROWS, COLUMNS, CUBE_SHAPE = 128, 128, (16, 32, 32)
WARMUP, REPEATS, ITERATIONS, PAIRED_ROUNDS = 50, 100, 50, 3
ROOT = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path("benchmarks/pytorch_head_to_head/extrema_gradient_runner.nim")
SCRIPT_SOURCE = Path("benchmarks/pytorch_head_to_head/benchmark_extrema_gradients.py")
SOURCE_PATHS = tuple(map(Path, (
    "src/arraymancer/autograd/gates_reduce.nim",
    "src/arraymancer/autograd/autograd_common.nim",
    "src/arraymancer/autograd/private/p_extrema_gradient.nim",
    "src/arraymancer/tensor/aggregate.nim",
    "src/arraymancer/tensor/private/p_extrema.nim",
    "src/arraymancer/tensor/private/p_accelerate.nim",
    "nim.cfg", "tests/parity/reference.json", str(RUNNER_SOURCE), str(SCRIPT_SOURCE),
)))


def make_sources(dtype: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    count = ROWS * COLUMNS
    raw = ((torch.arange(count, dtype=torch.int64) % 31) - 15).to(torch.float64)
    raw += (torch.arange(count, dtype=torch.int64) % 7).to(torch.float64) / 8
    matrix = raw.reshape(ROWS, COLUMNS).to(TORCH_DTYPES[dtype])
    cube = matrix.reshape(CUBE_SHAPE)
    return {"matrix": matrix, "cube": cube}, {
        "shape": [ROWS, COLUMNS], "rows": ROWS, "columns": COLUMNS,
        "values": matrix.reshape(-1).tolist(),
    }


def action_spec(name: str) -> tuple[str, bool, str, int | tuple[int, ...] | None]:
    op, mode, case = name.split("_", 2)
    dim = {"whole": None, "rows": 1, "columns": 0,
           "axes_0_2": (0, 2), "axes_2_1": (2, 1)}[case]
    return op, mode == "keepdim", case, dim


def output_shape(case: str, keepdim: bool) -> tuple[int, ...]:
    shape = (ROWS, COLUMNS) if case in ("whole", "rows", "columns") else CUBE_SHAPE
    axes = {"whole": tuple(range(len(shape))), "rows": (1,), "columns": (0,),
            "axes_0_2": (0, 2), "axes_2_1": (2, 1)}[case]
    return tuple(1 if i in axes else n for i, n in enumerate(shape)) if keepdim else tuple(
        n for i, n in enumerate(shape) if i not in axes)


def seed_for(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    count = math.prod(shape)
    values = [(index % 7 - 3) / 4 for index in range(count)]
    return torch.tensor(values, dtype=dtype).reshape(shape)


def negative_zero_indices(value: torch.Tensor) -> list[int]:
    flat = value.reshape(-1)
    return [i for i, x in enumerate(flat.tolist()) if x == 0 and math.copysign(1.0, float(x)) < 0]


def observe(value: torch.Tensor, dtype: str) -> dict[str, Any]:
    return {"dtype": dtype, "shape": list(value.shape), "values": value.reshape(-1).tolist(),
            "negative_zero_indices": negative_zero_indices(value)}


def bind_actions(sources: dict[str, torch.Tensor], dtype: str) -> tuple[dict[str, Callable[[], torch.Tensor]], dict[str, Any]]:
    callables: dict[str, Callable[[], torch.Tensor]] = {}
    seeds: dict[str, Any] = {}
    for name in ACTIONS:
        op, keepdim, case, dim = action_spec(name)
        base = sources["cube"] if case.startswith("axes_") else sources["matrix"]
        seed = seed_for(output_shape(case, keepdim), TORCH_DTYPES[dtype])
        seeds[name] = {"shape": list(seed.shape), "values": seed.reshape(-1).tolist()}
        method = getattr(torch, op)
        callables[name] = lambda base=base, seed=seed, method=method, dim=dim, keepdim=keepdim: _train(base, seed, method, dim, keepdim)
    return callables, seeds


def _train(base: torch.Tensor, seed: torch.Tensor, method: Callable[..., torch.Tensor],
           dim: int | tuple[int, ...] | None, keepdim: bool) -> torch.Tensor:
    leaf = base.detach().requires_grad_(True)
    output = method(leaf, dim=dim, keepdim=keepdim)
    output.backward(seed)
    assert leaf.grad is not None
    return leaf.grad


def run_group(args: argparse.Namespace, dtype: str, failures: list[str]) -> dict[str, Any]:
    sources, payload = make_sources(dtype)
    callables, seeds = bind_actions(sources, dtype)
    manifest = {"dtype": dtype, **payload, "cube_shape": list(CUBE_SHAPE),
                "cube_values": payload["values"], "actions": list(ACTIONS), "seeds": seeds,
                "warmup": WARMUP, "repeats": REPEATS, "iterations": ITERATIONS}
    input_path = args.output / f"input_{dtype}.json"
    input_path.write_text(json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    expected = {name: callables[name]() for name in ACTIONS}
    native_runs = 0

    def native() -> dict[str, Any]:
        nonlocal native_runs
        completed = subprocess.run([str(args.runner), str(input_path.resolve())], check=False,
                                   capture_output=True, text=True, timeout=120)
        try:
            if completed.returncode:
                raise ValueError(f"native runner exited with status {completed.returncode}")
            report = json.loads(completed.stdout)
            if type(report) is not dict or set(report) != set(ACTIONS):
                raise ValueError("native action set differs from strict manifest")
            (args.output / f"native_{dtype}_{native_runs}.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            native_runs += 1
            for name in ACTIONS:
                entry = report[name]
                if type(entry) is not dict or set(entry) != {"observation", "samples_ms", "median_ms", "p95_ms"}:
                    raise ValueError(f"{name}: malformed timing record")
                actual, reference = entry["observation"], expected[name]
                if (type(actual) is not dict or set(actual) != {"dtype", "shape", "values", "negative_zero_indices"}
                        or actual["dtype"] != dtype or actual["shape"] != list(reference.shape)
                        or type(actual["values"]) is not list or len(actual["values"]) != reference.numel()
                        or type(actual["negative_zero_indices"]) is not list
                        or any(type(index) is not int for index in actual["negative_zero_indices"])
                        or actual["negative_zero_indices"] != sorted(set(actual["negative_zero_indices"]))
                        or any(index < 0 or index >= reference.numel() for index in actual["negative_zero_indices"])
                        or actual["negative_zero_indices"] != negative_zero_indices(reference)):
                    raise ValueError(f"{name}: malformed gradient metadata or signed-zero sidecar")
                vals = actual["values"]
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in vals):
                    raise ValueError(f"{name}: gradient values must be finite JSON numbers")
                actual_tensor = torch.tensor(vals, dtype=TORCH_DTYPES[dtype]).reshape(actual["shape"])
                torch.testing.assert_close(actual_tensor, reference, atol=0, rtol=0, equal_nan=False)
                observed_negative_zeros = [
                    i for i, value in enumerate(vals)
                    if value == 0 and math.copysign(1.0, float(value)) < 0
                ]
                if actual["negative_zero_indices"] != observed_negative_zeros:
                    raise ValueError(f"{name}: negative-zero sidecar does not match gradient values")
                samples = entry["samples_ms"]
                if type(samples) is not list or len(samples) != REPEATS or any(
                        type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples):
                    raise ValueError(f"{name}: invalid native samples")
                ordered = sorted(samples)
                if entry["median_ms"] != statistics.median(samples) or entry["p95_ms"] != ordered[math.ceil(.95 * REPEATS) - 1]:
                    raise ValueError(f"{name}: timing statistics do not match samples")
            return report
        except (KeyError, TypeError, ValueError, AssertionError) as error:
            path = args.output / f"failed_native_{dtype}_{native_runs}.json"
            path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
            raise RuntimeError(f"{dtype}: invalid native report retained at {path}: {error}") from error

    def reference() -> dict[str, Any]:
        report = {}
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
            report[name] = {"samples_ms": samples, "median_ms": statistics.median(samples),
                            "p95_ms": ordered[math.ceil(.95 * REPEATS) - 1]}
        return report

    native()
    rounds = []
    for index in range(PAIRED_ROUNDS):
        order = "pytorch-first" if index % 2 == 0 else "arraymancer-first"
        if index % 2 == 0:
            pytorch_result, native_result = reference(), native()
        else:
            native_result, pytorch_result = native(), reference()
        rounds.append({"round": index + 1, "order": order, "pytorch": pytorch_result,
                       "arraymancer": native_result})
        (args.output / f"rounds_{dtype}.json").write_text(json.dumps(rounds, indent=2) + "\n", encoding="utf-8")
        for name in ACTIONS:
            for metric in ("median_ms", "p95_ms"):
                if native_result[name][metric] >= pytorch_result[name][metric]:
                    failures.append(f"{dtype}: round {index + 1} {name} {metric} did not win")
    return {"dtype": dtype, "actions": list(ACTIONS), "input_sha256": input_hash,
            "correctness": "passed", "rounds": rounds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/extrema_gradient_runner"))
    parser.add_argument("--output", type=Path, default=Path("build/extrema_gradient_benchmark"))
    args = parser.parse_args()
    args.runner = (ROOT / args.runner).resolve() if not args.runner.is_absolute() else args.runner
    args.output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    if sys.version_info[:3] != PYTHON_VERSION:
        raise RuntimeError("Requires pinned CPython 3.13.18")
    if torch.__version__.split("+")[0] != TORCH_VERSION or torch.version.git_version != TORCH_GIT_REVISION:
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
    nim_version = subprocess.run(["nim", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    failures, groups, input_hashes = [], [], {}
    for dtype in DTYPES:
        group = run_group(args, dtype, failures)
        groups.append(group)
        input_hashes[dtype] = group["input_sha256"]
    changed = [label for label, path in artifacts.items() if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[label]]
    if changed:
        raise RuntimeError("Sources or runner changed during measurement: " + ", ".join(changed))
    for dtype, digest in input_hashes.items():
        if hashlib.sha256((args.output / f"input_{dtype}.json").read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Input manifest changed during measurement: {dtype}")
    actions = sum(len(group["actions"]) for group in groups)
    comparisons = actions * PAIRED_ROUNDS * 2
    if len(groups) != 2 or actions != 40 or comparisons != 240:
        failures.append("extrema-gradient protocol requires 2 groups, 40 actions and 240 comparisons")
    report = {"torch": torch.__version__, "torch_git_revision": torch.version.git_version,
              "python": sys.version, "backend": "cpu",
              "runtime": {"nim_compiler": nim_version, "python": sys.version,
                          "platform": sys.platform, "machine": __import__("platform").machine()},
              "protocol": {"input_dtypes": list(DTYPES), "actions_per_group": list(ACTIONS),
                           "group_count": len(groups), "action_count": actions, "warmup_calls_per_action": WARMUP,
                           "samples_per_action_per_round": REPEATS, "allocating_calls_per_sample": ITERATIONS,
                           "paired_rounds_per_group": PAIRED_ROUNDS, "comparisons": comparisons},
              "source_sha256": {key: value for key, value in hashes.items() if key != "native_runner"},
              "runner_sha256": hashes["native_runner"], "input_sha256": input_hashes,
              "commands": {"build_cwd": str(ROOT), "build_command_template": compile_command,
                           "invoke_command_template": [str(args.runner), "<input-manifest.json>"]},
              "groups": groups, "failures": failures, "passed": not failures}
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"groups": len(groups), "actions": actions, "comparisons": comparisons,
                      "failures": failures, "passed": not failures, "results": str(result_path)}))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
