#!/usr/bin/env python3
"""Generate float32 Apple Metal observations from the pinned PyTorch MPS oracle.

Requires CPython 3.13.18, torch 2.15.0, and an available Apple MPS device.
This developer-only tool never imports the checked-out PyTorch source tree.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PIN_PATH = Path(__file__).resolve().parents[1] / "reference.json"
TOLERANCE = 1e-6


def _tensor(values: Sequence[int | float], shape: list[int], torch: Any) -> Any:
    return torch.tensor(values, dtype=torch.float32, device="mps").reshape(shape)


def _scalar(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return float(value)


def _observe(tensor: Any, torch: Any) -> dict[str, Any]:
    if tensor.device.type != "mps":
        raise ValueError(f"expected an MPS result, found device {tensor.device}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"expected float32 result, found dtype {tensor.dtype}")
    torch.mps.synchronize()
    host_values = tensor.detach().reshape(-1).cpu().tolist()
    return {
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "values": [_scalar(value) for value in host_values],
    }


def _case(case_id: str, inputs: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case_id,
        "inputs": inputs,
        "expected": expected,
        "atol": TOLERANCE,
        "rtol": TOLERANCE,
    }


def transpose_alias(torch: Any) -> dict[str, Any]:
    values = [1, 2, 3, 4, 5, 6]
    a = _tensor(values, [2, 3], torch)
    view = a.transpose(0, 1)
    mutation = {"index": [0, 1], "value": 9}
    view[tuple(mutation["index"])] = mutation["value"]
    return _case(
        "tensor.transpose_alias",
        {"a": {"shape": [2, 3], "values": values}, "mutation": mutation},
        {"base": _observe(a, torch), "view": _observe(view, torch)},
    )


def broadcast_same_rank(torch: Any) -> dict[str, Any]:
    a_values, b_values = [0, 10], [0, 1, 2]
    a = _tensor(a_values, [2, 1], torch)
    b = _tensor(b_values, [1, 3], torch)
    return _case(
        "tensor.broadcast_same_rank",
        {
            "a": {"shape": [2, 1], "values": a_values},
            "b": {"shape": [1, 3], "values": b_values},
        },
        {"result": _observe(a + b, torch)},
    )


def broadcast_leading_dims(torch: Any) -> dict[str, Any]:
    a_values, b_values = [0, 1, 2, 3, 4, 5], [10, 20, 30]
    a = _tensor(a_values, [2, 3], torch)
    b = _tensor(b_values, [3], torch)
    return _case(
        "tensor.broadcast_leading_dims",
        {
            "a": {"shape": [2, 3], "values": a_values},
            "b": {"shape": [3], "values": b_values},
        },
        {"result": _observe(a + b, torch)},
    )


def matmul_2d(torch: Any) -> dict[str, Any]:
    a_values, b_values = [1, 2, 3, 4], [5, 6, 7, 8]
    a = _tensor(a_values, [2, 2], torch)
    b = _tensor(b_values, [2, 2], torch)
    return _case(
        "tensor.matmul_2d",
        {
            "a": {"shape": [2, 2], "values": a_values},
            "b": {"shape": [2, 2], "values": b_values},
        },
        {"result": _observe(a @ b, torch)},
    )


def mean_gradient(torch: Any) -> dict[str, Any]:
    x_values = [1, 2, 3, 4]
    x = _tensor(x_values, [2, 2], torch).requires_grad_(True)
    with torch.enable_grad():
        mean = x.mean()
        mean.backward()
    return _case(
        "autograd.mean_gradient",
        {"x": {"shape": [2, 2], "values": x_values}},
        {"loss": _observe(mean, torch), "grad": _observe(x.grad, torch)},
    )


def mse_mean(torch: Any) -> dict[str, Any]:
    prediction_values = [1, 2, 3, 4, 5, 6]
    target_values = [0, 0, 0, 0, 0, 0]
    prediction = _tensor(prediction_values, [2, 3], torch).requires_grad_(True)
    target = _tensor(target_values, [2, 3], torch)
    with torch.enable_grad():
        loss = torch.nn.functional.mse_loss(prediction, target, reduction="mean")
        loss.backward()
    return _case(
        "loss.mse_mean",
        {
            "prediction": {"shape": [2, 3], "values": prediction_values},
            "target": {"shape": [2, 3], "values": target_values},
        },
        {"loss": _observe(loss, torch), "grad": _observe(prediction.grad, torch)},
    )


def sgd_step(torch: Any) -> dict[str, Any]:
    weights_values, input_values = [1, 2], [3, 4]
    target_value, learning_rate = 0, 0.01
    weights = _tensor(weights_values, [2], torch).requires_grad_(True)
    inputs = _tensor(input_values, [2], torch)
    target = _tensor([target_value], [], torch)
    prediction = (weights * inputs).sum()
    loss = torch.nn.functional.mse_loss(prediction, target, reduction="mean")
    loss.backward()

    optimizer = torch.optim.SGD([weights], lr=learning_rate, foreach=False)
    optimizer.step()
    if weights.grad is None:
        raise ValueError("SGD unexpectedly removed the gradient")
    gradient_after_step = _observe(weights.grad, torch)
    optimizer.zero_grad(set_to_none=False)
    if weights.grad is None:
        raise ValueError("zero_grad unexpectedly removed the gradient")

    return _case(
        "optim.sgd_step",
        {
            "weights": {"shape": [2], "values": weights_values},
            "input": {"shape": [2], "values": input_values},
            "target": {"shape": [], "values": [target_value]},
            "learning_rate": learning_rate,
        },
        {
            "prediction": _observe(prediction, torch),
            "loss": _observe(loss, torch),
            "grad": gradient_after_step,
            "weights_after_step": _observe(weights, torch),
            "grad_after_step": gradient_after_step,
            "grad_after_zero": _observe(weights.grad, torch),
        },
    )


CASE_BUILDERS = (
    transpose_alias,
    broadcast_same_rank,
    broadcast_leading_dims,
    matmul_2d,
    mean_gradient,
    mse_mean,
    sgd_step,
)


def _validate_runtime(torch: Any, pin: dict[str, Any]) -> tuple[str, str, str]:
    if sys.implementation.name != "cpython":
        raise ValueError(f"CPython required; found {sys.implementation.name}")
    python_version = ".".join(map(str, sys.version_info[:3]))
    if python_version != pin["python"]:
        raise ValueError(f"CPython {pin['python']} required; found {python_version}")
    installed_torch = str(torch.__version__)
    normalized_torch = installed_torch.split("+", 1)[0]
    if normalized_torch != pin["torch"]:
        raise ValueError(f"stable torch {pin['torch']} required; found {installed_torch}")
    git_commit = str(torch.version.git_version or "")
    if not git_commit:
        raise ValueError("installed torch does not report torch.version.git_version")
    if platform.system() != "Darwin" or platform.machine() != pin["apple"]["architecture"]:
        raise ValueError(
            "MPS fixture generation requires macOS arm64; found "
            f"{platform.system()} {platform.machine()}"
        )
    mps_backend = torch.backends.mps
    if not mps_backend.is_built():
        raise ValueError("installed torch was not built with MPS support")
    if not mps_backend.is_available():
        raise ValueError("PyTorch MPS is unavailable on this host")
    return installed_torch, python_version, git_commit


def generate(torch: Any, pin: dict[str, Any]) -> dict[str, Any]:
    torch_version, python_version, git_commit = _validate_runtime(torch, pin)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ValueError("PYTORCH_ENABLE_MPS_FALLBACK must be 0")
    torch.set_num_threads(1)
    cases = [builder(torch) for builder in CASE_BUILDERS]
    device_name = None
    get_device_name = getattr(torch.mps, "get_device_name", None)
    if callable(get_device_name):
        device_name = str(get_device_name())
    return {
        "format": 1,
        "reference": pin,
        "oracle": {
            "torch": torch_version,
            "python": python_version,
            "git_commit": git_commit,
            "torch_config": torch.__config__.show(),
            "device": "mps",
            "device_name": device_name,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        },
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PyTorch 2.15.0 MPS parity observations.")
    parser.add_argument("--output", help="write canonical JSON to PATH instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    try:
        if arguments:
            options = _parser().parse_args(arguments)
        else:
            _parser().print_help(sys.stderr)
            options = _parser().parse_args([])
        with PIN_PATH.open("r", encoding="utf-8") as source:
            pin = json.load(source)
        expected_pin = {
            "format": 1,
            "torch": "2.15.0",
            "source_commit": "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457",
            "python": "3.13.18",
            "apple": {
                "xcode": "27.1",
                "xcode_build": "27A9275",
                "macos_sdk": "27.0",
                "swift": "6.4",
                "minimum_macos": "27.0",
                "architecture": "arm64",
                "metal_api": "4",
            },
        }
        if pin != expected_pin:
            raise ValueError("reference.json is missing fields or does not match the supported exact pin")

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        torch = importlib.import_module("torch")
        document = generate(torch, pin)
        encoded = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if options.output:
            destination = Path(options.output)
            temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as output:
                    output.write(encoded)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        else:
            sys.stdout.write(encoded)
        return 0
    except Exception as error:
        print(f"generate_metal_fixtures.py: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
