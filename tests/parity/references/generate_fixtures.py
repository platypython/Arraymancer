#!/usr/bin/env python3
"""Generate deterministic CPU observations from the pinned PyTorch oracle.

Requires CPython 3.13.18 and torch 2.15.0. This developer-only tool never
imports the checked-out PyTorch source tree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

PIN_PATH = Path(__file__).resolve().parents[1] / "reference.json"
TOLERANCE = 1e-10


def _tensor(values: Sequence[int | float], shape: list[int], torch: Any) -> Any:
    return torch.tensor(values, dtype=torch.float64, device="cpu").reshape(shape)


def _scalar(value: float) -> float | str | bool:
    if isinstance(value, bool):
        return value
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return float(value)


def _observe(tensor: Any, *, include_values: bool = True) -> dict[str, Any]:
    observation = {
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
    }
    if include_values:
        values = tensor.detach().reshape(-1).tolist()
        observation["values"] = [int(value) for value in values] if observation["dtype"] == "int64" else [
            _scalar(value) for value in values
        ]
    return observation


def _case(
    case_id: str,
    area: str,
    inputs: dict[str, Any],
    expected: dict[str, Any],
    *,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    return {"id": case_id, "area": area, "inputs": inputs, "atol": tolerance, "rtol": tolerance, "expected": expected}



def transpose_alias(torch: Any) -> dict[str, Any]:
    a = _tensor([1, 2, 3, 4, 5, 6], [2, 3], torch)
    view = a.transpose(0, 1)
    mutation = {"index": [0, 1], "value": 9}
    view[tuple(mutation["index"])] = mutation["value"]
    return _case("tensor.transpose_alias", "api:tensor_view", {"a": {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]}, "mutation": mutation}, {"base": _observe(a), "view": _observe(view)})


def empty_shape(torch: Any) -> dict[str, Any]:
    result = torch.empty((2, 0, 3), dtype=torch.float64, device="cpu")
    return _case("tensor.empty_shape", "api:tensors", {"shape": [2, 0, 3]}, {"result": _observe(result)})


def elementwise_shapes(torch: Any) -> dict[str, Any]:
    """Observe scalar, offset-strided and empty pointwise results."""
    inputs = {
        "scalar": {"shape": [], "values": [4]},
        "strided": {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]},
        "empty": {"shape": [2, 0, 3], "values": []},
        "empty_transposed": {"shape": [2, 0, 3], "values": []},
    }
    expected = {}
    for name, desc in inputs.items():
        a = _tensor(desc["values"], desc["shape"], torch)
        if name == "strided":
            a = a.transpose(0, 1)[:, 1:]
        elif name == "empty_transposed":
            a = a.transpose(0, 2)
        expected[name] = {
            "exp": _observe(a.exp()), "sqrt": _observe(a.sqrt()),
            "pow": _observe(a.pow(2)), "add": _observe(a + 2),
            "scale": _observe(a * 2), "divide": _observe(a / 2),
            "reverse_divide": _observe(2 / a),
            "cast": _observe(a.to(dtype=torch.float32)),
            "greater": _observe(a > 2), "isnan": _observe(a.isnan()),
        }
    return _case("tensor.elementwise_shapes", "api:torch", inputs, expected)


def unary_math(torch: Any) -> dict[str, Any]:
    """Observe real unary VJPs, including zero and invalid-domain derivatives."""
    inputs: dict[str, Any] = {
        "scalar": {"a": {"shape": [], "values": [4]},
                   "seed": {"shape": [], "values": [3]}},
        "strided": {"a": {"shape": [2, 3], "values": [0.25, 1, 4, 9, 16, 25]},
                    "seed": {"shape": [3, 1], "values": [1, 2, 3]}},
        "empty": {"a": {"shape": [2, 0, 3], "values": []},
                  "seed": {"shape": [3, 0, 2], "values": []}},
        "domain": {"a": {"shape": [4], "values": [-1, -0.0, 0, 4]},
                   "seed": {"shape": [4], "values": [1, 1, 0, 2]}},
    }
    expected: dict[str, Any] = {}
    for name, spec in inputs.items():
        expected[name] = {}
        for operation, function in (("exp", torch.exp), ("ln", torch.log), ("sqrt", torch.sqrt)):
            a = _tensor(spec["a"]["values"], spec["a"]["shape"], torch)
            if name == "strided":
                a = a.transpose(0, 1)[:, 1:]
            elif name == "empty":
                a = a.transpose(0, 2)
            a.requires_grad_(True)
            output = function(a)
            output.backward(_tensor(spec["seed"]["values"], spec["seed"]["shape"], torch))
            expected[name][operation] = {"output": _observe(output), "grad": _observe(a.grad)}
    return _case("autograd.unary_math", "api:autograd", inputs, expected)


def activation_math(torch: Any) -> dict[str, Any]:
    """Observe activation VJPs at zeros, nonfinite inputs and masked NaN seeds."""
    inputs: dict[str, Any] = {
        "scalar": {"a": {"shape": [], "values": [-0.0]},
                   "seed": {"shape": [], "values": [3]}},
        "strided": {"a": {"shape": [2, 3], "values": [-4, -2, 0, 1, 2, 4]},
                    "seed": {"shape": [3, 1], "values": [1, 2, 3]}},
        "empty": {"a": {"shape": [2, 0, 3], "values": []},
                  "seed": {"shape": [3, 0, 2], "values": []}},
        "domain": {"a": {"shape": [4], "values": [-1, 0, 1, 4]},
                   "seed": {"shape": [4], "values": [1, 1, 0, 2]}, "transform": "ln"},
        "infinities": {"a": {"shape": [4], "values": [0, -0.0, -1, 1]},
                       "seed": {"shape": [4], "values": [1, 2, 3, 4]}, "transform": "reciprocal"},
        "masked_nan": {"a": {"shape": [4], "values": [-1, 0, 1, 4]},
                       "seed": {"shape": [4], "values": [-1, -1, -1, -1]}, "seed_transform": "ln"},
    }
    expected: dict[str, Any] = {}
    for name, spec in inputs.items():
        expected[name] = {}
        for operation, function in (("relu", torch.relu), ("sigmoid", torch.sigmoid), ("tanh", torch.tanh)):
            a = _tensor(spec["a"]["values"], spec["a"]["shape"], torch)
            if name == "strided":
                a = a.transpose(0, 1)[:, 1:]
            elif name == "empty":
                a = a.transpose(0, 2)
            if spec.get("transform") == "ln":
                a = a.log()
            elif spec.get("transform") == "reciprocal":
                a = 1 / a
            a.requires_grad_(True)
            seed = _tensor(spec["seed"]["values"], spec["seed"]["shape"], torch)
            if spec.get("seed_transform") == "ln":
                seed = seed.log()
            output = function(a)
            output.backward(seed)
            expected[name][operation] = {
                "output": _observe(output), "grad": _observe(a.grad),
                "negative_zero": _observe((output == 0) & torch.signbit(output)),
            }
    return _case("autograd.activation_math", "api:autograd", inputs, expected)


def scalar_metadata(torch: Any) -> dict[str, Any]:
    result = torch.empty((), dtype=torch.float64, device="cpu")
    return _case("tensor.scalar_metadata", "api:tensors", {"shape": []}, {"result": _observe(result, include_values=False)})


def broadcast_same_rank(torch: Any) -> dict[str, Any]:
    a_values, b_values = [0, 10], [0, 1, 2]
    a, b = _tensor(a_values, [2, 1], torch), _tensor(b_values, [1, 3], torch)
    return _case("tensor.broadcast_same_rank", "api:torch", {"a": {"shape": [2, 1], "values": a_values}, "b": {"shape": [1, 3], "values": b_values}}, {"result": _observe(a + b)})


def broadcast_leading_dims(torch: Any) -> dict[str, Any]:
    a_values, b_values = [0, 1, 2, 3, 4, 5], [10, 20, 30]
    a, b = _tensor(a_values, [2, 3], torch), _tensor(b_values, [3], torch)
    return _case("tensor.broadcast_leading_dims", "api:torch", {"a": {"shape": [2, 3], "values": a_values}, "b": {"shape": [3], "values": b_values}}, {"result": _observe(a + b)})


def matmul_2d(torch: Any) -> dict[str, Any]:
    a_values, b_values = [1, 2, 3, 4], [5, 6, 7, 8]
    a, b = _tensor(a_values, [2, 2], torch), _tensor(b_values, [2, 2], torch)
    return _case("tensor.matmul_2d", "api:torch", {"a": {"shape": [2, 2], "values": a_values}, "b": {"shape": [2, 2], "values": b_values}}, {"result": _observe(a @ b)})


def mean_gradient(torch: Any) -> dict[str, Any]:
    x_values = [1, 2, 3, 4]
    x = _tensor(x_values, [2, 2], torch).requires_grad_(True)
    with torch.enable_grad():
        x.mean().backward()
    return _case("autograd.mean_gradient", "api:autograd", {"x": {"shape": [2, 2], "values": x_values}}, {"grad": _observe(x.grad)})


def mse_mean(torch: Any) -> dict[str, Any]:
    prediction_values, target_values = [1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]
    prediction = _tensor(prediction_values, [2, 3], torch).requires_grad_(True)
    target = _tensor(target_values, [2, 3], torch)
    with torch.enable_grad():
        loss = torch.nn.functional.mse_loss(prediction, target, reduction="mean")
        loss.backward()
    return _case("loss.mse_mean", "api:nn.functional", {"prediction": {"shape": [2, 3], "values": prediction_values}, "target": {"shape": [2, 3], "values": target_values}}, {"loss": _observe(loss), "grad": _observe(prediction.grad)})


def no_grad_exception(torch: Any) -> dict[str, Any]:
    x = _tensor([1], [1], torch).requires_grad_(True)
    try:
        with torch.no_grad():
            raise ValueError("parity probe")
    except ValueError as error:
        if str(error) != "parity probe":
            raise
    return _case("autograd.no_grad_exception", "api:autograd", {"x": {"shape": [1], "values": [1]}}, {"tracking_after_exception": bool(torch.is_grad_enabled())})

def seeded_vjp(torch: Any) -> dict[str, Any]:
    values, seed_values = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]
    x = _tensor(values, [2, 3], torch).requires_grad_(True)
    seed = _tensor(seed_values, [2, 3], torch)
    output = x * x + x
    output.backward(seed)
    return _case("autograd.seeded_vjp", "api:autograd",
                 {"x": {"shape": [2, 3], "values": values}, "seed": {"shape": [2, 3], "values": seed_values}},
                 {"output": _observe(output), "grad": _observe(x.grad)})


def retained_vjp(torch: Any) -> dict[str, Any]:
    x = _tensor([2, 3], [2], torch).requires_grad_(True)
    squared = x * x
    output = squared * squared
    output.backward(_tensor([1, 2], [2], torch), retain_graph=True)
    output.backward(_tensor([3, 4], [2], torch))
    return _case("autograd.retained_vjp", "api:autograd",
                 {"x": {"shape": [2], "values": [2, 3]},
                  "first": {"shape": [2], "values": [1, 2]}, "second": {"shape": [2], "values": [3, 4]}},
                 {"grad": _observe(x.grad)})


def independent_graphs(torch: Any) -> dict[str, Any]:
    x = _tensor([2, 3], [2], torch).requires_grad_(True)
    unused = x / torch.zeros_like(x)
    earlier, later = x * x, x + x
    earlier.backward(torch.ones_like(x))
    later.backward(torch.ones_like(x))
    return _case("autograd.independent_graphs", "api:autograd",
                 {"x": {"shape": [2], "values": [2, 3]}}, {"grad": _observe(x.grad)})


def chunk_seed(torch: Any) -> dict[str, Any]:
    x = _tensor([1, 2, 3, 4], [4], torch).requires_grad_(True)
    parts = x.chunk(2, dim=0)
    parts[1].backward(_tensor([2, 3], [2], torch), retain_graph=True)
    parts[0].backward(_tensor([5, 7], [2], torch))
    return _case("autograd.chunk_seed", "api:autograd",
                 {"x": {"shape": [4], "values": [1, 2, 3, 4]},
                  "first": {"shape": [2], "values": [2, 3]}, "second": {"shape": [2], "values": [5, 7]}},
                 {"grad": _observe(x.grad)})


def slice_overlap(torch: Any) -> dict[str, Any]:
    x = _tensor([1, 2, 3, 4, 5], [5], torch).requires_grad_(True)
    output = x[1:4].sum() + torch.flip(x[::2], [0]).sum()
    output.backward()
    return _case("autograd.slice_overlap", "api:autograd",
                 {"x": {"shape": [5], "values": [1, 2, 3, 4, 5]}},
                 {"output": _observe(output), "grad": _observe(x.grad)})


def index_select_gradient(torch: Any) -> dict[str, Any]:
    x = _tensor([1, 2, 3, 4, 5, 6], [2, 3], torch).requires_grad_(True)
    output = x[:, [2, 0, 2]]
    seed = _tensor([1, 2, 3, 4, 5, 6], [2, 3], torch)
    output.backward(seed)
    return _case("autograd.index_select", "api:autograd",
                 {"x": {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]},
                  "seed": {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]}},
                 {"output": _observe(output), "grad": _observe(x.grad)})


def mask_select_gradient(torch: Any) -> dict[str, Any]:
    x = _tensor([1, 2, 3, 4, 5, 6], [2, 3], torch).transpose(0, 1).requires_grad_(True)
    mask = torch.tensor([[True, False], [False, True], [True, False]], device="cpu")
    output = x[mask]
    output.backward(_tensor([2, 3, 4], [3], torch))
    return _case("autograd.mask_select", "api:autograd",
                 {"x": {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]},
                  "seed": {"shape": [3], "values": [2, 3, 4]}},
                 {"output": _observe(output), "grad": _observe(x.grad)})


def axis_mean(torch: Any) -> dict[str, Any]:
    values = list(range(1, 25))
    x = _tensor(values, [2, 3, 4], torch).requires_grad_(True)
    output = x.mean(dim=1, keepdim=True)
    output.backward(torch.full_like(output, 6))
    return _case("autograd.axis_mean", "api:autograd",
                 {"x": {"shape": [2, 3, 4], "values": values}},
                 {"output": _observe(output), "grad": _observe(x.grad)})


def reduction_scalars(torch: Any) -> dict[str, Any]:
    x = _tensor([1, 2, 3, 4, 5, 6], [2, 3], torch).requires_grad_(True)
    mean, total = x.mean(), x.sum()
    total.backward(_tensor([3], [], torch))
    mean.backward(_tensor([2], [], torch))
    return _case("autograd.reduction_scalars", "api:autograd",
                 {"x": {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]}},
                 {"mean": _observe(mean), "sum": _observe(total), "grad": _observe(x.grad)})



def axis_options(torch: Any) -> dict[str, Any]:
    x = _tensor([1, 2, 3, 4, 5, 6], [2, 3], torch).transpose(0, 1).requires_grad_(True)
    total, average = x.sum(dim=-1), x.mean(dim=-1)
    total.backward(_tensor([2, 4, 6], [3], torch))
    average.backward(_tensor([4, 8, 12], [3], torch))
    return _case("autograd.axis_options", "api:autograd",
                 {"x": {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]},
                  "first": {"shape": [3], "values": [2, 4, 6]},
                  "second": {"shape": [3], "values": [4, 8, 12]}},
                 {"sum": _observe(total), "mean": _observe(average), "grad": _observe(x.grad)})


def _broadcast_arithmetic(torch: Any, *, divide: bool) -> dict[str, Any]:
    a = _tensor([2, 4], [2, 1], torch).requires_grad_(True)
    b = _tensor([1, 2, 4], [3], torch).requires_grad_(True)
    output = a / b if divide else a * b
    output.backward(_tensor([1, 2, 3, 4, 5, 6], [2, 3], torch))
    name = "autograd.broadcast_divide" if divide else "autograd.broadcast_multiply"
    return _case(name, "api:autograd",
                 {"a": {"shape": [2, 1], "values": [2, 4]},
                  "b": {"shape": [3], "values": [1, 2, 4]},
                  "seed": {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]}},
                 {"output": _observe(output), "grad_a": _observe(a.grad), "grad_b": _observe(b.grad)})


def broadcast_multiply(torch: Any) -> dict[str, Any]:
    return _broadcast_arithmetic(torch, divide=False)


def broadcast_divide(torch: Any) -> dict[str, Any]:
    return _broadcast_arithmetic(torch, divide=True)


def _mse_broadcast(torch: Any, reduction: str) -> dict[str, Any]:
    prediction = _tensor([2, 4], [2, 1], torch).requires_grad_(True)
    target = _tensor([1, 2, 3], [3], torch).requires_grad_(True)
    seed_spec = {"shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]} if reduction == "none" else {
        "shape": [], "values": [1]}
    loss = torch.nn.functional.mse_loss(prediction, target, reduction=reduction)
    loss.backward(_tensor(seed_spec["values"], seed_spec["shape"], torch))
    return _case("loss.mse_" + reduction + "_broadcast", "api:nn.functional",
                 {"prediction": {"shape": [2, 1], "values": [2, 4]},
                  "target": {"shape": [3], "values": [1, 2, 3]}, "seed": seed_spec},
                 {"loss": _observe(loss), "grad_prediction": _observe(prediction.grad),
                  "grad_target": _observe(target.grad)})


def mse_broadcast_mean(torch: Any) -> dict[str, Any]:
    return _mse_broadcast(torch, "mean")


def mse_broadcast_sum(torch: Any) -> dict[str, Any]:
    return _mse_broadcast(torch, "sum")


def mse_broadcast_none(torch: Any) -> dict[str, Any]:
    return _mse_broadcast(torch, "none")


def matmul_ranks(torch: Any) -> dict[str, Any]:
    specs = {
        "dot": ([3], [1, 2, 3], [3], [3, 2, 1]),
        "matrix_vector": ([2, 2], [1, 2, 3, 4], [2], [2, 3]),
        "vector_matrix": ([2], [2, 3], [2, 2], [1, 2, 3, 4]),
        "batch_vector": ([2, 2, 2], list(range(1, 9)), [2], [2, 3]),
        "vector_batch": ([2], [2, 3], [2, 2, 2], list(range(1, 9))),
        "broadcast_batches": ([2, 1, 2, 2], list(range(1, 9)), [1, 3, 2, 1], list(range(1, 7))),
        "empty_contract": ([2, 0], [], [0, 3], []),
        "empty_batch": ([1, 2, 3], list(range(1, 7)), [0, 3, 4], []),
    }
    inputs: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for name, (a_shape, a_values, b_shape, b_values) in specs.items():
        a = _tensor(a_values, a_shape, torch).requires_grad_(True)
        b = _tensor(b_values, b_shape, torch).requires_grad_(True)
        output = torch.matmul(a, b)
        seed_values = list(range(1, output.numel() + 1))
        seed_shape = list(output.shape)
        output.backward(_tensor(seed_values, seed_shape, torch))
        inputs[name] = {"a": {"shape": a_shape, "values": a_values},
                        "b": {"shape": b_shape, "values": b_values},
                        "seed": {"shape": seed_shape, "values": seed_values}}
        expected[name] = {"output": _observe(output), "grad_a": _observe(a.grad),
                          "grad_b": _observe(b.grad)}
    return _case("autograd.matmul_ranks", "api:autograd", inputs, expected)


def singleton_views(torch: Any) -> dict[str, Any]:
    x = _tensor([1, 2, 3, 4, 5, 6], [2, 1, 3], torch).transpose(0, 2).requires_grad_(True)
    output = x.squeeze(-2).unsqueeze(-1).squeeze(-1)
    output.backward(_tensor([1, 2, 3, 4, 5, 6], [3, 2], torch))
    noop = x.squeeze(-1)
    noop.backward(torch.ones_like(noop))
    scalar = _tensor([7], [1, 1], torch).requires_grad_(True)
    scalar_output = scalar.squeeze().unsqueeze(-1).squeeze(0).squeeze(-1)
    scalar_output.backward(_tensor([3], [], torch))
    return _case("autograd.singleton_views", "api:autograd",
                 {"x": {"shape": [2, 1, 3], "values": [1, 2, 3, 4, 5, 6]},
                  "seed": {"shape": [3, 2], "values": [1, 2, 3, 4, 5, 6]},
                  "scalar": {"shape": [1, 1], "values": [7]}},
                 {"output": _observe(output), "noop": _observe(noop), "grad": _observe(x.grad),
                  "scalar": _observe(scalar_output), "scalar_grad": _observe(scalar.grad),
                  "shares_storage": output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()})


def reduction_edges(torch: Any) -> dict[str, Any]:
    specs = {
        "scalar": ([], [3], [0, -1]),
        "empty_vector": ([0], [], [0, -1]),
        "empty_axis": ([2, 0, 3], [], [1, -2]),
        "empty_outer": ([0, 2, 3], [], [1, -2]),
        "empty_both": ([0, 0, 3], [], [1, -2]),
        "strided": ([2, 3], [1, 2, 3, 4, 5, 6], [0, -1]),
    }
    inputs, expected = {}, {}
    for name, (shape, values, axes) in specs.items():
        a = _tensor(values, shape, torch)
        if name == "strided":
            a = a.transpose(0, 1)[:, 1:]
        inputs[name] = {"a": {"shape": shape, "values": values}, "axes": axes}
        observed = {op: _observe(getattr(torch, op)(a)) for op in ("sum", "prod", "mean")}
        for axis in axes:
            observed[f"axis_{axis}"] = {
                op: _observe(getattr(torch, op)(a, dim=axis, keepdim=True))
                for op in ("sum", "prod", "mean")
            }
        expected[name] = observed
    return _case("tensor.reduction_edges", "api:torch", inputs, expected)


def product_gradient(torch: Any) -> dict[str, Any]:
    """Observe zero-count, nonfinite and strided full/axis product VJPs."""
    specs = {
        "dense": ([2, 3], [2, 3, -4, 5, -2, 1], "identity", []),
        "batched": ([2, 2, 5], list(range(2, 22)), "identity", []),
        "zeros": ([3, 3], [2, 3, -4, 0, 3, -4, 0, 0, -4], "identity", []),
        "infinity": ([3], [1, 2, 3], "identity", [(0, "+inf")]),
        "zero_infinity": ([3], [0, 2, 3], "identity", [(1, "+inf")]),
        "two_zeros_infinity": ([3], [0, 0, 3], "identity", [(2, "+inf")]),
        "zero_nan": ([3], [0, 2, 3], "identity", [(1, "nan")]),
        "two_zeros_nan": ([3], [0, 0, 3], "identity", [(2, "nan")]),
        "scalar": ([], [-3], "identity", []),
        "scalar_zero": ([], [0], "identity", []),
        "scalar_infinity": ([], [1], "identity", [(0, "+inf")]),
        "scalar_nan": ([], [1], "identity", [(0, "nan")]),
        "singleton_infinity": ([1], [1], "identity", [(0, "+inf")]),
        "singleton_nan": ([1], [1], "identity", [(0, "nan")]),
        "empty_vector": ([0], [], "identity", []),
        "empty_axis": ([2, 0, 3], [], "identity", []),
        "empty_outer": ([0, 2, 3], [], "identity", []),
        "transpose": ([2, 3], [2, 3, -4, 5, -2, 1], "transpose", []),
        "offset": ([2, 3], [2, 3, -4, 5, -2, 1], "offset", []),
        "reverse": ([6], [2, 0, -4, 5, -2, 1], "reverse", []),
        "broadcast": ([1, 3], [2, 0, -4], "broadcast", []),
    }
    inputs, expected = {}, {}
    for name, (shape, values, layout, nonfinite) in specs.items():
        source = _tensor(values, shape, torch)
        for index, value in nonfinite:
            source.reshape(-1)[index] = float(value.replace("+", ""))
        if layout == "transpose":
            source = source.transpose(0, 1)
        elif layout == "offset":
            source = source.transpose(0, 1)[:, 1:]
        elif layout == "reverse":
            source = source.flip([0])
        elif layout == "broadcast":
            source = source.expand(4, 3)
        operations, observations = [], []
        axes: list[int | None] = [None, *range(max(1, source.ndim)), -1]
        for axis in axes:
            for keepdim in ([True] if axis is None else [True, False]):
                for scale in (1, 0, -2):
                    leaf = source.detach().requires_grad_(True)
                    output = leaf.prod() if axis is None else leaf.prod(dim=axis, keepdim=keepdim)
                    seed = ((torch.arange(output.numel(), dtype=source.dtype) % 5) + 1).reshape(output.shape) * scale
                    output.backward(seed)
                    operations.append({"axis": axis, "keepdim": keepdim,
                                       "seed": {"shape": list(seed.shape), "values": seed.flatten().tolist()}})
                    observations.append({"output": _observe(output), "grad": _observe(leaf.grad)})
        inputs[name] = {
            "x": {"shape": shape, "values": values}, "layout": layout,
            "nonfinite": [{"index": index, "value": value} for index, value in nonfinite],
            "operations": operations,
        }
        expected[name] = observations
    return _case("autograd.product_gradient", "api:autograd", inputs, expected)


def _scan_gradient(torch: Any, kind: str) -> dict[str, Any]:
    """Observe inclusive scans and seeded VJPs on shared exact inputs."""
    specs = {
        "dense": ([2, 3], [2, 3, -4, 5, -2, 1], "identity", []),
        "nonfinite_seed": ([2, 3], [2, 3, -4, 5, -2, 1], "identity", []),
        "batched": ([2, 2, 5], list(range(-8, 12)), "identity", []),
        "vector": ([6], [2, 0, -4, 5, -2, 1], "identity", []),
        "nonfinite": ([4], [0, 2, 0, 0], "identity",
                      [(0, "+inf"), (2, "-inf"), (3, "nan")]),
        "nan": ([3], [0, 2, 3], "identity", [(0, "nan")]),
        "scalar": ([], [-3], "identity", []),
        "scalar_negative_zero": ([], [-0.0], "identity", []),
        "scalar_nan": ([], [0], "identity", [(0, "nan")]),
        "empty_vector": ([0], [], "identity", []),
        "empty_axis": ([2, 0, 3], [], "identity", []),
        "empty_outer": ([0, 2, 3], [], "identity", []),
        "transpose": ([2, 3], [2, 3, -4, 5, -2, 1], "transpose", []),
        "offset": ([2, 3], [2, 3, -4, 5, -2, 1], "offset", []),
        "reverse": ([6], [2, 0, -4, 5, -2, 1], "reverse", []),
        "broadcast": ([1, 3], [2, 0, -4], "broadcast", []),
    }
    if kind == "cumprod":
        specs.update({
            "nonfinite_seed": ([2, 3], [2, 0, -4, 0, 0, 1], "identity", []),
            "one_zero": ([4], [2, 0, -3, 4], "identity", []),
            "two_zeros": ([4], [2, 0, 0, 3], "identity", []),
            "leading_zeros": ([4], [0, 2, 0, 3], "identity", []),
            "zero_inf_zero": ([4], [0, 0, 0, 3], "identity", [(1, "+inf")]),
            "inf_zero": ([3], [0, 0, 2], "identity", [(0, "+inf")]),
            "zero_nan": ([4], [2, 0, 0, 0], "identity", [(2, "nan")]),
            "singleton_axis": ([3, 1], [0, 0, 0], "identity",
                               [(0, "nan"), (1, "+inf"), (2, "-inf")]),
        })
    inputs, expected = {}, {}
    for name, (shape, values, layout, nonfinite) in specs.items():
        source = _tensor(values, shape, torch)
        for index, value in nonfinite:
            source.reshape(-1)[index] = float(value.replace("+", ""))
        if layout == "transpose":
            source = source.transpose(0, 1)
        elif layout == "offset":
            source = source.transpose(0, 1)[:, 1:]
        elif layout == "reverse":
            source = source.flip([0])
        elif layout == "broadcast":
            source = source.expand(4, 3)
        operations, observations = [], []
        for axis in [*range(max(1, source.ndim)), -1]:
            for scale in (1, 0, -2):
                seed_variants = ([[(1, "+inf"), (2, "-inf")], [(2, "nan")]]
                                 if name == "nonfinite_seed" else [[]])
                for seed_nonfinite in seed_variants:
                    leaf = source.detach().requires_grad_(True)
                    output = getattr(leaf, kind)(dim=axis)
                    seed = ((torch.arange(output.numel(), dtype=source.dtype) % 5) + 1).reshape(output.shape) * scale
                    seed_values = seed.flatten().tolist()
                    for index, value in seed_nonfinite:
                        seed.reshape(-1)[index] = float(value)
                    output.backward(seed)
                    operations.append({
                        "axis": axis, "seed": {"shape": list(seed.shape), "values": seed_values},
                        "seed_nonfinite": [{"index": index, "value": value}
                                           for index, value in seed_nonfinite],
                    })
                    observations.append({"output": _observe(output), "grad": _observe(leaf.grad)})
        inputs[name] = {
            "x": {"shape": shape, "values": values}, "layout": layout,
            "nonfinite": [{"index": index, "value": value} for index, value in nonfinite],
            "operations": operations,
        }
        expected[name] = observations
    return _case(f"autograd.{kind}_gradient", "api:autograd", inputs, expected)


def cumsum_gradient(torch: Any) -> dict[str, Any]:
    """Record additive scan observations without zero-specific product layouts."""
    return _scan_gradient(torch, "cumsum")


def cumprod_gradient(torch: Any) -> dict[str, Any]:
    """Record rounded-prefix and zero-masked first-order product derivatives."""
    return _scan_gradient(torch, "cumprod")


def integer_scans(torch: Any) -> dict[str, Any]:
    """Observe default CPU integer scan promotion and exact int64 prefixes."""
    specs: dict[str, dict[str, Any]] = {
        "bool": {
            "dtype": "bool", "shape": [2, 3], "words": [0, 1, 0, 2, -1, 0],
            "layout": "identity", "axes": [0, 1, -1],
        },
        "int8": {
            "dtype": "int8", "shape": [2, 3], "words": [-128, 127, -1, 1, 2, 3],
            "layout": "identity", "axes": [0, 1, -1],
        },
        "int16": {
            "dtype": "int16", "shape": [2, 3], "words": [-32768, 32767, -1, 1, 2, 3],
            "layout": "identity", "axes": [0, 1, -1],
        },
        "int32": {
            "dtype": "int32", "shape": [2, 3], "words": [-2147483648, 2147483647, -1, 1, 2, 3],
            "layout": "identity", "axes": [0, 1, -1],
        },
        "int64": {
            "dtype": "int64", "shape": [2, 3], "words": [-3, 4, 2, -5, 0, 1],
            "layout": "identity", "axes": [0, 1, -1],
        },
        "uint8": {
            "dtype": "uint8", "shape": [2, 3], "words": [0, 255, 1, 2, 128, 3],
            "layout": "identity", "axes": [0, 1, -1],
        },
        "uint16": {
            "dtype": "uint16", "shape": [2, 3], "words": [0, 65535, 1, 2, 32768, 3],
            "layout": "identity", "axes": [0, 1, -1],
        },
        "uint32": {
            "dtype": "uint32", "shape": [2, 3], "words": [0, 4294967295, 1, 2, 2147483648, 3],
            "layout": "identity", "axes": [0, 1, -1],
        },
        "uint64": {
            "dtype": "uint64", "shape": [2, 3],
            "words": [0, 9223372036854775807, -9223372036854775808, -1, -2, 1],
            "layout": "identity", "axes": [0, 1, -1],
        },
        "scalar": {
            "dtype": "int64", "shape": [], "words": [-9223372036854775808],
            "layout": "identity", "axes": [0, -1],
        },
        "empty": {
            "dtype": "uint64", "shape": [2, 0, 3], "words": [],
            "layout": "identity", "axes": [0, 1, 2, -1],
        },
        "overflow": {
            "dtype": "int64", "shape": [5],
            "words": [9223372036854775807, 2, -3, 9223372036854775807, -1],
            "layout": "identity", "axes": [0, -1],
        },
        "transpose": {
            "dtype": "int64", "shape": [2, 3], "words": [2, -3, 5, 7, 0, -1],
            "layout": "transpose", "axes": [0, 1, -1],
        },
        "offset": {
            "dtype": "uint16", "shape": [3, 4], "words": list(range(12)),
            "layout": "offset", "axes": [0, 1, -1],
        },
        "reverse": {
            "dtype": "int64", "shape": [5], "words": [3, -2, 0, 4, -1],
            "layout": "reverse", "axes": [0, -1],
        },
        "broadcast": {
            "dtype": "uint8", "shape": [1, 3], "words": [2, 3, 0],
            "layout": "broadcast", "axes": [0, 1, -1],
        },
    }
    torch_dtypes = {name: getattr(torch, name) for name in
                    ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")}
    inputs: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for name, spec in specs.items():
        words = spec["words"]
        source = torch.tensor(words, dtype=torch.int64, device="cpu").to(
            dtype=torch_dtypes[spec["dtype"]]
        ).reshape(spec["shape"])
        layout = spec["layout"]
        if layout == "transpose":
            source = source.transpose(0, 1)
        elif layout == "offset":
            source = source.transpose(0, 1)[:, 1:]
        elif layout == "reverse":
            source = source.flip([0])
        elif layout == "broadcast":
            source = source.expand(4, 3)
        inputs[name] = {**spec}
        observations: dict[str, Any] = {}
        for operation in ("cumsum", "cumprod"):
            axes: dict[str, Any] = {}
            for axis in spec["axes"]:
                output = getattr(source, operation)(dim=axis)
                if output.dtype != torch.int64:
                    raise ValueError(f"{name}.{operation} returned {output.dtype}, expected int64")
                axes[str(axis)] = _observe(output)
            observations[operation] = axes
        expected[name] = observations
    return _case("tensor.integer_scans", "api:torch", inputs, expected, tolerance=0.0)


CASE_BUILDERS: tuple[Callable[[Any], dict[str, Any]], ...] = (
    transpose_alias, empty_shape, scalar_metadata, elementwise_shapes, broadcast_same_rank,
    broadcast_leading_dims, matmul_2d, mean_gradient, mse_mean,
    no_grad_exception, seeded_vjp, retained_vjp, independent_graphs, chunk_seed,
    slice_overlap, index_select_gradient, mask_select_gradient, axis_mean, reduction_scalars,
    axis_options, broadcast_multiply, broadcast_divide,
    mse_broadcast_mean, mse_broadcast_sum, mse_broadcast_none, matmul_ranks, singleton_views,
    unary_math, activation_math, reduction_edges, product_gradient, cumsum_gradient, cumprod_gradient,
    integer_scans,
)


def _validate_runtime(torch: Any, pin: dict[str, Any]) -> tuple[str, str, str]:
    if sys.implementation.name != "cpython":
        raise ValueError(f"CPython required; found {sys.implementation.name}")
    python_version = ".".join(map(str, sys.version_info[:3]))
    if python_version != pin["python"]:
        raise ValueError(f"CPython {pin['python']} required; found {python_version}")
    installed_torch = str(torch.__version__)
    normalized_torch = installed_torch.split("+", 1)[0]
    if normalized_torch != pin["torch"] or any(part in normalized_torch for part in ("a", "b", "rc")):
        raise ValueError(f"stable torch {pin['torch']} required; found {installed_torch}")
    git_commit = str(torch.version.git_version or "")
    if not git_commit:
        raise ValueError("installed torch does not report torch.version.git_version")
    return installed_torch, python_version, git_commit


def generate(torch: Any, pin: dict[str, Any]) -> dict[str, Any]:
    torch_version, python_version, git_commit = _validate_runtime(torch, pin)
    torch.set_num_threads(1)
    cases = [builder(torch) for builder in CASE_BUILDERS]
    return {
        "format": 1,
        "reference": pin,
        "oracle": {"torch": torch_version, "python": python_version, "git_commit": git_commit, "device": "cpu"},
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PyTorch 2.15.0 CPU parity observations.")
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
        import importlib

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
        print(f"generate_fixtures.py: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
