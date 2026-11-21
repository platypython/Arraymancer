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

def _real_dtype_layout(source: Any, layout: str) -> Any:
    if layout == "transpose":
        return source.transpose(0, 1)
    if layout == "offset":
        return source.transpose(0, 1)[:, 1:]
    if layout == "reverse":
        return source.flip([0])
    if layout == "broadcast":
        return source.expand(4, 3)
    return source


def float64_reductions(torch: Any) -> dict[str, Any]:
    """Observe CPU float64 reductions, scans, and deterministic seeded VJPs."""
    precision_values = [
        1.0000001, 0.9999998, 1.0000003, 0.9999996, 1.0000005,
        0.9999994, 1.0000007, 0.9999992, 1.0000009, 0.9999990,
        1.0000011, 0.9999988, 1.0000013, 0.9999986, 1.0000015,
    ]
    reduction_specs: dict[str, tuple[list[int], list[float], str, list[int | None]]] = {
        "packed_odd": ([3, 5], precision_values, "identity", [None, 0, 1, -1]),
        "transpose": ([2, 3], [1.25, -0.75, 2.5, 0.5, -1.5, 3.0],
                      "transpose", [None, 0, -1]),
        "offset": ([3, 4], [0.75, 1.25, -0.5, 2.0, 1.5, -1.0,
                             0.25, 3.0, -2.0, 0.5, 1.75, -0.25],
                   "offset", [None, 0, -1]),
        "reverse": ([5], [1.25, -0.5, 2.0, 0.75, -1.5], "reverse", [None, 0, -1]),
        "broadcast": ([1, 3], [1.25, -0.5, 2.0], "broadcast", [None, 0, 1, -1]),
        "scalar": ([], [1.0000001], "identity", [None, 0, -1]),
        "empty": ([2, 0, 3], [], "identity", [None, 0, 1, 2, -1]),
    }
    reductions_input: dict[str, Any] = {}
    reductions_expected: dict[str, Any] = {}
    for name, (shape, values, layout, axes) in reduction_specs.items():
        source = _real_dtype_layout(_tensor(values, shape, torch), layout)
        reductions_input[name] = {
            "shape": shape, "values": values, "layout": layout, "axes": axes,
        }
        reduction_observations: dict[str, Any] = {}
        for operation in ("sum", "prod", "mean"):
            reducer = getattr(torch, operation)
            by_axis: dict[str, Any] = {"all": _observe(reducer(source))}
            for axis in axes:
                if axis is not None:
                    by_axis[str(axis)] = _observe(reducer(source, dim=axis, keepdim=True))
            reduction_observations[operation] = by_axis
        reductions_expected[name] = reduction_observations

    scan_specs: dict[str, tuple[list[int], list[float], str, list[int]]] = {
        "packed_odd": ([3, 5], precision_values, "identity", [0, 1, -1]),
        "scan_precision": ([5], [1e8, 1.0, 1.0, 1.0, 0.0001], "identity", [0, -1]),
        "transpose": ([2, 3], [2.0, 0.5, -1.0, 1.5, 0.25, 3.0],
                      "transpose", [0, -1]),
        "offset": ([3, 4], [0.75, 1.25, -0.5, 2.0, 1.5, -1.0,
                             0.25, 3.0, -2.0, 0.5, 1.75, -0.25],
                   "offset", [0, -1]),
        "reverse": ([5], [2.0, 0.0, -3.0, 4.0, 0.5], "reverse", [0, -1]),
        "broadcast": ([1, 3], [1.25, 0.0, -2.0], "broadcast", [0, 1, -1]),
        "scalar": ([], [-3.0], "identity", [0, -1]),
        "empty": ([2, 0, 3], [], "identity", [0, 1, 2, -1]),
    }
    scans_input: dict[str, Any] = {}
    scans_expected: dict[str, Any] = {}
    for name, (shape, values, layout, axes) in scan_specs.items():
        source = _real_dtype_layout(_tensor(values, shape, torch), layout)
        scans_input[name] = {
            "shape": shape, "values": values, "layout": layout, "axes": axes,
        }
        scan_observations: dict[str, Any] = {}
        for operation in ("cumsum", "cumprod"):
            by_axis = {
                str(axis): _observe(getattr(source, operation)(dim=axis))
                for axis in axes
            }
            scan_observations[operation] = by_axis
        scans_expected[name] = scan_observations

    reduction_vjp_specs: dict[str, dict[str, Any]] = {
        "offset": {
            "shape": [3, 4],
            "values": [0.75, 1.25, -0.5, 2.0, 1.5, -1.0,
                       0.25, 3.0, -2.0, 0.5, 1.75, -0.25],
            "layout": "offset", "nonfinite": [],
            "operations": [
                {"operation": "sum", "axis": None, "keepdim": True},
                {"operation": "sum", "axis": 1, "keepdim": False},
                {"operation": "mean", "axis": None, "keepdim": True},
                {"operation": "mean", "axis": 0, "keepdim": True},
                {"operation": "prod", "axis": 1, "keepdim": False},
            ],
        },
        "one_zero": {
            "shape": [5], "values": [2.0, 0.0, -3.0, 4.0, 0.5],
            "layout": "reverse", "nonfinite": [],
            "operations": [
                {"operation": "prod", "axis": None, "keepdim": True},
                {"operation": "prod", "axis": 0, "keepdim": True},
            ],
        },
        "multiple_zeros": {
            "shape": [2, 3], "values": [0.0, 2.0, 0.0, 4.0, 5.0, 6.0],
            "layout": "transpose", "nonfinite": [],
            "operations": [
                {"operation": "prod", "axis": None, "keepdim": True},
                {"operation": "prod", "axis": 0, "keepdim": True},
                {"operation": "prod", "axis": 1, "keepdim": True},
            ],
        },
        "infinity": {
            "shape": [3], "values": [1.0, 2.0, 3.0], "layout": "identity",
            "nonfinite": [{"index": 0, "value": "+inf"}],
            "operations": [{"operation": "prod", "axis": None, "keepdim": True}],
        },
        "nan": {
            "shape": [3], "values": [1.0, 2.0, 3.0], "layout": "identity",
            "nonfinite": [{"index": 0, "value": "nan"}],
            "operations": [{"operation": "prod", "axis": None, "keepdim": True}],
        },
        "zero_infinity": {
            "shape": [3], "values": [0.0, 2.0, 3.0], "layout": "identity",
            "nonfinite": [{"index": 1, "value": "+inf"}],
            "operations": [{"operation": "prod", "axis": None, "keepdim": True}],
        },
        "zero_nan": {
            "shape": [3], "values": [0.0, 2.0, 3.0], "layout": "identity",
            "nonfinite": [{"index": 1, "value": "nan"}],
            "operations": [{"operation": "prod", "axis": 0, "keepdim": True}],
        },
    }
    reduction_vjps_input: dict[str, Any] = {}
    reduction_vjps_expected: dict[str, Any] = {}
    for name, spec in reduction_vjp_specs.items():
        source = _tensor(spec["values"], spec["shape"], torch)
        for special in spec["nonfinite"]:
            source.reshape(-1)[special["index"]] = float(special["value"].replace("+", ""))
        source = _real_dtype_layout(source, spec["layout"])
        operation_inputs: list[dict[str, Any]] = []
        reduction_vjp_observations: list[dict[str, Any]] = []
        for operation in spec["operations"]:
            leaf = source.detach().requires_grad_(True)
            axis = operation["axis"]
            if axis is None:
                output = getattr(leaf, operation["operation"])()
            else:
                output = getattr(leaf, operation["operation"])(
                    dim=axis, keepdim=operation["keepdim"]
                )
            seed = (
                torch.arange(output.numel(), dtype=torch.float64, device="cpu") + 1.25
            ).reshape(output.shape)
            output.backward(seed)
            operation_inputs.append({
                **operation,
                "seed": {"shape": list(seed.shape), "values": seed.reshape(-1).tolist()},
            })
            reduction_vjp_observations.append({"output": _observe(output), "grad": _observe(leaf.grad)})
        reduction_vjps_input[name] = {
            "x": {"shape": spec["shape"], "values": spec["values"]},
            "layout": spec["layout"], "nonfinite": spec["nonfinite"],
            "operations": operation_inputs,
        }
        reduction_vjps_expected[name] = reduction_vjp_observations

    scan_vjp_specs: dict[str, dict[str, Any]] = {
        "packed_odd": {
            "shape": [3, 5], "values": precision_values, "layout": "identity",
            "nonfinite": [],
            "operations": [
                {"operation": "cumsum", "axis": 0},
                {"operation": "cumsum", "axis": 1},
                {"operation": "cumprod", "axis": 1},
            ],
        },
        "broadcast": {
            "shape": [1, 3], "values": [1.25, 0.0, -2.0], "layout": "broadcast",
            "nonfinite": [],
            "operations": [
                {"operation": "cumsum", "axis": 0},
                {"operation": "cumprod", "axis": 1},
            ],
        },
    }
    scan_vjps_input: dict[str, Any] = {}
    scan_vjps_expected: dict[str, Any] = {}
    for name, spec in scan_vjp_specs.items():
        source = _real_dtype_layout(_tensor(spec["values"], spec["shape"], torch), spec["layout"])
        operation_inputs: list[dict[str, Any]] = []
        scan_vjp_observations: list[dict[str, Any]] = []
        for operation in spec["operations"]:
            leaf = source.detach().requires_grad_(True)
            output = getattr(leaf, operation["operation"])(dim=operation["axis"])
            seed = (
                torch.arange(output.numel(), dtype=torch.float64, device="cpu") + 1.75
            ).reshape(output.shape)
            output.backward(seed)
            operation_inputs.append({
                **operation,
                "seed": {"shape": list(seed.shape), "values": seed.reshape(-1).tolist()},
            })
            scan_vjp_observations.append({"output": _observe(output), "grad": _observe(leaf.grad)})
        scan_vjps_input[name] = {
            "x": {"shape": spec["shape"], "values": spec["values"]},
            "layout": spec["layout"], "nonfinite": spec["nonfinite"],
            "operations": operation_inputs,
        }
        scan_vjps_expected[name] = scan_vjp_observations

    return _case(
        "tensor.float64_reductions",
        "api:autograd",
        {
            "reductions": reductions_input,
            "scans": scans_input,
            "reduction_vjps": reduction_vjps_input,
            "scan_vjps": scan_vjps_input,
        },
        {
            "reductions": reductions_expected,
            "scans": scans_expected,
            "reduction_vjps": reduction_vjps_expected,
            "scan_vjps": scan_vjps_expected,
        },
        tolerance=1e-12,
    )




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

def extrema_gradient(torch: Any) -> dict[str, Any]:
    """Observe pinned float32/float64 CPU amin/amax seeded first-order VJPs."""
    specs: dict[str, dict[str, Any]] = {
        "rows_ties": {
            "shape": [2, 3], "values": [1, 1, 3, 3, 3, 2], "axes": 1,
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "values", "values": [2, -3]},
        },
        "columns_ties": {
            "shape": [3, 2], "values": [-2, 4, -2, 4, 1, 3], "axes": 0,
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "multi_axes": {
            "shape": [2, 3, 2], "values": [index % 3 for index in range(12)], "axes": [0, -1],
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "empty_axes_all": {
            "shape": [7], "values": [3] * 7, "axes": [],
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "whole": {
            "shape": [2, 3], "values": [1, 1, 3, 3, 3, 2], "axes": None,
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "nan_row": {
            "shape": [2, 3], "values": [1, 0, -2, 3, 3, 1], "axes": 1,
            "layout": "identity", "nonfinite": [{"index": 1, "value": "nan"}],
            "seed": {"kind": "values", "values": [2, -3]},
        },
        "infinite_values": {
            "shape": [2, 3], "values": [0, 0, 3, 0, 1, 0], "axes": 1,
            "layout": "identity", "nonfinite": [
                {"index": 0, "value": "-inf"}, {"index": 1, "value": "-inf"},
                {"index": 3, "value": "+inf"}, {"index": 5, "value": "+inf"},
            ],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "infinite_seed": {
            "shape": [2, 3], "values": [1, 1, 3, 3, 3, 2], "axes": 1,
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "values", "values": ["+inf", "-inf"]},
        },
        "negative_zero_seed": {
            "shape": [2, 3], "values": [1, 1, 3, 3, 3, 2], "axes": 1,
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "constant", "value": "-0"},
        },
        "nan_seed": {
            "shape": [2, 3], "values": [1, 1, 3, 3, 3, 2], "axes": 0,
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "constant", "value": "nan"},
        },
        "scalar_negative_zero": {
            "shape": [], "values": [-0.0], "axes": -1,
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "scalar_nan": {
            "shape": [], "values": [0], "axes": None,
            "layout": "identity", "nonfinite": [{"index": 0, "value": "nan"}],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "scalar_infinity": {
            "shape": [], "values": [0], "axes": 0,
            "layout": "identity", "nonfinite": [{"index": 0, "value": "+inf"}],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "empty_outer": {
            "shape": [0, 3], "values": [], "axes": 1,
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "empty_middle": {
            "shape": [2, 0, 3], "values": [], "axes": [0, 2],
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "transpose": {
            "shape": [2, 3], "values": [1, 1, 3, 3, 3, 2], "axes": 1,
            "layout": "transpose", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "offset": {
            "shape": [3, 4], "values": [index % 3 for index in range(12)], "axes": 0,
            "layout": "offset", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "reverse": {
            "shape": [5], "values": [2, 1, 3, 1, 3], "axes": 0,
            "layout": "reverse", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "broadcast": {
            "shape": [1, 3], "values": [1, 3, 2], "axes": 0,
            "layout": "broadcast", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "permuted": {
            "shape": [4, 8, 3], "values": [index % 7 for index in range(96)], "axes": [0, 1],
            "layout": "permute", "nonfinite": [],
            "seed": {"kind": "constant", "value": -3.0},
        },
        "strided_seed": {
            "shape": [2, 3], "values": [1, 1, 3, 3, 3, 2], "axes": 0,
            "layout": "identity", "nonfinite": [],
            "seed": {"kind": "strided_sequence"},
        },
    }
    operations = [
        {"operation": operation, "keepdim": keepdim}
        for operation in ("amin", "amax")
        for keepdim in (False, True)
    ]
    inputs: dict[str, Any] = {}
    for name, spec in specs.items():
        inputs[name] = {
            "x": {"shape": spec["shape"], "values": spec["values"]},
            "layout": spec["layout"],
            "nonfinite": spec["nonfinite"],
            "axes": spec["axes"],
            "seed": spec["seed"],
            "operations": operations,
        }

    expected: dict[str, Any] = {}
    for dtype_name in ("float32", "float64"):
        dtype = getattr(torch, dtype_name)
        expected[dtype_name] = {}
        for name, spec in specs.items():
            source = torch.tensor(spec["values"], dtype=dtype, device="cpu").reshape(spec["shape"])
            for special in spec["nonfinite"]:
                source.reshape(-1)[special["index"]] = _extrema_gradient_number(special["value"])
            layout = spec["layout"]
            if layout == "transpose":
                source = source.transpose(0, 1)
            elif layout == "offset":
                source = source.transpose(0, 1)[:, 1:]
            elif layout == "reverse":
                source = source.flip([0])
            elif layout == "broadcast":
                source = source.expand(4, 3)
            elif layout == "permute":
                source = source.permute(2, 1, 0)

            observations = []
            for request in operations:
                leaf = source.detach().requires_grad_(True)
                kwargs: dict[str, Any] = {"keepdim": request["keepdim"]}
                axes = spec["axes"]
                if axes is not None:
                    kwargs["dim"] = tuple(axes) if isinstance(axes, list) else axes
                output = getattr(torch, request["operation"])(leaf, **kwargs)
                seed_spec = spec["seed"]
                if seed_spec["kind"] == "constant":
                    seed = torch.full_like(output, _extrema_gradient_number(seed_spec["value"]))
                elif seed_spec["kind"] == "values":
                    seed = torch.tensor(
                        [_extrema_gradient_number(value) for value in seed_spec["values"]],
                        dtype=dtype, device="cpu",
                    ).reshape(output.shape)
                else:
                    seed = torch.arange(output.numel() * 2, dtype=dtype, device="cpu")[::2].reshape(
                        output.shape
                    )
                output.backward(seed)
                observations.append({
                    "output": _observe(output),
                    "grad": _observe_extrema_gradient(leaf.grad),
                })
            expected[dtype_name][name] = observations

    return _case("autograd.extrema", "api:autograd", inputs, expected, tolerance=1e-7)

def paired_extrema_gradient(torch: Any) -> dict[str, Any]:
    """Observe first-index VJPs from the CPU dimension min/max APIs."""
    finite_seeds = ["finite", "negative_zero", "nonfinite", "strided"]
    basic_seeds = ["finite", "negative_zero", "nonfinite"]
    specs: dict[str, dict[str, Any]] = {
        "ties": {
            "x": {"shape": [2, 3], "values": [1, 1, 2, 3, 3, 0]},
            "layout": "identity", "nonfinite": [], "axes": [1, -1],
            "seed_kinds": finite_seeds,
        },
        "transpose": {
            "x": {"shape": [2, 3], "values": [1, 1, 2, 3, 3, 0]},
            "layout": "transpose", "nonfinite": [], "axes": [0, -2],
            "seed_kinds": finite_seeds,
        },
        "reverse": {
            "x": {"shape": [2, 3], "values": [1, 1, 2, 3, 3, 0]},
            "layout": "reverse", "nonfinite": [], "axes": [1, -1],
            "seed_kinds": finite_seeds,
        },
        "broadcast": {
            "x": {"shape": [1, 3], "values": [1, 1, 2]},
            "layout": "broadcast", "nonfinite": [], "axes": [1, -1, 0, -2],
            "seed_kinds": finite_seeds,
        },
        "nan_first": {
            "x": {"shape": [2, 3], "values": [0, 0, 0, 0, -1, 2]},
            "layout": "identity",
            "nonfinite": [
                {"index": 1, "value": "nan"}, {"index": 2, "value": "nan"},
                {"index": 3, "value": "nan"},
            ],
            "axes": [1, -1], "seed_kinds": finite_seeds,
        },
        "zero_ties": {
            "x": {"shape": [2, 3], "values": [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0]},
            "layout": "identity", "nonfinite": [], "axes": [1, -1],
            "seed_kinds": finite_seeds,
        },
        "scalar": {
            "x": {"shape": [], "values": [0]},
            "layout": "identity", "nonfinite": [{"index": 0, "value": "nan"}],
            "axes": [-1, 0], "seed_kinds": basic_seeds,
        },
        "empty": {
            "x": {"shape": [0, 3], "values": []},
            "layout": "identity", "nonfinite": [], "axes": [1, -1],
            "seed_kinds": basic_seeds,
        },
        "cube": {
            "x": {"shape": [2, 2, 3], "values": [index % 3 for index in range(12)]},
            "layout": "identity", "nonfinite": [],
            "axes": [1, -2, 0, -3, 2, -1], "seed_kinds": basic_seeds,
        },
        "offset": {
            "x": {"shape": [3, 4], "values": [index % 3 for index in range(12)]},
            "layout": "offset", "nonfinite": [], "axes": [0, -2],
            "seed_kinds": finite_seeds,
        },
        "infinities": {
            "x": {"shape": [2, 4], "values": [0] * 8},
            "layout": "identity",
            "nonfinite": [
                {"index": 0, "value": "-inf"}, {"index": 1, "value": "-inf"},
                {"index": 3, "value": "+inf"}, {"index": 4, "value": "+inf"},
                {"index": 6, "value": "+inf"}, {"index": 7, "value": "-inf"},
            ],
            "axes": [1, -1], "seed_kinds": finite_seeds,
        },
    }
    inputs = {
        name: {
            "x": spec["x"],
            "layout": spec["layout"],
            "nonfinite": spec["nonfinite"],
            "axes": spec["axes"],
            "seed_kinds": spec["seed_kinds"],
            "invalid_seed": {"kind": "shape_mismatch"},
        }
        for name, spec in specs.items()
    }
    expected: dict[str, Any] = {}
    for dtype_name in ("float32", "float64"):
        dtype = getattr(torch, dtype_name)
        expected[dtype_name] = {}
        for name, spec in specs.items():
            source = torch.tensor(spec["x"]["values"], dtype=dtype, device="cpu").reshape(
                spec["x"]["shape"]
            )
            for special in spec["nonfinite"]:
                source.reshape(-1)[special["index"]] = _extrema_gradient_number(special["value"])
            layout = spec["layout"]
            if layout == "transpose":
                source = source.transpose(0, 1)
            elif layout == "offset":
                source = source.transpose(0, 1)[:, 1:]
            elif layout == "reverse":
                source = source.flip([1])
            elif layout == "broadcast":
                source = source.expand(2, 3)

            observations: dict[str, Any] = {}
            for operation in ("min", "max"):
                for axis in spec["axes"]:
                    for keepdim in (False, True):
                        request_id = (
                            f"{operation}_axis_{axis}_{'keep' if keepdim else 'drop'}"
                        )
                        for seed_kind in spec["seed_kinds"]:
                            leaf = source.detach().requires_grad_(True)
                            pair = (
                                torch.min(leaf, dim=axis, keepdim=keepdim)
                                if operation == "min"
                                else torch.max(leaf, dim=axis, keepdim=keepdim)
                            )
                            count = pair.values.numel()
                            finite_values = [index % 5 - 2 for index in range(count)]
                            if seed_kind == "finite":
                                seed = torch.tensor(finite_values, dtype=dtype, device="cpu").reshape(
                                    pair.values.shape
                                )
                            elif seed_kind == "negative_zero":
                                seed = torch.full_like(pair.values, -0.0)
                            elif seed_kind == "nonfinite":
                                values = [
                                    float("inf") if index % 2 == 0 else float("nan")
                                    for index in range(count)
                                ]
                                seed = torch.tensor(values, dtype=dtype, device="cpu").reshape(
                                    pair.values.shape
                                )
                            elif seed_kind == "strided":
                                seed_storage = torch.zeros(count * 2 + 1, dtype=dtype, device="cpu")
                                seed_storage[1::2] = torch.tensor(
                                    finite_values, dtype=dtype, device="cpu"
                                )
                                seed = seed_storage[1::2].reshape(pair.values.shape)
                            else:
                                raise ValueError(f"unknown paired extrema seed kind: {seed_kind}")
                            pair.values.backward(seed)
                            if leaf.grad is None:
                                raise ValueError("paired extrema backward did not populate the input gradient")
                            observations[f"{request_id}_seed_{seed_kind}"] = {
                                "values": _observe_extrema_gradient(pair.values),
                                "indices": _observe_integer_output(pair.indices),
                                "seed": _observe_extrema_gradient(seed),
                                "gradient": _observe_extrema_gradient(leaf.grad),
                                "indices_require_grad": pair.indices.requires_grad,
                            }

                        leaf = source.detach().requires_grad_(True)
                        pair = (
                            torch.min(leaf, dim=axis, keepdim=keepdim)
                            if operation == "min"
                            else torch.max(leaf, dim=axis, keepdim=keepdim)
                        )
                        invalid_seed = torch.zeros(
                            pair.values.numel() + 1, dtype=dtype, device="cpu"
                        )
                        try:
                            pair.values.backward(invalid_seed)
                        except RuntimeError as error:
                            message = str(error).lower()
                            if "shape" not in message and "size" not in message:
                                raise
                            observations[f"{request_id}_invalid_seed"] = {
                                "error": "seed_shape_mismatch"
                            }
                        else:
                            raise ValueError("PyTorch accepted a mismatched paired extrema seed shape")
            expected[dtype_name][name] = observations

    return _case("autograd.paired_extrema", "api:autograd", inputs, expected, tolerance=0.0)




def _extrema_gradient_number(value: int | float | str) -> float:
    if isinstance(value, str):
        return {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf"), "-0": -0.0}[value]
    return float(value)


def _observe_extrema_gradient(tensor: Any) -> dict[str, Any]:
    observation = _observe(tensor)
    values = tensor.detach().reshape(-1).tolist()
    observation["negative_zero_indices"] = [
        index for index, value in enumerate(values)
        if value == 0 and math.copysign(1.0, float(value)) < 0
    ]
    return observation


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


def integer_reductions(torch: Any) -> dict[str, Any]:
    """Observe default CPU integer sum/product promotion and exact int64 outputs."""
    specs: dict[str, dict[str, Any]] = {
        "bool": {
            "dtype": "bool", "shape": [2, 3], "words": [0, 1, 0, 2, -1, 0],
            "layout": "identity", "axes": [None, 0, 1, -1, -2],
        },
        "int8": {
            "dtype": "int8", "shape": [2, 3], "words": [-128, 127, -1, 1, 2, 3],
            "layout": "identity", "axes": [None, 0, 1, -1, -2],
        },
        "int16": {
            "dtype": "int16", "shape": [2, 3], "words": [-32768, 32767, -1, 1, 2, 3],
            "layout": "identity", "axes": [None, 0, 1, -1, -2],
        },
        "int32": {
            "dtype": "int32", "shape": [2, 3], "words": [-2147483648, 2147483647, -1, 1, 2, 3],
            "layout": "identity", "axes": [None, 0, 1, -1, -2],
        },
        "int64": {
            "dtype": "int64", "shape": [2, 3], "words": [-3, 4, 2, -5, 0, 1],
            "layout": "identity", "axes": [None, 0, 1, -1, -2],
        },
        "uint8": {
            "dtype": "uint8", "shape": [2, 3], "words": [0, 255, 1, 2, 128, 3],
            "layout": "identity", "axes": [None, 0, 1, -1, -2],
        },
        "uint16": {
            "dtype": "uint16", "shape": [2, 3], "words": [0, 65535, 1, 2, 32768, 3],
            "layout": "identity", "axes": [None, 0, 1, -1, -2],
        },
        "uint32": {
            "dtype": "uint32", "shape": [2, 3],
            "words": [0, 4294967295, 1, 2, 2147483648, 3],
            "layout": "identity", "axes": [None, 0, 1, -1, -2],
        },
        "uint64": {
            "dtype": "uint64", "shape": [2, 3],
            "words": [0, 9223372036854775807, -9223372036854775808, -1, -2, 1],
            "layout": "identity", "axes": [None, 0, 1, -1, -2],
        },
        "scalar": {
            "dtype": "int64", "shape": [], "words": [-9223372036854775808],
            "layout": "identity", "axes": [None, 0, -1],
        },
        "empty": {
            "dtype": "uint64", "shape": [2, 0, 3], "words": [],
            "layout": "identity", "axes": [None, 0, 1, 2, -1, -2, -3],
        },
        "overflow": {
            "dtype": "int64", "shape": [5],
            "words": [9223372036854775807, 2, -3, 9223372036854775807, -1],
            "layout": "identity", "axes": [None, 0, -1],
        },
        "transpose": {
            "dtype": "int64", "shape": [2, 3], "words": [2, -3, 5, 7, 0, -1],
            "layout": "transpose", "axes": [None, 0, 1, -1, -2],
        },
        "offset": {
            "dtype": "uint16", "shape": [3, 4], "words": list(range(12)),
            "layout": "offset", "axes": [None, 0, 1, -1, -2],
        },
        "reverse": {
            "dtype": "int64", "shape": [5], "words": [3, -2, 0, 4, -1],
            "layout": "reverse", "axes": [None, 0, -1],
        },
        "broadcast": {
            "dtype": "uint8", "shape": [1, 3], "words": [2, 3, 0],
            "layout": "broadcast", "axes": [None, 0, 1, -1, -2],
        },
    }
    torch_dtypes = {name: getattr(torch, name) for name in
                    ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")}
    inputs: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for name, spec in specs.items():
        source = torch.tensor(spec["words"], dtype=torch.int64, device="cpu").to(
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
        operations: dict[str, Any] = {}
        for operation in ("sum", "prod"):
            axes: dict[str, Any] = {}
            for axis in spec["axes"]:
                output = getattr(source, operation)() if axis is None else getattr(source, operation)(
                    dim=axis, keepdim=True
                )
                if output.dtype != torch.int64:
                    raise ValueError(f"{name}.{operation} returned {output.dtype}, expected int64")
                axes["all" if axis is None else str(axis)] = _observe(output)
            operations[operation] = axes
        expected[name] = operations
    return _case("tensor.integer_reductions", "api:torch", inputs, expected, tolerance=0.0)

def _default_mean_dtype_rejection(source: Any, axis: int | None, label: str) -> dict[str, str]:
    try:
        if axis is None:
            source.mean()
        else:
            source.mean(dim=axis)
    except RuntimeError as error:
        expected_message = (
            "mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got:"
        )
        if not str(error).startswith(expected_message):
            raise
        return {"error": "default_dtype_rejected"}
    raise ValueError(f"{label} unexpectedly accepted a default integer mean")


_INTEGER_DTYPE_NAMES = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
_REAL_DTYPE_SPECS: dict[str, dict[str, Any]] = {
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
        "dtype": "uint32", "shape": [2, 3],
        "words": [0, 4294967295, 1, 2, 2147483648, 3],
        "layout": "identity", "axes": [0, 1, -1],
    },
    "uint64": {
        "dtype": "uint64", "shape": [2, 3],
        "words": [0, 9223372036854775807, -9223372036854775808, -1, -2, 1],
        "layout": "identity", "axes": [0, 1, -1],
    },
    "float32": {
        "dtype": "float32", "shape": [2, 3],
        "values": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        "layout": "identity", "axes": [0, 1, -1],
    },
    "float64": {
        "dtype": "float64", "shape": [2, 3],
        "values": [0.1, -1.5, 2.0, 3.25, 4.5, -0.5],
        "layout": "transpose", "axes": [0, 1, -1],
    },
    "float64_precision": {
        "dtype": "float64", "shape": [2], "values": [16777217, -16777216],
        "layout": "identity", "axes": [0, -1],
    },
    "uint64_upper": {
        "dtype": "uint64", "shape": [2], "words": [-1, -1],
        "layout": "identity", "axes": [0, -1],
    },
    "packed_odd": {
        "dtype": "int16", "shape": [3, 5], "words": list(range(-7, 8)),
        "layout": "identity", "axes": [0, 1, -1],
    },
    "transpose": {
        "dtype": "uint16", "shape": [2, 3], "words": [0, 65535, 1, 2, 32768, 3],
        "layout": "transpose", "axes": [0, -1],
    },
    "offset": {
        "dtype": "uint32", "shape": [3, 4], "words": list(range(12)),
        "layout": "offset", "axes": [0, -1],
    },
    "reverse": {
        "dtype": "int64", "shape": [5], "words": [3, -2, 0, 4, -1],
        "layout": "reverse", "axes": [0, -1],
    },
    "broadcast": {
        "dtype": "uint8", "shape": [1, 3], "words": [2, 3, 0],
        "layout": "broadcast", "axes": [0, 1, -1],
    },
    "scalar": {
        "dtype": "int64", "shape": [], "words": [-1],
        "layout": "identity", "axes": [0, -1],
    },
    "empty": {
        "dtype": "uint64", "shape": [2, 0, 3], "words": [],
        "layout": "identity", "axes": [0, 1, 2, -1],
    },
}
_REAL_SCAN_DTYPE_SPECS: dict[str, dict[str, Any]] = {
    "float32_wide_scan": {
        "dtype": "float32", "shape": [3], "values": [16777216, 1, -16777216],
        "layout": "identity", "axes": [0, -1],
    },
    "float64_overflow": {
        "dtype": "float64", "shape": [3], "values": [1e20, 1e20, 1e-20],
        "layout": "identity", "axes": [0, -1],
    },
    "float64_underflow": {
        "dtype": "float64", "shape": [3], "values": [1e-30, 1e-30, 1e20],
        "layout": "identity", "axes": [0, -1],
    },
    "nonfinite": {
        "dtype": "float64", "shape": [4], "values": [1, 2, 3, 4],
        "layout": "identity", "axes": [0, -1],
        "nonfinite": [
            {"index": 1, "value": "+inf"},
            {"index": 2, "value": "-inf"},
        ],
    },
    "scalar_negative_zero": {
        "dtype": "float64", "shape": [], "values": [-0.0],
        "layout": "identity", "axes": [0, -1],
    },
    "vector_negative_zero": {
        "dtype": "float64", "shape": [1], "values": [-0.0],
        "layout": "identity", "axes": [0, -1],
    },
}
_REAL_PROD_DTYPE_SPECS: dict[str, dict[str, Any]] = {
    "float64_overflow": {
        "dtype": "float64", "shape": [3], "values": [1e20, 1e20, 1e-20],
        "layout": "identity", "axes": [0, -1],
    },
    "float64_underflow": {
        "dtype": "float64", "shape": [3], "values": [1e-30, 1e-30, 1e30],
        "layout": "identity", "axes": [0, -1],
    },
    "float64_zero_nan": {
        "dtype": "float64", "shape": [3], "values": [1e20, 1e20, 0.0],
        "layout": "identity", "axes": [0, -1],
    },
    "uint64_upper_zero": {
        "dtype": "uint64", "shape": [3], "words": [-1, -1, 0],
        "layout": "identity", "axes": [0, -1],
    },
    "lane_range": {
        "dtype": "float64", "shape": [16],
        "values": [1e-20, 1e-20, 1e20, 1e20] + [1.0] * 12,
        "layout": "identity", "axes": [0, -1],
    },
    "early_zero4": {
        "dtype": "float64", "shape": [4], "values": [0.0, 1e20, 1e20, 1e20],
        "layout": "identity", "axes": [0, -1],
    },
    "zero_columns": {
        "dtype": "float64", "shape": [4, 5],
        "values": [0.0] * 5 + [1e200] * 15,
        "layout": "identity", "axes": [0, 1, -1],
    },
    "scalar_negative_zero": {
        "dtype": "float64", "shape": [], "values": [-0.0],
        "layout": "identity", "axes": [0, -1],
    },
    "cast_before_product": {
        "dtype": "float64", "shape": [64], "values": [1.0 + 2**-24] * 64,
        "layout": "identity", "axes": [0, -1],
    },
}


def _real_dtype_source(torch: Any, spec: dict[str, Any]) -> Any:
    dtype_name = spec["dtype"]
    dtype = getattr(torch, dtype_name)
    if dtype_name in _INTEGER_DTYPE_NAMES:
        source = torch.tensor(spec["words"], dtype=torch.int64, device="cpu").to(dtype=dtype).reshape(
            spec["shape"]
        )
    else:
        source = torch.tensor(spec["values"], dtype=dtype, device="cpu").reshape(spec["shape"])
    for special in spec.get("nonfinite", []):
        value = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}[special["value"]]
        source.reshape(-1)[special["index"]] = value
    return _real_dtype_layout(source, spec["layout"])

def _real_dtype_scans(torch: Any, operation: str) -> dict[str, Any]:
    torch_dtypes = {name: getattr(torch, name) for name in ("float32", "float64")}
    inputs: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    specs = {**_REAL_DTYPE_SPECS, **_REAL_SCAN_DTYPE_SPECS}
    for name, spec in specs.items():
        source = _real_dtype_source(torch, spec)
        inputs[name] = {**spec}
        axes: dict[str, Any] = {}
        for axis in spec["axes"]:
            by_dtype: dict[str, Any] = {}
            for output_dtype, torch_dtype in torch_dtypes.items():
                cast_source = source.to(dtype=torch_dtype)
                directions: dict[str, Any] = {}
                for reverse in (False, True):
                    if reverse and source.ndim:
                        scan_source = torch.flip(cast_source, dims=[axis])
                        output = getattr(scan_source, operation)(dim=axis, dtype=torch_dtype)
                        output = torch.flip(output, dims=[axis])
                    else:
                        output = getattr(source, operation)(dim=axis, dtype=torch_dtype)
                    directions["reverse" if reverse else "forward"] = _observe(output)
                by_dtype[output_dtype] = directions
            axes[str(axis)] = by_dtype
        expected[name] = {"axes": axes}
    return _case(f"tensor.{operation}_dtype", "api:torch", inputs, expected, tolerance=1e-6)


def cumsum_dtype(torch: Any) -> dict[str, Any]:
    """Observe explicit CPU cumulative-sum output dtypes and scan directions."""
    return _real_dtype_scans(torch, "cumsum")


def cumprod_dtype(torch: Any) -> dict[str, Any]:
    """Observe explicit CPU cumulative-product output dtypes and scan directions."""
    return _real_dtype_scans(torch, "cumprod")




def _real_dtype_reductions(torch: Any, operation: str) -> dict[str, Any]:
    torch_dtypes = {name: getattr(torch, name) for name in (*_INTEGER_DTYPE_NAMES, "float32", "float64")}
    inputs: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    specs = {**_REAL_DTYPE_SPECS,
             **(_REAL_PROD_DTYPE_SPECS if operation == "prod" else {})}
    for name, spec in specs.items():
        dtype_name = spec["dtype"]
        source = _real_dtype_source(torch, spec)
        inputs[name] = {**spec}

        reducer = getattr(source, operation)
        whole: dict[str, Any] = {}
        for output_dtype in ("float32", "float64"):
            output = reducer(dtype=torch_dtypes[output_dtype])
            whole[output_dtype] = _observe(output)
        by_axis: dict[str, Any] = {}
        for axis in spec["axes"]:
            by_dtype: dict[str, Any] = {}
            for output_dtype in ("float32", "float64"):
                output = reducer(dim=axis, keepdim=True, dtype=torch_dtypes[output_dtype])
                by_dtype[output_dtype] = _observe(output)
            by_axis[str(axis)] = by_dtype
        observations: dict[str, Any] = {"all": whole, "axes": by_axis}

        if operation == "mean" and dtype_name in _INTEGER_DTYPE_NAMES:
            defaults: dict[str, Any] = {
                "all": _default_mean_dtype_rejection(source, None, name),
                "axes": {},
            }
            for axis in spec["axes"]:
                defaults["axes"][str(axis)] = _default_mean_dtype_rejection(source, axis, name)
            observations["default"] = defaults
        expected[name] = observations
    return _case(f"tensor.{operation}_dtype", "api:torch", inputs, expected, tolerance=1e-6)


def mean_dtype(torch: Any) -> dict[str, Any]:
    """Observe explicit CPU mean casts, views, and default integer rejection."""
    return _real_dtype_reductions(torch, "mean")


def sum_dtype(torch: Any) -> dict[str, Any]:
    """Observe explicit CPU sum casts and views."""
    return _real_dtype_reductions(torch, "sum")


def prod_dtype(torch: Any) -> dict[str, Any]:
    """Observe explicit CPU product casts, views, and reduction grouping."""
    return _real_dtype_reductions(torch, "prod")


_INTEGER_OUTPUT_DTYPE_NAMES = (
    "bool", "uint8", "int8", "int16", "int32", "int64", "uint16", "uint32", "uint64"
)
_INTEGER_OUTPUT_SOURCE_NAMES = (*_INTEGER_DTYPE_NAMES, "float32", "float64")
_INTEGER_OUTPUT_TORCH_NAMES = {
    "bool": "Bool", "uint8": "Byte", "int8": "Char", "int16": "Short",
    "int32": "Int", "int64": "Long", "uint16": "UInt16", "uint32": "UInt32",
    "uint64": "UInt64",
}


def _observe_integer_output(tensor: Any) -> dict[str, Any]:
    """Record every integer exactly; uint64 values use signed two's-complement words."""
    observation = _observe(tensor, include_values=False)
    dtype_name = observation["dtype"]
    values = tensor.detach().reshape(-1).tolist()
    if dtype_name == "bool":
        observation["values"] = [bool(value) for value in values]
    elif dtype_name == "uint64":
        observation["word_encoding"] = "signed_twos_complement"
        observation["values"] = [
            int(value) if int(value) < (1 << 63) else int(value) - (1 << 64)
            for value in values
        ]
    else:
        observation["values"] = [int(value) for value in values]
    return observation


def _unsupported_integer_output(
    call: Callable[[], Any], operation: str, dtype_name: str, should_reject: bool, label: str,
    observer: Callable[[Any], dict[str, Any]] = _observe_integer_output,
) -> dict[str, Any]:
    if should_reject:
        try:
            call()
        except RuntimeError as error:
            operator = {
                "sum": '"sum_cpu"',
                "prod": '"prod_out_cpu"',
                "cumsum": '"cumsum_out_cpu"',
                "cumprod": '"cumprod_out_cpu"',
            }[operation]
            expected = f"{operator} not implemented for '{_INTEGER_OUTPUT_TORCH_NAMES[dtype_name]}'"
            if expected not in str(error):
                raise
            return {"error": "unsupported_output_dtype"}
        raise ValueError(f"{label} unexpectedly accepted output dtype {dtype_name}")
    return observer(call())


def _integer_mean_rejection(call: Callable[[], Any], label: str) -> dict[str, str]:
    try:
        call()
    except RuntimeError as error:
        message = str(error)
        if not message.startswith("mean():") or "floating point or complex dtype" not in message:
            raise
        return {"error": "integer_mean_dtype_rejected"}
    raise ValueError(f"{label} unexpectedly accepted an integer mean dtype")


def _integer_invalid_scan_axis(
    call: Callable[[], Any], label: str
) -> dict[str, str]:
    try:
        call()
    except (IndexError, RuntimeError) as error:
        if "dimension out of range" not in str(error).lower():
            raise
        return {"error": "invalid_axis"}
    raise ValueError(f"{label} unexpectedly accepted an invalid scan axis")


def _integer_output_call(
    torch: Any, source: Any, operation: str, dtype: Any, axis: int | None, reverse: bool = False
) -> Any:
    if operation in ("sum", "prod"):
        reducer = source.sum if operation == "sum" else source.prod
        if axis is None:
            return reducer(dtype=dtype)
        return reducer(dim=axis, keepdim=True, dtype=dtype)
    if operation in ("cumsum", "cumprod"):
        scan_source = source
        if reverse and source.ndim:
            scan_source = torch.flip(source.to(dtype=dtype), dims=[axis])
        output = getattr(scan_source, operation)(dim=axis, dtype=dtype)
        if reverse and source.ndim:
            output = torch.flip(output, dims=[axis])
        return output
    raise ValueError(f"unsupported integer output operation: {operation}")


def _integer_output_case_for_input(torch: Any, spec: dict[str, Any], label: str) -> dict[str, Any]:
    source = _real_dtype_source(torch, spec)
    targets = spec.get("output_dtypes", _INTEGER_OUTPUT_DTYPE_NAMES)
    expected: dict[str, Any] = {"reductions": {}, "scans": {}, "mean": {}}
    nonempty = source.numel() > 0
    scan_rejects_nonempty = nonempty and source.ndim > 0
    for operation in ("sum", "prod"):
        by_operation: dict[str, Any] = {"all": {}, "axes": {}}
        for output_name in targets:
            output_dtype = getattr(torch, output_name)
            reject = output_name in ("uint16", "uint32", "uint64") and nonempty
            by_operation["all"][output_name] = _unsupported_integer_output(
                lambda: _integer_output_call(torch, source, operation, output_dtype, None),
                operation, output_name, reject, f"{label}.{operation}.all",
            )
            for axis in spec["axes"]:
                by_operation["axes"].setdefault(str(axis), {})[output_name] = _unsupported_integer_output(
                    lambda axis=axis: _integer_output_call(
                        torch, source, operation, output_dtype, axis
                    ),
                    operation, output_name, reject, f"{label}.{operation}.{axis}",
                )
        expected["reductions"][operation] = by_operation

    for operation in ("cumsum", "cumprod"):
        by_operation = {"axes": {}}
        for axis in spec["axes"]:
            for output_name in targets:
                output_dtype = getattr(torch, output_name)
                reject = output_name in ("bool", "uint16", "uint32", "uint64") and scan_rejects_nonempty
                directions = {}
                for reverse in (False, True):
                    key = "reverse" if reverse else "forward"
                    directions[key] = _unsupported_integer_output(
                        lambda axis=axis, reverse=reverse: _integer_output_call(
                            torch, source, operation, output_dtype, axis, reverse
                        ),
                        operation, output_name, reject,
                        f"{label}.{operation}.{axis}.{key}",
                    )
                by_operation["axes"].setdefault(str(axis), {})[output_name] = directions
        expected["scans"][operation] = by_operation

    mean: dict[str, Any] = {"all": {}, "axes": {}}
    for output_name in targets:
        output_dtype = getattr(torch, output_name)
        mean["all"][output_name] = _integer_mean_rejection(
            lambda: source.mean(dtype=output_dtype), f"{label}.mean.all.{output_name}"
        )
        for axis in spec["axes"]:
            mean["axes"].setdefault(str(axis), {})[output_name] = _integer_mean_rejection(
                lambda axis=axis: source.mean(dim=axis, keepdim=True, dtype=output_dtype),
                f"{label}.mean.{axis}.{output_name}",
            )
    expected["mean"] = mean

    if "invalid_axes" in spec:
        invalid_axes = {}
        for operation in ("cumsum", "cumprod"):
            by_operation = {}
            for axis in spec["invalid_axes"]:
                by_axis = {}
                for output_name in targets:
                    output_dtype = getattr(torch, output_name)
                    by_axis[output_name] = _integer_invalid_scan_axis(
                        lambda: _integer_output_call(
                            torch, source, operation, output_dtype, axis
                        ),
                        f"{label}.{operation}.{axis}.{output_name}",
                    )
                by_operation[str(axis)] = by_axis
            invalid_axes[operation] = by_operation
        expected["invalid_scan_axes"] = invalid_axes
    return expected






def integer_output_dtype(torch: Any) -> dict[str, Any]:
    """Observe every explicit CPU integer/bool output dtype across source and shape classes."""
    inputs = {
        name: {**_REAL_DTYPE_SPECS[name]}
        for name in _INTEGER_OUTPUT_SOURCE_NAMES
    }
    inputs["float32"]["values"] = [1.9, 2.9, 3.4, 4.1, 0.25, 5.8]
    inputs["float64"]["values"] = [0.1, 1.5, 2.0, 3.25, 4.5, 0.5]
    inputs["int64"]["words"] = [-(1 << 63), (1 << 63) - 1, -1, 1, 2, 3]
    inputs["transpose"] = {**_REAL_DTYPE_SPECS["transpose"]}
    inputs["offset"] = {**_REAL_DTYPE_SPECS["offset"]}
    inputs["reverse"] = {**_REAL_DTYPE_SPECS["reverse"]}
    inputs["broadcast"] = {**_REAL_DTYPE_SPECS["broadcast"]}
    inputs["wrap"] = {
        "dtype": "int64", "shape": [2, 3], "words": [127, 3, 255, 2, -128, -1],
        "layout": "identity", "axes": [0, 1, -1],
    }
    inputs["scalar"] = {
        "dtype": "uint64", "shape": [], "words": [-1], "layout": "identity",
        "axes": [0, -1], "invalid_axes": [1],
    }
    inputs["empty"] = {
        "dtype": "uint64", "shape": [2, 0, 3], "words": [], "layout": "identity",
        "axes": [0, 1, 2, -1, -2, -3], "invalid_axes": [3],
    }
    inputs["singleton"] = {
        "dtype": "int32", "shape": [1], "words": [2], "layout": "identity",
        "axes": [0, -1],
    }
    inputs["bool_nan"] = {
        "dtype": "float64", "shape": [3], "values": [0.0, 1.0, 0.0],
        "layout": "identity", "axes": [0, -1], "output_dtypes": ["bool"],
        "nonfinite": [{"index": 1, "value": "nan"}],
    }
    inputs["float_signed"] = {
        "dtype": "float64", "shape": [4], "values": [-128.9, 127.9, -1.9, 2.9],
        "layout": "identity", "axes": [0, -1],
        "output_dtypes": ["int8", "int16", "int32", "int64"],
    }
    expected = {
        name: _integer_output_case_for_input(torch, spec, name)
        for name, spec in inputs.items()
    }
    return _case("tensor.integer_output_dtype", "api:torch", inputs, expected, tolerance=0.0)


def _complex_scalar(value: float) -> float | str:
    return _scalar(float(value))


def _observe_complex(tensor: Any) -> dict[str, Any]:
    values = tensor.detach().reshape(-1).tolist()
    negative_zero = {
        "real": [
            index for index, value in enumerate(values)
            if value.real == 0.0 and math.copysign(1.0, value.real) < 0.0
        ],
        "imag": [
            index for index, value in enumerate(values)
            if value.imag == 0.0 and math.copysign(1.0, value.imag) < 0.0
        ],
    }
    observation = {
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "values": [
            {"real": _complex_scalar(value.real), "imag": _complex_scalar(value.imag)}
            for value in values
        ],
        "negative_zero": negative_zero,
    }
    return observation


def _complex_scan_spec(
    dtype: str,
    shape: list[int],
    values: list[list[int | float]],
    axes: list[int],
    layout: str = "identity",
    nonfinite: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    dimensions = max(1, len(shape))
    spec: dict[str, Any] = {
        "dtype": dtype,
        "shape": shape,
        "values": values,
        "layout": layout,
        "axes": axes,
        "invalid_axes": [dimensions, -dimensions - 1],
    }
    if nonfinite is not None:
        spec["nonfinite"] = nonfinite
    return spec


def _complex_scan_source(torch: Any, spec: dict[str, Any]) -> Any:
    values = [complex(float(real), float(imag)) for real, imag in spec["values"]]
    for special in spec.get("nonfinite", []):
        value = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}[
            special["value"]
        ]
        index = special["index"]
        current = values[index]
        values[index] = (
            complex(value, current.imag)
            if special["component"] == "real"
            else complex(current.real, value)
        )
    source = torch.tensor(
        values, dtype=getattr(torch, spec["dtype"]), device="cpu"
    ).reshape(spec["shape"])
    return _real_dtype_layout(source, spec["layout"])


def _complex_invalid_scan_axis(call: Callable[[], Any], label: str) -> dict[str, str]:
    try:
        call()
    except (IndexError, RuntimeError) as error:
        if "dimension out of range" not in str(error).lower():
            raise
        return {"error": "invalid_axis"}
    raise ValueError(f"{label} unexpectedly accepted an invalid scan axis")


def complex_scans(torch: Any) -> dict[str, Any]:
    """Observe default CPU complex scans, their component signs, and axis rejections."""
    specs = {
        "complex64": _complex_scan_spec(
            "complex64", [2, 3],
            [[1.0, 2.0], [-3.0, 0.5], [0.25, -1.0],
             [2.0, -2.0], [0.0, 3.0], [-1.5, 0.25]],
            [0, 1, -1],
        ),
        "complex128": _complex_scan_spec(
            "complex128", [2, 3],
            [[0.5, -1.0], [2.0, 3.0], [-4.0, 0.25],
             [1.5, 2.5], [-0.5, -2.0], [3.0, 0.75]],
            [0, 1, -1],
        ),
        "transpose": _complex_scan_spec(
            "complex64", [2, 3],
            [[2.0, -1.0], [3.0, 4.0], [-2.0, 0.5],
             [1.0, 2.0], [-4.0, 1.0], [0.25, -3.0]],
            [0, 1, -1], "transpose",
        ),
        "offset": _complex_scan_spec(
            "complex128", [3, 4],
            [[float(i - 5), float((i % 4) - 2)] for i in range(12)],
            [0, 1, -1], "offset",
        ),
        "reverse": _complex_scan_spec(
            "complex64", [5],
            [[3.0, -1.0], [-2.0, 4.0], [0.5, 2.0], [1.0, -3.0], [-1.5, 0.25]],
            [0, -1], "reverse",
        ),
        "broadcast": _complex_scan_spec(
            "complex128", [1, 3], [[2.0, -1.0], [0.5, 3.0], [-2.0, 0.25]],
            [0, 1, -1], "broadcast",
        ),
        "scalar": _complex_scan_spec(
            "complex128", [], [[-0.0, -0.0]], [0, -1],
        ),
        "empty": _complex_scan_spec(
            "complex64", [2, 0, 3], [], [0, 1, 2, -1],
        ),
        "singleton": _complex_scan_spec(
            "complex64", [1], [[-0.0, -0.0]], [0, -1],
        ),
        "precision_cancellation": _complex_scan_spec(
            "complex64", [3],
            [[16777216.0, 16777216.0], [1.0, 1.0], [-16777216.0, -16777216.0]],
            [0, -1],
        ),
        "overflow_recovery": _complex_scan_spec(
            "complex64", [3], [[1e20, 0.0], [1e20, 0.0], [1e-20, 0.0]], [0, -1],
        ),
        "underflow_recovery": _complex_scan_spec(
            "complex64", [3], [[1e-30, 0.0], [1e-30, 0.0], [1e30, 0.0]], [0, -1],
        ),
        "signed_zero": _complex_scan_spec(
            "complex128", [2], [[-0.0, -0.0], [-0.0, -0.0]], [0, -1],
        ),
        "nonfinite": _complex_scan_spec(
            "complex128", [5],
            [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            [0, -1],
            nonfinite=[
                {"index": 1, "component": "real", "value": "+inf"},
                {"index": 2, "component": "real", "value": "-inf"},
                {"index": 4, "component": "imag", "value": "nan"},
            ],
        ),
    }
    inputs: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for name, spec in specs.items():
        source = _complex_scan_source(torch, spec)
        inputs[name] = {**spec}
        observations: dict[str, Any] = {}
        for operation in ("cumsum", "cumprod"):
            axes: dict[str, Any] = {}
            for axis in spec["axes"]:
                directions: dict[str, Any] = {}
                for reverse in (False, True):
                    if reverse and source.ndim:
                        scan_source = torch.flip(source, dims=[axis])
                        output = getattr(scan_source, operation)(dim=axis)
                        output = torch.flip(output, dims=[axis])
                    else:
                        output = getattr(source, operation)(dim=axis)
                    if output.dtype != getattr(torch, spec["dtype"]):
                        raise ValueError(
                            f"{name}.{operation} returned {output.dtype}, expected {spec['dtype']}"
                        )
                    directions["reverse" if reverse else "forward"] = _observe_complex(output)
                axes[str(axis)] = directions
            invalid_axes = {
                str(axis): _complex_invalid_scan_axis(
                    lambda axis=axis: getattr(source, operation)(dim=axis),
                    f"{name}.{operation}.{axis}",
                )
                for axis in spec["invalid_axes"]
            }
            observations[operation] = {"axes": axes, "invalid_axes": invalid_axes}
        expected[name] = observations
    return _case("tensor.complex_scans", "api:torch", inputs, expected, tolerance=1e-6)


def _complex_scan_dtype_spec(
    dtype: str, shape: list[int], values: list[list[int | float]], axes: list[int],
    output_dtypes: tuple[str, ...], layout: str = "identity",
    nonfinite: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec = _complex_scan_spec(dtype, shape, values, axes, layout, nonfinite)
    spec["output_dtypes"] = list(output_dtypes)
    return spec


def _real_scan_dtype_spec(
    dtype: str, shape: list[int], axes: list[int], output_dtypes: tuple[str, ...],
    *, values: list[int | float] | None = None, words: list[int] | None = None,
) -> dict[str, Any]:
    dimensions = max(1, len(shape))
    spec: dict[str, Any] = {
        "dtype": dtype,
        "shape": shape,
        "layout": "identity",
        "axes": axes,
        "invalid_axes": [dimensions, -dimensions - 1],
        "output_dtypes": list(output_dtypes),
    }
    if values is not None:
        spec["values"] = values
    elif words is not None:
        spec["words"] = words
    else:
        raise ValueError(f"{dtype} scan dtype source needs values or words")
    return spec


def _complex_scan_dtype_observation(tensor: Any) -> dict[str, Any]:
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if dtype_name in ("complex64", "complex128"):
        return _observe_complex(tensor)
    if dtype_name in ("float32", "float64"):
        observation = _observe(tensor)
        values = tensor.detach().reshape(-1).tolist()
        observation["negative_zero"] = [
            index for index, value in enumerate(values)
            if value == 0.0 and math.copysign(1.0, value) < 0.0
        ]
        return observation
    return _observe_integer_output(tensor)


def _complex_scan_dtype_call(
    torch: Any, source: Any, operation: str, axis: int, dtype: Any, reverse: bool
) -> Any:
    scan_source = torch.flip(source, dims=[axis]) if reverse and source.ndim else source
    output = getattr(scan_source, operation)(dim=axis, dtype=dtype)
    return torch.flip(output, dims=[axis]) if reverse and source.ndim else output


def _complex_scan_dtype_case(torch: Any, spec: dict[str, Any], label: str) -> dict[str, Any]:
    source = (
        _complex_scan_source(torch, spec)
        if spec["dtype"] in ("complex64", "complex128")
        else _real_dtype_source(torch, spec)
    )
    expected: dict[str, Any] = {}
    rejects_nonempty_scan = source.numel() > 0 and source.ndim > 0
    for output_name in spec["output_dtypes"]:
        output_dtype = getattr(torch, output_name)
        reject = output_name in ("bool", "uint16", "uint32", "uint64") and rejects_nonempty_scan
        by_operation: dict[str, Any] = {}
        for operation in ("cumsum", "cumprod"):
            axes: dict[str, Any] = {}
            for axis in spec["axes"]:
                directions: dict[str, Any] = {}
                for reverse in (False, True):
                    direction = "reverse" if reverse else "forward"
                    directions[direction] = _unsupported_integer_output(
                        lambda axis=axis, reverse=reverse, operation=operation:
                            _complex_scan_dtype_call(
                                torch, source, operation, axis, output_dtype, reverse
                            ),
                        operation,
                        output_name,
                        reject,
                        f"{label}.{output_name}.{operation}.{axis}.{direction}",
                        observer=_complex_scan_dtype_observation,
                    )
                axes[str(axis)] = directions
            invalid_axes = {
                str(axis): _complex_invalid_scan_axis(
                    lambda axis=axis, operation=operation: _complex_scan_dtype_call(
                        torch, source, operation, axis, output_dtype, False
                    ),
                    f"{label}.{output_name}.{operation}.invalid_axis.{axis}",
                )
                for axis in spec["invalid_axes"]
            }
            by_operation[operation] = {"axes": axes, "invalid_axes": invalid_axes}
        expected[output_name] = by_operation
    return expected


def _complex_scan_dtype_inputs() -> dict[str, dict[str, Any]]:
    """Return fresh data-only specs for the complex scan dtype matrix."""
    all_outputs = (*_INTEGER_OUTPUT_DTYPE_NAMES, "float32", "float64", "complex64", "complex128")
    complex_outputs = ("complex64", "complex128")
    axes_2d = [0, 1, -1]
    specs: dict[str, dict[str, Any]] = {}
    integer_words = [-2, -1, 0, 1, 2, 3]
    real_values = [1.9, 2.9, 0.0, 3.1, 4.9, 1.1]
    for dtype_name in _INTEGER_DTYPE_NAMES:
        specs[dtype_name] = _real_scan_dtype_spec(
            dtype_name, [2, 3], axes_2d, complex_outputs, words=integer_words
        )
    for dtype_name in ("float32", "float64"):
        specs[dtype_name] = _real_scan_dtype_spec(
            dtype_name, [2, 3], axes_2d, complex_outputs, values=real_values
        )

    base_complex_values = [
        [1.9, 0.25], [2.9, -0.5], [0.0, 1.25],
        [3.1, -1.0], [4.9, 2.0], [1.1, -2.5],
    ]
    for dtype_name in ("complex64", "complex128"):
        specs[dtype_name] = _complex_scan_dtype_spec(
            dtype_name, [2, 3], base_complex_values, axes_2d, all_outputs
        )

    specs["transpose"] = _complex_scan_dtype_spec(
        "complex64", [2, 3],
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [1.0, 0.0]],
        axes_2d, all_outputs, "transpose",
    )
    specs["offset"] = _complex_scan_dtype_spec(
        "complex128", [3, 4],
        [[1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [2.0, 0.0]] * 3,
        axes_2d, all_outputs, "offset",
    )
    specs["reverse"] = _complex_scan_dtype_spec(
        "complex64", [5],
        [[1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
        [0, -1], all_outputs, "reverse",
    )
    specs["broadcast"] = _complex_scan_dtype_spec(
        "complex128", [1, 3], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        axes_2d, all_outputs, "broadcast",
    )

    specs["scalar"] = _complex_scan_dtype_spec(
        "complex128", [], [[1.9, 2.0]], [0, -1], all_outputs,
    )
    specs["scalar_imaginary"] = _complex_scan_dtype_spec(
        "complex128", [], [[0.0, 4.0]], [0, -1], all_outputs,
    )
    specs["scalar_negative_zero"] = _complex_scan_dtype_spec(
        "complex64", [], [[-0.0, -0.0]], [0, -1], all_outputs,
    )
    specs["empty"] = _complex_scan_dtype_spec(
        "complex128", [2, 0, 3], [], [0, 1, 2, -1], all_outputs,
    )

    specs["precision"] = _complex_scan_dtype_spec(
        "complex128", [2], [[16777217.0, 16777217.0], [-16777216.0, -16777216.0]],
        [0, -1], ("complex64", "complex128", "float32", "float64", "int32", "int64"),
    )
    specs["overflow_recovery"] = _complex_scan_dtype_spec(
        "complex128", [3], [[1e20, 0.0], [1e20, 0.0], [1e-20, 0.0]],
        [0, -1], ("complex64", "complex128", "float32", "float64"),
    )
    specs["underflow_recovery"] = _complex_scan_dtype_spec(
        "complex128", [3], [[1e-30, 0.0], [1e-30, 0.0], [1e30, 0.0]],
        [0, -1], ("complex64", "complex128", "float32", "float64"),
    )
    specs["nonfinite"] = _complex_scan_dtype_spec(
        "complex128", [3], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [0, -1], ("complex64", "complex128", "float32", "float64"),
        nonfinite=[
            {"index": 0, "component": "real", "value": "+inf"},
            {"index": 1, "component": "imag", "value": "+inf"},
            {"index": 2, "component": "real", "value": "nan"},
        ],
    )
    specs["complex_negative"] = _complex_scan_dtype_spec(
        "complex128", [3], [[-127.9, 0.5], [-128.9, -1.25], [-0.9, 2.0]],
        [0, -1],
        ("complex64", "complex128", "float32", "float64", "int8", "int16", "int32", "int64"),
    )
    specs["uint64_precision"] = _real_scan_dtype_spec(
        "uint64", [3], [0, -1], complex_outputs,
        words=[-(1 << 63) + 0x8000000001, 0, 1],
    )
    return specs


def _complex_reduction_dtype_inputs() -> dict[str, dict[str, Any]]:
    """Return fresh shared data-only specs for complex reduction dtype cases."""
    all_outputs = (*_INTEGER_OUTPUT_DTYPE_NAMES, "float32", "float64", "complex64", "complex128")
    specs = _complex_scan_dtype_inputs()
    specs["rounding"] = _complex_scan_dtype_spec(
        "complex64", [3], [[1.9, 2.0], [2.9, -3.0], [0.0, 4.0]], [0, -1],
        all_outputs,
    )
    specs["complex64_cancellation"] = _complex_scan_dtype_spec(
        "complex64", [3], [[16777216.0, 0.0], [1.0, 0.0], [-16777216.0, 0.0]],
        [0, -1], ("complex64", "complex128", "float32", "float64", "int32", "int64"),
    )
    specs["double_precision"] = _complex_scan_dtype_spec(
        "complex128", [3],
        [
            [9007199254740992, 9007199254740992],
            [1, 1],
            [-9007199254740992, -9007199254740992],
        ],
        [0, -1], ("complex64", "complex128", "float32", "float64", "int64"),
    )
    packed_values: list[list[int | float]] = [
        [value, value]
        for value in (0, 0, 16777216, 1, -16777216)
        for _ in range(17)
    ]
    specs["packed_columns_cancellation"] = _complex_scan_dtype_spec(
        "complex64", [5, 17], packed_values, [0, 1, -1],
        ("complex64", "complex128", "float32", "float64", "int32", "int64"),
    )
    return specs


def complex_scan_dtype(torch: Any) -> dict[str, Any]:
    """Observe explicit CPU complex scan output dtypes across source and layout classes."""
    specs = _complex_scan_dtype_inputs()
    expected = {name: _complex_scan_dtype_case(torch, spec, name) for name, spec in specs.items()}
    return _case("tensor.complex_scan_dtype", "api:torch", specs, expected, tolerance=1e-6)


def _complex_invalid_reduction_axis(call: Callable[[], Any], label: str) -> dict[str, str]:
    try:
        call()
    except (IndexError, RuntimeError) as error:
        if "dimension out of range" not in str(error).lower():
            raise
        return {"error": "invalid_axis"}
    raise ValueError(f"{label} unexpectedly accepted an invalid reduction axis")


def complex_mean_dtype(torch: Any) -> dict[str, Any]:
    """Observe explicit and default CPU complex mean output dtypes."""
    specs = _complex_reduction_dtype_inputs()
    all_outputs = (*_INTEGER_OUTPUT_DTYPE_NAMES, "float32", "float64", "complex64", "complex128")
    for spec in specs.values():
        spec["output_dtypes"] = list(
            (*all_outputs, "default")
            if spec["dtype"] in ("complex64", "complex128")
            else ("complex64", "complex128")
        )

    expected: dict[str, Any] = {}
    for name, spec in specs.items():
        source = (
            _complex_scan_source(torch, spec)
            if spec["dtype"] in ("complex64", "complex128")
            else _real_dtype_source(torch, spec)
        )
        source_expected: dict[str, Any] = {}
        for target in spec["output_dtypes"]:
            output_dtype = None if target == "default" else getattr(torch, target)

            def mean_call(axis: int | None) -> Any:
                if target == "default":
                    return source.mean() if axis is None else source.mean(dim=axis, keepdim=True)
                return (
                    source.mean(dtype=output_dtype)
                    if axis is None
                    else source.mean(dim=axis, keepdim=True, dtype=output_dtype)
                )

            is_integer_target = target in _INTEGER_DTYPE_NAMES
            all_call = lambda: mean_call(None)
            target_expected: dict[str, Any] = {
                "all": (
                    _integer_mean_rejection(all_call, f"{name}.mean.all.{target}")
                    if is_integer_target
                    else _complex_scan_dtype_observation(all_call())
                )
            }
            axes: dict[str, Any] = {}
            for axis in spec["axes"]:
                axis_call = lambda axis=axis: mean_call(axis)
                axes[str(axis)] = (
                    _integer_mean_rejection(axis_call, f"{name}.mean.{axis}.{target}")
                    if is_integer_target
                    else _complex_scan_dtype_observation(axis_call())
                )
            invalid_axes: dict[str, Any] = {}
            for axis in spec["invalid_axes"]:
                axis_call = lambda axis=axis: mean_call(axis)
                invalid_axes[str(axis)] = (
                    _integer_mean_rejection(
                        axis_call, f"{name}.mean.invalid_axis.{axis}.{target}"
                    )
                    if is_integer_target
                    else _complex_invalid_reduction_axis(
                        axis_call, f"{name}.mean.invalid_axis.{axis}.{target}"
                    )
                )
            target_expected["axes"] = axes
            target_expected["invalid_axes"] = invalid_axes
            source_expected[target] = target_expected
        expected[name] = source_expected
    return _case("tensor.complex_mean_dtype", "api:torch", specs, expected, tolerance=1e-6)


def complex_sum_dtype(torch: Any) -> dict[str, Any]:
    """Observe explicit and default CPU complex sum output dtypes."""
    specs = _complex_reduction_dtype_inputs()
    for spec in specs.values():
        if spec["dtype"] in ("complex64", "complex128"):
            spec["output_dtypes"].append("default")

    expected: dict[str, Any] = {}
    for name, spec in specs.items():
        source = (
            _complex_scan_source(torch, spec)
            if spec["dtype"] in ("complex64", "complex128")
            else _real_dtype_source(torch, spec)
        )
        source_expected: dict[str, Any] = {}
        for target in spec["output_dtypes"]:
            output_dtype = None if target == "default" else getattr(torch, target)

            def sum_call(axis: int | None) -> Any:
                if target == "default":
                    return (
                        source.sum()
                        if axis is None
                        else source.sum(dim=axis, keepdim=True)
                    )
                return (
                    source.sum(dtype=output_dtype)
                    if axis is None
                    else source.sum(dim=axis, keepdim=True, dtype=output_dtype)
                )

            reject = target in ("uint16", "uint32", "uint64") and source.numel() > 0
            target_expected: dict[str, Any] = {
                "all": _unsupported_integer_output(
                    lambda: sum_call(None), "sum", target, reject,
                    f"{name}.sum.all.{target}", observer=_complex_scan_dtype_observation,
                )
            }
            axes: dict[str, Any] = {}
            for axis in spec["axes"]:
                axes[str(axis)] = _unsupported_integer_output(
                    lambda axis=axis: sum_call(axis), "sum", target, reject,
                    f"{name}.sum.{axis}.{target}",
                    observer=_complex_scan_dtype_observation,
                )
            target_expected["axes"] = axes

            invalid_axes: dict[str, Any] = {}
            for axis in spec["invalid_axes"]:
                invalid_axes[str(axis)] = _complex_invalid_reduction_axis(
                    lambda axis=axis: sum_call(axis),
                    f"{name}.sum.invalid_axis.{axis}.{target}",
                )
            target_expected["invalid_axes"] = invalid_axes
            source_expected[target] = target_expected
        expected[name] = source_expected
    return _case("tensor.complex_sum_dtype", "api:torch", specs, expected, tolerance=1e-6)


def complex_prod_dtype(torch: Any) -> dict[str, Any]:
    """Observe explicit and default CPU complex product output dtypes."""
    specs = _complex_reduction_dtype_inputs()
    for spec in specs.values():
        if spec["dtype"] in ("complex64", "complex128"):
            spec["output_dtypes"].append("default")

    expected: dict[str, Any] = {}
    for name, spec in specs.items():
        source = (
            _complex_scan_source(torch, spec)
            if spec["dtype"] in ("complex64", "complex128")
            else _real_dtype_source(torch, spec)
        )
        source_expected: dict[str, Any] = {}
        for target in spec["output_dtypes"]:
            output_dtype = None if target == "default" else getattr(torch, target)

            def prod_call(axis: int | None) -> Any:
                if target == "default":
                    return (
                        source.prod()
                        if axis is None
                        else source.prod(dim=axis, keepdim=True)
                    )
                return (
                    source.prod(dtype=output_dtype)
                    if axis is None
                    else source.prod(dim=axis, keepdim=True, dtype=output_dtype)
                )

            reject = target in ("uint16", "uint32", "uint64") and source.numel() > 0
            target_expected: dict[str, Any] = {
                "all": _unsupported_integer_output(
                    lambda: prod_call(None), "prod", target, reject,
                    f"{name}.prod.all.{target}",
                    observer=_complex_scan_dtype_observation,
                )
            }
            axes: dict[str, Any] = {}
            for axis in spec["axes"]:
                axes[str(axis)] = _unsupported_integer_output(
                    lambda axis=axis: prod_call(axis), "prod", target, reject,
                    f"{name}.prod.{axis}.{target}",
                    observer=_complex_scan_dtype_observation,
                )
            target_expected["axes"] = axes

            invalid_axes: dict[str, Any] = {}
            for axis in spec["invalid_axes"]:
                invalid_axes[str(axis)] = _complex_invalid_reduction_axis(
                    lambda axis=axis: prod_call(axis),
                    f"{name}.prod.invalid_axis.{axis}.{target}",
                )
            target_expected["invalid_axes"] = invalid_axes
            source_expected[target] = target_expected
        expected[name] = source_expected
    return _case("tensor.complex_prod_dtype", "api:torch", specs, expected, tolerance=1e-6)


def _extrema_requests_2d(*, include_validation_errors: bool) -> list[dict[str, Any]]:
    variants: tuple[tuple[str, int | list[int] | None, bool], ...] = (
        ("all_drop", None, False),
        ("all_keep", None, True),
        ("axes_empty_drop", [], False),
        ("axes_empty_keep", [], True),
        ("axis_0_drop", 0, False),
        ("axis_0_keep", 0, True),
        ("axis_neg1_drop", -1, False),
        ("axes_0_1_drop", [0, 1], False),
        ("axes_1_0_keep", [1, 0], True),
    )
    requests = [
        {
            "id": f"{operation}_{name}",
            "operation": operation,
            "dimensions": dimensions,
            "keepdim": keepdim,
        }
        for operation in ("amin", "amax")
        for name, dimensions, keepdim in variants
    ]
    if include_validation_errors:
        requests.extend(
            [
                {
                    "id": "amin_bad_axis_drop",
                    "operation": "amin",
                    "dimensions": 2,
                    "keepdim": False,
                },
                {
                    "id": "amax_duplicate_axis_drop",
                    "operation": "amax",
                    "dimensions": [0, -2],
                    "keepdim": False,
                },
            ]
        )
    return requests


def _extrema_requests_1d() -> list[dict[str, Any]]:
    return [
        {"id": "amin_all_drop", "operation": "amin", "dimensions": None, "keepdim": False},
        {"id": "amax_all_drop", "operation": "amax", "dimensions": None, "keepdim": False},
        {"id": "amin_all_keep", "operation": "amin", "dimensions": None, "keepdim": True},
        {"id": "amax_all_keep", "operation": "amax", "dimensions": None, "keepdim": True},
        {"id": "amin_axes_empty_drop", "operation": "amin", "dimensions": [], "keepdim": False},
        {"id": "amax_axes_empty_keep", "operation": "amax", "dimensions": [], "keepdim": True},
        {"id": "amin_axis_0_drop", "operation": "amin", "dimensions": 0, "keepdim": False},
        {"id": "amax_axis_neg1_keep", "operation": "amax", "dimensions": -1, "keepdim": True},
        {"id": "amin_bad_axis_drop", "operation": "amin", "dimensions": 1, "keepdim": False},
        {"id": "amax_duplicate_axis_drop", "operation": "amax", "dimensions": [0, -1], "keepdim": False},
    ]


def _extrema_requests_scalar() -> list[dict[str, Any]]:
    return [
        {"id": "amin_all_drop", "operation": "amin", "dimensions": None, "keepdim": False},
        {"id": "amax_all_keep", "operation": "amax", "dimensions": None, "keepdim": True},
        {"id": "amin_axes_empty_drop", "operation": "amin", "dimensions": [], "keepdim": False},
        {"id": "amax_axes_empty_keep", "operation": "amax", "dimensions": [], "keepdim": True},
        {"id": "amin_axis_0_drop", "operation": "amin", "dimensions": 0, "keepdim": False},
        {"id": "amax_axis_neg1_keep", "operation": "amax", "dimensions": -1, "keepdim": True},
        {"id": "amin_bad_axis_drop", "operation": "amin", "dimensions": 1, "keepdim": False},
        {"id": "amax_duplicate_axis_drop", "operation": "amax", "dimensions": [0, -1], "keepdim": False},
    ]


def _extrema_requests_empty() -> list[dict[str, Any]]:
    return [
        {"id": "amin_all_drop", "operation": "amin", "dimensions": None, "keepdim": False},
        {"id": "amax_axes_empty_keep", "operation": "amax", "dimensions": [], "keepdim": True},
        {"id": "amin_zero_axis_drop", "operation": "amin", "dimensions": 1, "keepdim": False},
        {"id": "amax_axis_0_drop", "operation": "amax", "dimensions": 0, "keepdim": False},
        {"id": "amin_axes_0_2_drop", "operation": "amin", "dimensions": [0, 2], "keepdim": False},
        {"id": "amax_axes_2_0_keep", "operation": "amax", "dimensions": [2, 0], "keepdim": True},
        {"id": "amin_duplicate_axis_drop", "operation": "amin", "dimensions": [0, -3], "keepdim": False},
        {"id": "amax_zero_axis_duplicate_drop", "operation": "amax", "dimensions": [1, 1], "keepdim": False},
        {"id": "amin_bad_axis_drop", "operation": "amin", "dimensions": 3, "keepdim": False},
    ]


def _extrema_requests_empty_outer() -> list[dict[str, Any]]:
    return [
        {"id": "amin_all_drop", "operation": "amin", "dimensions": None, "keepdim": False},
        {"id": "amax_axes_empty_drop", "operation": "amax", "dimensions": [], "keepdim": False},
        {"id": "amax_axis_1_drop", "operation": "amax", "dimensions": 1, "keepdim": False},
        {"id": "amax_axis_1_keep", "operation": "amax", "dimensions": 1, "keepdim": True},
        {"id": "amin_bad_axis_drop", "operation": "amin", "dimensions": 2, "keepdim": False},
        {"id": "amax_duplicate_zero_axis_drop", "operation": "amax", "dimensions": [0, 0], "keepdim": False},
    ]


def _extrema_requests_multi_axis() -> list[dict[str, Any]]:
    return [
        {"id": "amin_axes_0_2_drop", "operation": "amin", "dimensions": [0, 2], "keepdim": False},
        {"id": "amax_axes_2_0_keep", "operation": "amax", "dimensions": [2, 0], "keepdim": True},
        {"id": "amin_axes_0_neg1_drop", "operation": "amin", "dimensions": [0, -1], "keepdim": False},
        {"id": "amax_axes_neg1_0_keep", "operation": "amax", "dimensions": [-1, 0], "keepdim": True},
        {"id": "amin_axes_empty_drop", "operation": "amin", "dimensions": [], "keepdim": False},
        {"id": "amax_axes_empty_keep", "operation": "amax", "dimensions": [], "keepdim": True},
        {"id": "amin_bad_axis_drop", "operation": "amin", "dimensions": 3, "keepdim": False},
        {"id": "amax_duplicate_axis_drop", "operation": "amax", "dimensions": [0, -3], "keepdim": False},
    ]


def _extrema_source(torch: Any, spec: dict[str, Any]) -> Any:
    dtype_name = spec["dtype"]
    dtype = getattr(torch, dtype_name)
    if "words" in spec:
        source = torch.tensor(spec["words"], dtype=torch.int64, device="cpu").to(dtype=dtype)
    elif dtype_name.startswith("complex"):
        values = [complex(real, imaginary) for real, imaginary in spec["values"]]
        source = torch.tensor(values, dtype=dtype, device="cpu")
    else:
        source = torch.tensor(spec["values"], dtype=dtype, device="cpu")
    source = source.reshape(spec["shape"])
    for special in spec["specials"]:
        value = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}[
            special["value"]
        ]
        source.reshape(-1)[special["index"]] = value
    if spec["layout"] == "permute_210":
        return source.permute(2, 1, 0)
    return _real_dtype_layout(source, spec["layout"])


def _extrema_error_code(error: Exception) -> str | None:
    message = str(error).lower().replace("non-zero", "nonzero")
    if "dimension out of range" in message:
        return "axis_out_of_range"
    if "have nonzero size" in message:
        return "empty_reduced_axis"
    if "expected reduction dim to be specified" in message:
        return "empty_reduction"
    if "appears multiple times" in message:
        return "duplicate_axes"
    if "not implemented" in message:
        return "unsupported_dtype"
    return None


def _observe_extrema_request(torch: Any, source: Any, request: dict[str, Any]) -> dict[str, Any]:
    operation = getattr(torch, request["operation"])
    dimensions = request["dimensions"]
    try:
        if dimensions is None:
            output = operation(source, keepdim=request["keepdim"])
        else:
            if isinstance(dimensions, list):
                dimensions = tuple(dimensions)
            output = operation(source, dim=dimensions, keepdim=request["keepdim"])
    except (IndexError, NotImplementedError, RuntimeError) as error:
        code = _extrema_error_code(error)
        if code is None:
            raise ValueError(f"unexpected {request['operation']} rejection: {error}") from error
        return {"error": code}
    if not output.is_floating_point() and not output.is_complex():
        return _observe_integer_output(output)
    return _observe(output)


def extrema(torch: Any) -> dict[str, Any]:
    """Observe PyTorch CPU amin/amax values, shapes, layouts and rejection precedence."""
    inputs: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        dtype: str,
        shape: list[int],
        payload_key: str,
        payload: list[Any],
        requests: list[dict[str, Any]],
        *,
        layout: str = "identity",
        specials: list[dict[str, Any]] | None = None,
    ) -> None:
        spec: dict[str, Any] = {
            "dtype": dtype,
            "shape": shape,
            "layout": layout,
            payload_key: payload,
            "specials": [] if specials is None else specials,
            "requests": requests,
        }
        inputs[name] = spec

    requests_2d = _extrema_requests_2d(include_validation_errors=True)
    requests_1d = _extrema_requests_1d()
    requests_scalar = _extrema_requests_scalar()
    for name, dtype, words in (
        ("bool", "bool", [0, 1, 1, 0, 0, 1]),
        ("int8", "int8", [-128, 127, -1, 1, 2, 3]),
        ("int16", "int16", [-32768, 32767, -1, 1, 2, 3]),
        ("int32", "int32", [-2147483648, 2147483647, -1, 1, 2, 3]),
        ("int64", "int64", [-(1 << 63), 8, -2, (1 << 63) - 1, 4, -9]),
        ("uint8", "uint8", [0, 255, 1, 2, 128, 3]),
        ("native_int", "int64", [-(1 << 63), 8, -2, (1 << 63) - 1, 4, -9]),
    ):
        add(name, dtype, [2, 3], "words", words, requests_2d)

    for name, dtype, values in (
        ("float32", "float32", [-4.5, 7.25, 1.5, 0.25, -9.0, 3.0]),
        ("float64", "float64", [-4.5, 7.25, 1.5, 0.25, -9.0, 3.0]),
        ("native_float", "float64", [-4.5, 7.25, 1.5, 0.25, -9.0, 3.0]),
    ):
        add(name, dtype, [2, 3], "values", values, requests_2d)

    for name, dtype in (
        ("reject_uint16_scalar", "uint16"),
        ("reject_uint32_scalar", "uint32"),
        ("reject_uint64_scalar", "uint64"),
    ):
        add(name, dtype, [], "words", [1], requests_scalar)
    for name, dtype in (
        ("reject_complex64_scalar", "complex64"),
        ("reject_complex128_scalar", "complex128"),
    ):
        add(name, dtype, [], "values", [[1.0, -2.0]], requests_scalar)
    add("reject_uint16_matrix", "uint16", [2, 3], "words", [1, 2, 3, 4, 5, 6], requests_2d)

    add("transpose", "float64", [2, 3], "values", [2.0, -3.0, 5.0, 7.0, 0.0, -1.0],
        requests_2d, layout="transpose")
    add("offset", "int32", [3, 4], "words", list(range(12)), requests_2d, layout="offset")
    add("reverse", "float32", [5], "values", [1.0, -5.0, 2.0, 0.0, 4.0],
        requests_1d, layout="reverse")
    add("broadcast", "uint8", [1, 3], "words", [2, 3, 0], requests_2d, layout="broadcast")

    add("scalar", "int64", [], "words", [-(1 << 63)], requests_scalar)
    add(
        "multi_axes", "float32", [2, 3, 2], "values",
        [-5.0, 9.0, 8.0, 4.0, 3.0, 2.0, 7.0, 6.0, 1.0, 0.0, 11.0, -2.0],
        _extrema_requests_multi_axis(),
    )
    coalesced_requests = [
        {"id": "amin_axes_0_1_drop", "operation": "amin", "dimensions": [0, 1], "keepdim": False},
        {"id": "amax_axes_0_1_drop", "operation": "amax", "dimensions": [0, 1], "keepdim": False},
    ]
    coalesced_values = [0.0] * 96
    coalesced_values[4] = -0.0
    for name, dtype in (
        ("coalesced_axes_f32", "float32"),
        ("coalesced_axes_f64", "float64"),
    ):
        add(name, dtype, [4, 8, 3], "values", coalesced_values, coalesced_requests,
            layout="permute_210")

    add("empty", "int8", [2, 0, 3], "words", [], _extrema_requests_empty())
    for name, dtype, payload_key in (
        ("empty_uint16", "uint16", "words"),
        ("empty_uint32", "uint32", "words"),
        ("empty_uint64", "uint64", "words"),
        ("empty_complex64", "complex64", "values"),
        ("empty_complex128", "complex128", "values"),
    ):
        add(name, dtype, [0, 3], payload_key, [], _extrema_requests_empty_outer())

    add("nan", "float32", [3], "values", [1.0, 2.0, 3.0], requests_1d,
        specials=[{"index": 1, "value": "nan"}])
    add("infinities", "float64", [3], "values", [1.0, 2.0, 3.0], requests_1d,
        specials=[{"index": 0, "value": "+inf"}, {"index": 2, "value": "-inf"}])
    tie_requests = [
        {"id": "amin_all_drop", "operation": "amin", "dimensions": None, "keepdim": False},
        {"id": "amax_all_drop", "operation": "amax", "dimensions": None, "keepdim": False},
    ]
    add("tie_zeros", "float64", [2], "values", [0.0, -0.0], tie_requests)
    add("tie_zeros_reversed", "float64", [2], "values", [0.0, -0.0], tie_requests,
        layout="reverse")


    for name, dtype, size, index in (
        ("vector_zero_lane_f32", "float32", 16, 0),
        ("vector_zero_tail_f32", "float32", 17, 16),
        ("vector_zero_lane_f64", "float64", 8, 0),
        ("vector_zero_tail_f64", "float64", 9, 8),
    ):
        values = [0.0] * size
        values[index] = -0.0
        add(name, dtype, [size], "values", values, requests_1d)

    column_requests = [
        {"id": f"{operation}_axis_0_drop", "operation": operation, "dimensions": 0, "keepdim": False}
        for operation in ("amin", "amax")
    ] + [
        {"id": f"{operation}_axis_0_keep", "operation": operation, "dimensions": 0, "keepdim": True}
        for operation in ("amin", "amax")
    ]
    for name, dtype, columns in (
        ("packed_columns_f32", "float32", 17),
        ("packed_columns_f64", "float64", 9),
    ):
        add(name, dtype, [2, columns], "values", [0.0] * columns + [-0.0] * columns,
            column_requests)

    expected: dict[str, Any] = {}
    for name, spec in inputs.items():
        source = _extrema_source(torch, spec)
        expected[name] = {
            request["id"]: _observe_extrema_request(torch, source, request)
            for request in spec["requests"]
        }
    return _case("tensor.extrema", "api:torch", inputs, expected, tolerance=0.0)


_INDEXED_EXTREMA_DTYPES = (
    "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool",
    "uint16", "uint32", "uint64", "complex64", "complex128",
)
_INDEXED_EXTREMA_INTEGER_DTYPES = (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
)


def _indexed_extrema_raw_cases(torch: Any, dtype_name: str) -> dict[str, dict[str, Any]]:
    def case(
        shape: list[int],
        values: list[int | float | str],
        axes: list[int | None],
        layout: str = "identity",
    ) -> dict[str, Any]:
        return {"shape": shape, "layout": layout, "values": values, "axes": axes}

    cases = {
        "ties": case([2, 3], [2, 1, 3, 1, 3, 3], [None, 0, 1, -1, 2, -3]),
        "scalar": case([], [3], [None, 0, -1, 1, -2]),
        "singleton": case([1], [3], [None, 0, -1, 1]),
        "singleton_axis": case([2, 1, 3], [2, 1, 3, 1, 3, 3], [None, 1, -2]),
        "empty_outer": case([0, 3], [], [None, 0, 1, 2]),
        "empty_reduced": case([2, 0], [], [None, 0, 1, -1]),
        "empty_singleton": case([0, 1], [], [None, 0, 1]),
        "empty_scalar_axis": case([0], [], [None, 0, -1]),
        "transpose": case([2, 3], [2, 1, 3, 1, 3, 3], [None, 0, 1], "transpose"),
        "offset": case(
            [3, 4], [index % 3 for index in range(12)], [None, 0, 1], "offset"
        ),
        "reverse": case([5], [2, 1, 3, 1, 3], [None, 0, -1], "reverse"),
        "broadcast": case([1, 3], [1, 3, 2], [None, 0, 1], "broadcast"),
        "permuted": case(
            [4, 8, 3], [index % 7 for index in range(96)], [None, 0, 1, 2], "permute"
        ),
    }
    if dtype_name in ("float32", "float64"):
        cases.update(
            {
                "nan_first": case(
                    [2, 3], [2, "nan", "+inf", "nan", "-inf", "nan"], [None, 0, 1]
                ),
                "zero_ties": case(
                    [2, 3], ["-0", 0, "-0", 0, "-0", 0], [None, 0, 1]
                ),
                "infinities": case(
                    [2, 3], ["-inf", "-inf", "+inf", "+inf", "+inf", "-inf"], [None, 0, 1]
                ),
            }
        )
    elif dtype_name not in ("bool", "complex64", "complex128"):
        bounds = torch.iinfo(getattr(torch, dtype_name))
        cases["integer_bounds"] = case(
            [2, 2],
            [str(int(bounds.max)), str(int(bounds.max) - 1), str(int(bounds.min)), str(int(bounds.max))],
            [None, 0, 1],
        )
    return cases


def _indexed_extrema_word(value: int | str) -> int:
    word = int(value)
    if word >= 1 << 63:
        word -= 1 << 64
    return word


def _indexed_extrema_input(dtype_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    result = {key: spec[key] for key in ("shape", "layout", "axes")}
    if dtype_name in _INDEXED_EXTREMA_INTEGER_DTYPES:
        result["words"] = [_indexed_extrema_word(value) for value in spec["values"]]
    elif dtype_name in ("complex64", "complex128"):
        result["values"] = [[value, 0] for value in spec["values"]]
    else:
        result["values"] = spec["values"]
    return result


def _indexed_extrema_number(value: int | float | str) -> int | float:
    if isinstance(value, str):
        return {"nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf"), "-0": -0.0}[
            value
        ]
    return value


def _indexed_extrema_source(torch: Any, dtype_name: str, spec: dict[str, Any]) -> Any:
    dtype = getattr(torch, dtype_name)
    if "words" in spec:
        words = spec["words"]
        if dtype_name == "uint64":
            values = [word + (1 << 64) if word < 0 else word for word in words]
        else:
            values = words
        source = torch.tensor(values, dtype=dtype, device="cpu")
    elif dtype_name in ("complex64", "complex128"):
        source = torch.tensor(
            [complex(real, imaginary) for real, imaginary in spec["values"]],
            dtype=dtype,
            device="cpu",
        )
    else:
        source = torch.tensor(
            [_indexed_extrema_number(value) for value in spec["values"]],
            dtype=dtype,
            device="cpu",
        )
    source = source.reshape(spec["shape"])
    layout = spec["layout"]
    if layout == "transpose":
        return source.transpose(0, 1)
    if layout == "offset":
        return source.transpose(0, 1)[:, 1:]
    if layout == "reverse":
        return source.flip([0])
    if layout == "broadcast":
        return source.expand(4, 3)
    if layout == "permute":
        return source.permute(2, 1, 0)
    return source


def _indexed_extrema_request_id(operation: str, axis: int | None, keepdim: bool) -> str:
    axis_name = "all" if axis is None else f"axis_{axis}"
    return f"{operation}_{axis_name}_{'keep' if keepdim else 'drop'}"


def _indexed_extrema_error_code(error: Exception) -> str | None:
    message = str(error).lower().replace("non-zero", "nonzero")
    if isinstance(error, (NotImplementedError, TypeError)):
        return "unsupported_dtype"
    if "dimension out of range" in message or "axis out of range" in message:
        return "axis_out_of_range"
    if "nonzero size" in message or "empty reduced axis" in message:
        return "empty_reduced_axis"
    if "expected reduction dim to be specified" in message or "empty reduction" in message:
        return "empty_reduction"
    if "not implemented" in message or ("unsupported" in message and "dtype" in message):
        return "unsupported_dtype"
    return None


def _observe_indexed_extrema_request(
    torch: Any, source: Any, operation: str, axis: int | None, keepdim: bool, label: str
) -> dict[str, Any]:
    method = getattr(torch, operation)
    try:
        if axis is None:
            output = method(source, keepdim=keepdim)
        else:
            output = method(source, dim=axis, keepdim=keepdim)
    except (IndexError, NotImplementedError, RuntimeError, TypeError) as error:
        code = _indexed_extrema_error_code(error)
        if code is None:
            raise ValueError(f"unexpected indexed {operation} rejection for {label}: {error}") from error
        return {"error": code}
    if str(output.dtype).removeprefix("torch.") != "int64":
        raise ValueError(f"indexed {operation} returned non-int64 indices for {label}")
    return _observe_integer_output(output)


def indexed_extrema(torch: Any) -> dict[str, Any]:
    """Observe CPU argmin/argmax dtype, shape, layout and rejection contracts."""
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype_name in _INDEXED_EXTREMA_DTYPES:
        inputs[dtype_name] = {}
        expected[dtype_name] = {}
        for case_name, raw_spec in _indexed_extrema_raw_cases(torch, dtype_name).items():
            spec = _indexed_extrema_input(dtype_name, raw_spec)
            inputs[dtype_name][case_name] = spec
            source = _indexed_extrema_source(torch, dtype_name, spec)
            by_request: dict[str, Any] = {}
            for operation in ("argmin", "argmax"):
                for axis in spec["axes"]:
                    for keepdim in (False, True):
                        request_id = _indexed_extrema_request_id(operation, axis, keepdim)
                        by_request[request_id] = _observe_indexed_extrema_request(
                            torch,
                            source,
                            operation,
                            axis,
                            keepdim,
                            f"{dtype_name}.{case_name}.{request_id}",
                        )
            expected[dtype_name][case_name] = by_request
    return _case("tensor.indexed_extrema", "api:torch", inputs, expected, tolerance=0.0)

def _observe_paired_extrema_request(
    torch: Any,
    source: Any,
    dtype_name: str,
    operation: str,
    axis: int,
    keepdim: bool,
    label: str,
) -> dict[str, Any]:
    try:
        output = getattr(torch, operation)(source, dim=axis, keepdim=keepdim)
    except (IndexError, NotImplementedError, RuntimeError, TypeError) as error:
        code = _indexed_extrema_error_code(error)
        if code not in {"axis_out_of_range", "empty_reduced_axis", "unsupported_dtype"}:
            raise ValueError(f"unexpected paired {operation} rejection for {label}: {error}") from error
        return {"error": code}

    if dtype_name in ("complex64", "complex128"):
        raise ValueError(f"paired {operation} unexpectedly accepted complex input for {label}")
    if str(output.values.dtype).removeprefix("torch.") != dtype_name:
        raise ValueError(f"paired {operation} changed value dtype for {label}")
    if str(output.indices.dtype).removeprefix("torch.") != "int64":
        raise ValueError(f"paired {operation} returned non-int64 indices for {label}")
    observe_values = _observe_extrema_gradient if dtype_name in ("float32", "float64") else _observe_integer_output
    return {
        "values": observe_values(output.values),
        "indices": _observe_integer_output(output.indices),
    }


def paired_extrema(torch: Any) -> dict[str, Any]:
    """Observe CPU min/max values and indices from the paired PyTorch dimension APIs."""
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype_name in _INDEXED_EXTREMA_DTYPES:
        inputs[dtype_name] = {}
        expected[dtype_name] = {}
        for case_name, raw_spec in _indexed_extrema_raw_cases(torch, dtype_name).items():
            spec = _indexed_extrema_input(dtype_name, raw_spec)
            spec["axes"] = [axis for axis in spec["axes"] if axis is not None]
            inputs[dtype_name][case_name] = spec
            source = _indexed_extrema_source(torch, dtype_name, spec)
            by_request: dict[str, Any] = {}
            for operation in ("min", "max"):
                for axis in spec["axes"]:
                    for keepdim in (False, True):
                        request_id = _indexed_extrema_request_id(operation, axis, keepdim)
                        by_request[request_id] = _observe_paired_extrema_request(
                            torch,
                            source,
                            dtype_name,
                            operation,
                            axis,
                            keepdim,
                            f"{dtype_name}.{case_name}.{request_id}",
                        )
            expected[dtype_name][case_name] = by_request
    return _case("tensor.paired_extrema", "api:torch", inputs, expected, tolerance=0.0)


def _cumulative_extrema_inputs(torch: Any) -> dict[str, dict[str, dict[str, Any]]]:
    """Describe the complete CPU cummin/cummax matrix used by the pinned probe."""
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype_name in _INDEXED_EXTREMA_DTYPES:
        profiles = {
            name: _indexed_extrema_input(dtype_name, raw)
            for name, raw in _indexed_extrema_raw_cases(torch, dtype_name).items()
        }
        is_complex = dtype_name in ("complex64", "complex128")
        is_float = dtype_name in ("float32", "float64")
        if is_complex or is_float:
            width = 8 if dtype_name in ("float32", "complex64") else 16
            negative_zero = "0x80000000" if width == 8 else "0x8000000000000000"
            nan = "0x7fc12345" if width == 8 else "0x7ff8123456789abc"
            profiles["scalar_negative_zero"] = {
                "shape": [],
                "layout": "identity",
                "bits": [negative_zero] * (2 if is_complex else 1),
            }
            if is_complex:
                profiles["scalar_phase"] = {
                    "shape": [], "layout": "identity", "values": [[2.0, 7.0]]
                }
                negative_infinity = (
                    "0xff800000" if width == 8 else "0xfff0000000000000"
                )
                profiles["scalar_components"] = {
                    "shape": [], "layout": "identity", "bits": [nan, negative_infinity]
                }
            else:
                second_nan = (
                    "0x7fc54321" if width == 8 else "0x7ff876543210abcd"
                )
                one = "0x3f800000" if width == 8 else "0x3ff0000000000000"
                two = "0x40000000" if width == 8 else "0x4000000000000000"
                minus_one = "0xbf800000" if width == 8 else "0xbff0000000000000"
                profiles["nan_payloads"] = {
                    "shape": [5],
                    "layout": "identity",
                    "bits": [one, nan, two, second_nan, minus_one],
                }
                profiles["scalar_nan_payload"] = {
                    "shape": [], "layout": "identity", "bits": [nan]
                }
        elif dtype_name == "bool":
            profiles["bool_mixed"] = {
                "shape": [2, 3],
                "layout": "identity",
                "words": [1, 0, 1, 0, 1, 0],
            }
            profiles["scalar_false"] = {
                "shape": [], "layout": "identity", "words": [0]
            }
        else:
            bounds = torch.iinfo(getattr(torch, dtype_name))
            profiles["scalar_upper"] = {
                "shape": [],
                "layout": "identity",
                "words": [_indexed_extrema_word(int(bounds.max))],
            }

        for spec in profiles.values():
            rank = max(len(spec["shape"]), 1)
            spec["axes"] = list(range(-rank - 1, rank + 1))
        inputs[dtype_name] = profiles
    return inputs


def _cumulative_extrema_source(torch: Any, dtype_name: str, spec: dict[str, Any]) -> Any:
    if "bits" not in spec:
        return _indexed_extrema_source(torch, dtype_name, spec)
    width = 8 if dtype_name in ("float32", "complex64") else 16
    word_dtype = torch.uint32 if width == 8 else torch.uint64
    words = [int(word, 16) for word in spec["bits"]]
    return torch.tensor(words, dtype=word_dtype, device="cpu").view(
        getattr(torch, dtype_name)
    ).reshape(spec["shape"])


def _observe_cumulative_extrema_values(torch: Any, tensor: Any) -> dict[str, Any]:
    if not tensor.is_complex():
        return _observe_aminmax_tensor(torch, tensor)
    observation = _observe_complex(tensor)
    width = 8 if tensor.dtype == torch.complex64 else 16
    word_dtype = torch.uint32 if width == 8 else torch.uint64
    observation["bits"] = [
        f"0x{int(word):0{width}x}"
        for word in tensor.detach().reshape(-1).view(word_dtype).tolist()
    ]
    return observation


def _observe_cumulative_extrema_request(
    torch: Any,
    source: Any,
    dtype_name: str,
    operation: str,
    axis: int,
    label: str,
) -> dict[str, Any]:
    axis_limit = max(source.ndim, 1)
    expected_error = (
        "axis_out_of_range"
        if axis < -axis_limit or axis >= axis_limit
        else "unsupported_dtype"
        if source.ndim > 0
        and source.numel() > 0
        and dtype_name in {"uint16", "uint32", "uint64", "complex64", "complex128"}
        else None
    )
    try:
        output = getattr(torch, operation)(source, dim=axis)
    except (IndexError, NotImplementedError, RuntimeError, TypeError) as error:
        if isinstance(error, IndexError) and (axis < -axis_limit or axis >= axis_limit):
            code = "axis_out_of_range"
        else:
            code = _indexed_extrema_error_code(error)
        if code not in {"axis_out_of_range", "unsupported_dtype"}:
            raise ValueError(
                f"unexpected cumulative {operation} rejection for {label}: {error}"
            ) from error
        if code != expected_error:
            raise ValueError(
                f"unexpected cumulative {operation} rejection for {label}: "
                f"expected {expected_error}, observed {code} ({error})"
            ) from error
        return {"error": code}

    if expected_error is not None:
        raise ValueError(
            f"cumulative {operation} unexpectedly accepted {label}; "
            f"expected {expected_error}"
        )

    if str(output.values.dtype).removeprefix("torch.") != dtype_name:
        raise ValueError(f"cumulative {operation} changed value dtype for {label}")
    if str(output.indices.dtype).removeprefix("torch.") != "int64":
        raise ValueError(f"cumulative {operation} returned non-int64 indices for {label}")
    if output.values.shape != source.shape or output.indices.shape != source.shape:
        raise ValueError(f"cumulative {operation} changed tensor shape for {label}")
    return {
        "values": _observe_cumulative_extrema_values(torch, output.values),
        "indices": _observe_integer_output(output.indices),
    }


def cumulative_extrema(torch: Any) -> dict[str, Any]:
    """Observe actual CPU cummin/cummax outputs and dtype/axis rejections."""
    inputs = _cumulative_extrema_inputs(torch)
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype_name, profiles in inputs.items():
        expected[dtype_name] = {}
        for profile, spec in profiles.items():
            source = _cumulative_extrema_source(torch, dtype_name, spec)
            expected[dtype_name][profile] = {
                f"{operation}_axis_{axis}": _observe_cumulative_extrema_request(
                    torch,
                    source,
                    dtype_name,
                    operation,
                    axis,
                    f"{dtype_name}.{profile}.{operation}_axis_{axis}",
                )
                for operation in ("cummin", "cummax")
                for axis in spec["axes"]
            }
    return _case("tensor.cumulative_extrema", "api:torch", inputs, expected, tolerance=0.0)


_CUMULATIVE_EXTREMA_GRADIENT_DTYPES = (
    "float32", "float64", "complex64", "complex128",
)
_CUMULATIVE_EXTREMA_GRADIENT_SEED_KINDS = (
    "finite", "mixed", "zero", "negative_zero", "nan", "infinity",
    "negative_infinity", "strided",
)


def _cumulative_extrema_gradient_seed(
    torch: Any, output: Any, dtype_name: str, kind: str
) -> Any:
    """Build a cumulative-extrema-only rounded cancellation seed."""
    if kind != "rounding":
        return _logcumsumexp_gradient_seed(torch, output, dtype_name, kind)
    boundary = 2**24 if dtype_name == "float32" else 2**53
    count = output.numel()
    if output.ndim == 0:
        values = [boundary]
    else:
        row_width = output.shape[-1]
        values = (
            [(boundary, 1, -boundary, 1)[(index % row_width) % 4] for index in range(count)]
            if row_width
            else []
        )
    return torch.tensor(values, dtype=getattr(torch, dtype_name), device="cpu").reshape(
        output.shape
    )


def _cumulative_extrema_gradient_inputs(
    torch: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extend the forward corpus without changing its original fixture schema."""
    forward_inputs = _cumulative_extrema_inputs(torch)
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype_name in _CUMULATIVE_EXTREMA_GRADIENT_DTYPES:
        seed_kinds: list[str] = list(_CUMULATIVE_EXTREMA_GRADIENT_SEED_KINDS)
        if dtype_name in ("float32", "float64"):
            seed_kinds.append("rounding")
        if dtype_name in ("complex64", "complex128"):
            seed_kinds.append("imaginary_infinity")
        inputs[dtype_name] = {
            profile: {
                **spec,
                "seed_kinds": seed_kinds,
                "invalid_seed": {"kind": "shape_mismatch"},
            }
            for profile, spec in forward_inputs[dtype_name].items()
        }
    return inputs


def _cumulative_extrema_gradient_expected_error(
    dtype_name: str, source: Any, axis: int
) -> str | None:
    axis_limit = max(source.ndim, 1)
    if axis < -axis_limit or axis >= axis_limit:
        return "axis_out_of_range"
    if dtype_name in ("complex64", "complex128"):
        if source.ndim > 0 and source.numel() > 0:
            return "unsupported_dtype"
        return "unsupported_differentiation"
    return None


def _cumulative_extrema_gradient_error_code(
    error: Exception, axis: int, axis_limit: int
) -> str | None:
    if isinstance(error, IndexError) and (axis < -axis_limit or axis >= axis_limit):
        return "axis_out_of_range"
    if "does not support automatic differentiation for outputs with complex dtype" in str(error):
        return "unsupported_differentiation"
    return _indexed_extrema_error_code(error)


def cumulative_extrema_gradient(torch: Any) -> dict[str, Any]:
    """Observe actual CPU cumulative-extrema outputs and seeded first-order VJPs."""
    inputs = _cumulative_extrema_gradient_inputs(torch)
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    counts = {
        "gradients": 0,
        "axis_out_of_range": 0,
        "unsupported_dtype": 0,
        "unsupported_differentiation": 0,
        "seed_shape_mismatch": 0,
    }
    profile_count = 0
    for dtype_name, profiles in inputs.items():
        expected[dtype_name] = {}
        profile_count += len(profiles)
        for profile, spec in profiles.items():
            source = _cumulative_extrema_source(torch, dtype_name, spec)
            observations: dict[str, Any] = {}
            expected[dtype_name][profile] = observations
            axis_limit = max(source.ndim, 1)
            for operation in ("cummin", "cummax"):
                for axis in spec["axes"]:
                    expected_error = _cumulative_extrema_gradient_expected_error(
                        dtype_name, source, axis
                    )
                    for seed_kind in spec["seed_kinds"]:
                        request_id = f"{operation}_axis_{axis}_seed_{seed_kind}"
                        label = f"{dtype_name}.{profile}.{request_id}"
                        leaf = source.detach().requires_grad_(True)
                        try:
                            output = getattr(torch, operation)(leaf, dim=axis)
                        except (IndexError, NotImplementedError, RuntimeError, TypeError) as error:
                            code = _cumulative_extrema_gradient_error_code(
                                error, axis, axis_limit
                            )
                            if (
                                code is None
                                or code != expected_error
                                or code not in counts
                            ):
                                raise ValueError(
                                    f"unexpected cumulative {operation} rejection for "
                                    f"{label}: expected {expected_error}, observed {code} "
                                    f"({error})"
                                ) from error
                            observations[request_id] = {"error": code}
                            counts[code] += 1
                            continue

                        if expected_error is not None:
                            raise ValueError(
                                f"cumulative {operation} unexpectedly accepted {label}; "
                                f"expected {expected_error}"
                            )
                        seed = _cumulative_extrema_gradient_seed(
                            torch, output.values, dtype_name, seed_kind
                        )
                        try:
                            gradient, = torch.autograd.grad(output.values, leaf, seed)
                        except (IndexError, NotImplementedError, RuntimeError, TypeError) as error:
                            raise ValueError(
                                f"cumulative {operation} unexpectedly rejected the "
                                f"seeded VJP for {label}: {error}"
                            ) from error
                        observations[request_id] = {
                            "output": {
                                "values": _observe_cumulative_extrema_values(
                                    torch, output.values
                                ),
                                "indices": _observe_integer_output(output.indices),
                            },
                            "seed": _observe_logsumexp_gradient_tensor(seed),
                            "gradient": _observe_logsumexp_gradient_tensor(gradient),
                        }
                        counts["gradients"] += 1

                    if expected_error is None:
                        request_id = f"{operation}_axis_{axis}_invalid_seed"
                        label = f"{dtype_name}.{profile}.{request_id}"
                        leaf = source.detach().requires_grad_(True)
                        output = getattr(torch, operation)(leaf, dim=axis)
                        invalid_seed = torch.zeros(
                            output.values.numel() + 1,
                            dtype=getattr(torch, dtype_name),
                            device="cpu",
                        )
                        try:
                            torch.autograd.grad(output.values, leaf, invalid_seed)
                        except (IndexError, NotImplementedError, RuntimeError, TypeError) as error:
                            if not isinstance(error, RuntimeError) or "Mismatch in shape" not in str(error):
                                raise ValueError(
                                    f"unexpected invalid-seed rejection for {label}: {error}"
                                ) from error
                            observations[request_id] = {"error": "seed_shape_mismatch"}
                            counts["seed_shape_mismatch"] += 1
                        else:
                            raise ValueError(
                                f"PyTorch accepted a mismatched cumulative-extrema seed "
                                f"for {label}"
                            )

    expected_counts = {
        "gradients": 2376,
        "axis_out_of_range": 2520,
        "unsupported_dtype": 1152,
        "unsupported_differentiation": 792,
        "seed_shape_mismatch": 264,
    }
    if len(inputs) != 4 or profile_count != 70 or counts != expected_counts:
        raise ValueError(
            "cumulative-extrema gradient corpus mismatch: "
            f"dtypes={len(inputs)}, profiles={profile_count}, counts={counts}"
        )
    return _case(
        "autograd.cumulative_extrema",
        "api:autograd",
        inputs,
        expected,
        tolerance=0.0,
    )






def _aminmax_request_id(dim: int | None, keepdim: bool) -> str:
    axis_name = "all" if dim is None else f"axis_{dim}"
    return f"{axis_name}_{'keep' if keepdim else 'drop'}"


def _aminmax_raw_inputs() -> dict[str, dict[str, Any]]:
    rank_two_dims = [None, 0, 1, -1, -2, 2, -3]
    scalar_dims = [None, 0, -1, 1, -2]
    inputs: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        dtype: str,
        shape: list[int],
        layout: str,
        payload_key: str,
        payload: list[Any],
        dims: list[int | None],
    ) -> None:
        inputs[name] = {
            "dtype": dtype,
            "shape": shape,
            "layout": layout,
            payload_key: payload,
            "requests": [
                {
                    "id": _aminmax_request_id(dim, keepdim),
                    "dim": dim,
                    "keepdim": keepdim,
                }
                for dim in dims
                for keepdim in (False, True)
            ],
        }

    add("bool", "bool", [2, 3], "identity", "words", [0, 1, 0, 1, 0, 1], rank_two_dims)
    add("int8", "int8", [2, 3], "identity", "words", [-128, 127, -1, 1, 2, 3], rank_two_dims)
    add("int16", "int16", [2, 2], "identity", "words", [-32768, 32767, -1, 5], rank_two_dims)
    add(
        "int32",
        "int32",
        [2, 2],
        "identity",
        "words",
        [-(1 << 31), (1 << 31) - 1, -1, 5],
        rank_two_dims,
    )
    add(
        "int64",
        "int64",
        [2, 3],
        "identity",
        "words",
        [-(1 << 63), (1 << 63) - 1, -2, 1, 4, -9],
        rank_two_dims,
    )
    add("uint8", "uint8", [2, 3], "identity", "words", [0, 255, 1, 2, 128, 3], rank_two_dims)
    add(
        "transpose",
        "float64",
        [2, 3],
        "transpose",
        "values",
        [2.0, -3.0, 5.0, 7.0, 0.0, -1.0],
        rank_two_dims,
    )
    add("offset", "int32", [3, 4], "offset", "words", list(range(12)), rank_two_dims)
    add("reverse", "float32", [5], "reverse", "values", [1.0, -5.0, 2.0, 0.0, 4.0], scalar_dims)
    add("broadcast", "uint8", [1, 3], "broadcast", "words", [2, 3, 0], rank_two_dims)
    add("scalar", "int64", [], "identity", "words", [-(1 << 63)], scalar_dims)
    add("empty_reduced", "int8", [2, 0], "identity", "words", [], rank_two_dims)
    add("empty_outer", "int8", [0, 3], "identity", "words", [], rank_two_dims)
    add(
        "zero_ties_f32",
        "float32",
        [2, 3],
        "identity",
        "bits",
        ["0x80000000", "0x00000000", "0x80000000", "0x00000000", "0x80000000", "0x00000000"],
        rank_two_dims,
    )
    add(
        "zero_ties_f64",
        "float64",
        [2, 3],
        "identity",
        "bits",
        [
            "0x8000000000000000",
            "0x0000000000000000",
            "0x8000000000000000",
            "0x0000000000000000",
            "0x8000000000000000",
            "0x0000000000000000",
        ],
        rank_two_dims,
    )
    add(
        "payload_f32",
        "float32",
        [2, 4],
        "identity",
        "bits",
        [
            "0x3f800000",
            "0x7fc12345",
            "0x3f800000",
            "0x3f800000",
            "0x7fc54321",
            "0x3f800000",
            "0x3f800000",
            "0x3f800000",
        ],
        rank_two_dims,
    )
    add(
        "payload_f64",
        "float64",
        [2, 2],
        "identity",
        "bits",
        [
            "0x3ff0000000000000",
            "0x7ff8123456789abc",
            "0x7ff876543210abcd",
            "0x3ff0000000000000",
        ],
        rank_two_dims,
    )
    add("scalar_payload_f32", "float32", [], "identity", "bits", ["0x7fc12345"], scalar_dims)
    add(
        "scalar_payload_f64",
        "float64",
        [],
        "identity",
        "bits",
        ["0x7ff8123456789abc"],
        scalar_dims,
    )
    add(
        "infinities_f32",
        "float32",
        [2, 3],
        "identity",
        "bits",
        [
            "0xff800000",
            "0x7f800000",
            "0x00000000",
            "0x7f800000",
            "0xff800000",
            "0x3f800000",
        ],
        rank_two_dims,
    )
    add(
        "infinities_f64",
        "float64",
        [2, 3],
        "identity",
        "bits",
        [
            "0xfff0000000000000",
            "0x7ff0000000000000",
            "0x0000000000000000",
            "0x7ff0000000000000",
            "0xfff0000000000000",
            "0x3ff0000000000000",
        ],
        rank_two_dims,
    )
    add(
        "uint16_matrix",
        "uint16",
        [2, 3],
        "identity",
        "words",
        [0, 65535, 1, 2, 40000, 3],
        rank_two_dims,
    )
    add("uint32_singleton", "uint32", [1], "identity", "words", [(1 << 32) - 1], scalar_dims)
    add("uint64_empty", "uint64", [0, 3], "identity", "words", [], rank_two_dims)
    add("uint16_scalar", "uint16", [], "identity", "words", [65535], scalar_dims)
    add("uint32_scalar", "uint32", [], "identity", "words", [(1 << 32) - 1], scalar_dims)
    add("uint64_scalar", "uint64", [], "identity", "words", [(1 << 64) - 1], scalar_dims)
    add("complex64_empty", "complex64", [0, 3], "identity", "values", [], rank_two_dims)
    add(
        "complex128_scalar",
        "complex128",
        [],
        "identity",
        "values",
        [[1.0, -2.0]],
        scalar_dims,
    )
    for name in ("uint64_empty", "uint64_scalar"):
        inputs[name]["words"] = [
            _indexed_extrema_word(value) for value in inputs[name]["words"]
        ]
    return inputs


def _aminmax_source(torch: Any, spec: dict[str, Any]) -> Any:
    dtype_name = spec["dtype"]
    dtype = getattr(torch, dtype_name)
    if "words" in spec:
        words = spec["words"]
        values = [word + (1 << 64) if dtype_name == "uint64" and word < 0 else word for word in words]
        source = torch.tensor(values, dtype=dtype, device="cpu")
    elif "bits" in spec:
        if dtype_name == "float32":
            words = [int(word, 16) for word in spec["bits"]]
            words = [word - (1 << 32) if word >= 1 << 31 else word for word in words]
            source = torch.tensor(words, dtype=torch.int32, device="cpu").view(dtype)
        else:
            words = [_indexed_extrema_word(int(word, 16)) for word in spec["bits"]]
            source = torch.tensor(words, dtype=torch.int64, device="cpu").view(dtype)
    elif dtype_name in ("complex64", "complex128"):
        source = torch.tensor(
            [complex(real, imaginary) for real, imaginary in spec["values"]],
            dtype=dtype,
            device="cpu",
        )
    else:
        source = torch.tensor(spec["values"], dtype=dtype, device="cpu")
    source = source.reshape(spec["shape"])
    layout = spec["layout"]
    if layout == "transpose":
        return source.transpose(0, 1)
    if layout == "offset":
        return source.transpose(0, 1)[:, 1:]
    if layout == "reverse":
        return source.flip([0])
    if layout == "broadcast":
        return source.expand(4, 3)
    return source


def _observe_aminmax_tensor(torch: Any, tensor: Any) -> dict[str, Any]:
    if not tensor.is_floating_point() and not tensor.is_complex():
        return _observe_integer_output(tensor)
    observation = _observe(tensor)
    if observation["dtype"] in ("float32", "float64"):
        width = 8 if observation["dtype"] == "float32" else 16
        word_dtype = torch.uint32 if width == 8 else torch.uint64
        observation["bits"] = [
            f"0x{int(word):0{width}x}"
            for word in tensor.detach().reshape(-1).view(word_dtype).tolist()
        ]
    return observation


def _aminmax_error_code(error: Exception) -> str | None:
    message = str(error).lower().replace("non-zero", "nonzero")
    if isinstance(error, (NotImplementedError, TypeError)):
        return "unsupported_dtype"
    if isinstance(error, IndexError):
        if "dimension out of range" in message or "axis out of range" in message:
            return "axis_out_of_range"
        if "nonzero size" in message or "empty reduced axis" in message:
            return "empty_reduced_axis"
    if isinstance(error, RuntimeError):
        if "nonzero size" in message or "empty reduced axis" in message:
            return "empty_reduced_axis"
        if (
            "empty dimension" in message
            or "empty reduction" in message
            or "numel() == 0" in message
            or "reduction dim to be specified" in message
        ):
            return "empty_reduction"
        if "not implemented" in message or "unsupported" in message:
            return "unsupported_dtype"
    return None


def _observe_aminmax_request(
    torch: Any, source: Any, request: dict[str, Any], label: str
) -> dict[str, Any]:
    try:
        output = torch.aminmax(source, dim=request["dim"], keepdim=request["keepdim"])
    except (IndexError, NotImplementedError, RuntimeError, TypeError) as error:
        code = _aminmax_error_code(error)
        if code is None:
            raise ValueError(f"unexpected aminmax rejection for {label}: {error}") from error
        return {"error": code}
    minimum, maximum = output.min, output.max
    return {
        "min": _observe_aminmax_tensor(torch, minimum),
        "max": _observe_aminmax_tensor(torch, maximum),
        "distinct": minimum.numel() == 0 or minimum.data_ptr() != maximum.data_ptr(),
    }


def aminmax(torch: Any) -> dict[str, Any]:
    """Observe authentic PyTorch CPU joint aminmax tuple results and rejections."""
    inputs = _aminmax_raw_inputs()
    expected: dict[str, dict[str, Any]] = {}
    for name, spec in inputs.items():
        source = _aminmax_source(torch, spec)
        expected[name] = {
            request["id"]: _observe_aminmax_request(
                torch, source, request, f"{name}.{request['id']}"
            )
            for request in spec["requests"]
        }
    return _case("tensor.aminmax", "api:torch", inputs, expected, tolerance=0.0)




def _logsumexp_request(
    kind: str, dimensions: int | list[int] | None, keepdim: bool
) -> dict[str, Any]:
    if kind == "whole":
        name = "all"
    elif kind == "axis":
        name = f"axis_{dimensions}"
    else:
        assert isinstance(dimensions, list)
        name = "axes_" + ("empty" if not dimensions else "_".join(map(str, dimensions)))
    return {
        "id": f"{name}_{'keep' if keepdim else 'drop'}",
        "kind": kind,
        "keepdim": keepdim,
        **({"axis": dimensions} if kind == "axis" else
           {"axes": dimensions} if kind == "axes" else {}),
    }


def _logsumexp_requests(profile: str) -> list[dict[str, Any]]:
    if profile == "rank2":
        return [
            _logsumexp_request("whole", None, False),
            _logsumexp_request("whole", None, True),
            _logsumexp_request("axis", 0, False),
            _logsumexp_request("axis", -1, True),
            _logsumexp_request("axes", [0, 1], False),
            _logsumexp_request("axes", [1, 0], True),
            _logsumexp_request("axes", [], False),
            _logsumexp_request("axes", [], True),
            _logsumexp_request("axis", 2, False),
            _logsumexp_request("axes", [0, -2], False),
        ]
    if profile == "multi_axes":
        return [
            _logsumexp_request("whole", None, False),
            _logsumexp_request("whole", None, True),
            _logsumexp_request("axis", 1, False),
            _logsumexp_request("axes", [0, 2], False),
            _logsumexp_request("axes", [2, 0], True),
            _logsumexp_request("axes", [0, -1], False),
            _logsumexp_request("axes", [], False),
            _logsumexp_request("axes", [], True),
            _logsumexp_request("axis", 3, False),
            _logsumexp_request("axes", [0, -3], False),
        ]
    if profile == "scalar":
        return [
            _logsumexp_request("whole", None, False),
            _logsumexp_request("whole", None, True),
            _logsumexp_request("axis", 0, False),
            _logsumexp_request("axis", -1, True),
            _logsumexp_request("axes", [], False),
            _logsumexp_request("axes", [], True),
            _logsumexp_request("axis", 1, False),
            _logsumexp_request("axis", -2, False),
        ]
    if profile == "empty":
        return [
            _logsumexp_request("whole", None, False),
            _logsumexp_request("whole", None, True),
            _logsumexp_request("axis", 0, False),
            _logsumexp_request("axis", 1, False),
            _logsumexp_request("axis", 2, True),
            _logsumexp_request("axes", [1], True),
            _logsumexp_request("axes", [0, 2], False),
            _logsumexp_request("axes", [], False),
            _logsumexp_request("axes", [], True),
            _logsumexp_request("axis", 3, False),
            _logsumexp_request("axes", [0, -3], False),
        ]
    if profile == "empty_outer":
        return [
            _logsumexp_request("whole", None, False),
            _logsumexp_request("whole", None, True),
            _logsumexp_request("axis", 0, False),
            _logsumexp_request("axis", 1, True),
            _logsumexp_request("axes", [], False),
            _logsumexp_request("axes", [], True),
            _logsumexp_request("axis", 2, False),
        ]
    if profile == "vector":
        return [
            _logsumexp_request("whole", None, False),
            _logsumexp_request("whole", None, True),
            _logsumexp_request("axis", 0, False),
            _logsumexp_request("axis", -1, True),
            _logsumexp_request("axes", [], False),
            _logsumexp_request("axes", [], True),
            _logsumexp_request("axis", 1, False),
            _logsumexp_request("axes", [0, -1], False),
        ]
    raise ValueError(f"unknown logsumexp request profile: {profile}")


def _logsumexp_raw_inputs() -> dict[str, dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        dtype: str,
        shape: list[int],
        layout: str,
        payload_key: str,
        payload: list[Any],
        profile: str,
        specials: list[dict[str, Any]] | None = None,
    ) -> None:
        inputs[name] = {
            "dtype": dtype,
            "shape": shape,
            "layout": layout,
            payload_key: payload,
            "specials": [] if specials is None else specials,
            "requests": _logsumexp_requests(profile),
        }

    integer_values: dict[str, list[int]] = {
        "bool": [0, 1, 0, 1, 0, 1],
        "int8": [-128, 127, -1, 1, 2, 3],
        "int16": [-(1 << 15), (1 << 15) - 1, -1, 1, 2, 3],
        "int32": [-(1 << 31), (1 << 31) - 1, -1, 1, 2, 3],
        "int64": [-(1 << 63), (1 << 63) - 1, -2, 1, 4, -9],
        "uint8": [0, 255, 1, 2, 128, 3],
        "uint16": [0, 65535, 1, 2, 40000, 3],
        "uint32": [0, (1 << 32) - 1, 1, 2, (1 << 32) - 2, 3],
        "uint64": [0, (1 << 64) - 1, 1, (1 << 63) + 7, (1 << 63), 3],
    }
    for dtype, values in integer_values.items():
        words = [_indexed_extrema_word(value) for value in values] if dtype == "uint64" else values
        add(dtype, dtype, [2, 3], "identity", "words", words, "rank2")

    for dtype, values in (
        ("float32", [0.0, 1.0, -1.0, 3.0, -4.0, 2.0]),
        ("float64", [0.0, 1.0, -1.0, 3.0, -4.0, 2.0]),
    ):
        add(dtype, dtype, [2, 3], "identity", "values", values, "rank2")

    add("complex64", "complex64", [2, 3], "identity", "values",
        [[1.0, 0.5], [-2.0, 1.0], [0.0, -1.0],
         [3.0, -0.5], [0.25, 2.0], [-1.0, 0.0]], "rank2")
    add("complex128", "complex128", [2, 3], "identity", "values",
        [[1.0, -0.5], [2.0, 1.25], [-3.0, 0.75],
         [0.5, -2.0], [-1.0, 0.25], [4.0, 1.0]], "rank2")

    add("transpose", "float64", [2, 3], "transpose", "values",
        [2.0, -3.0, 5.0, 7.0, 0.0, -1.0], "rank2")
    add("offset", "int32", [3, 4], "offset", "words",
        list(range(12)), "rank2")
    add("reverse", "float32", [5], "reverse", "values",
        [1.0, -5.0, 2.0, 0.0, 4.0], "vector")
    add("broadcast", "uint8", [1, 3], "broadcast", "words",
        [2, 3, 0], "rank2")
    add("multi_axes", "float32", [2, 3, 2], "identity", "values",
        [-5.0, 9.0, 8.0, 4.0, 3.0, 2.0, 7.0, 6.0, 1.0, 0.0, 11.0, -2.0],
        "multi_axes")

    add("scalar_negative_zero", "float32", [], "identity", "values",
        [-0.0], "scalar")
    add("complex_principal", "complex128", [], "identity", "values",
        [[2.0, 7.0]], "scalar")
    add("empty_reduced", "float32", [2, 0, 3], "identity", "values",
        [], "empty")
    add("empty_outer", "float64", [0, 3], "identity", "values",
        [], "empty_outer")
    add("empty_complex", "complex128", [0, 3], "identity", "values",
        [], "empty_outer")
    add("stable_float32", "float32", [2], "identity", "values",
        [1000.0, 999.0], "vector")
    add("nonfinite_float32", "float32", [2, 3], "identity", "values",
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "rank2",
        specials=[
            {"index": 0, "value": "-inf"},
            {"index": 1, "value": "+inf"},
            {"index": 2, "value": "nan"},
        ])
    add("all_negative_infinity", "float64", [3], "identity", "values",
        [0.0, 1.0, 2.0], "vector",
        specials=[{"index": index, "value": "-inf"} for index in range(3)])
    add("complex_nonfinite", "complex128", [5], "identity", "values",
        [[1.0, 0.0], [2.0, 1.0], [3.0, -1.0], [4.0, 2.0], [5.0, 0.5]],
        "vector",
        specials=[
            {"index": 0, "component": "real", "value": "+inf"},
            {"index": 1, "component": "real", "value": "-inf"},
            {"index": 2, "component": "imag", "value": "nan"},
            {"index": 3, "component": "real", "value": "nan"},
        ])
    return inputs


def _logsumexp_source(torch: Any, spec: dict[str, Any]) -> Any:
    dtype_name = spec["dtype"]
    dtype = getattr(torch, dtype_name)
    if "words" in spec:
        words = spec["words"]
        values = [
            word + (1 << 64) if dtype_name == "uint64" and word < 0 else word
            for word in words
        ]
        source = torch.tensor(values, dtype=dtype, device="cpu")
    elif dtype_name in ("complex64", "complex128"):
        source = torch.tensor(
            [complex(real, imaginary) for real, imaginary in spec["values"]],
            dtype=dtype,
            device="cpu",
        )
    else:
        source = torch.tensor(spec["values"], dtype=dtype, device="cpu")
    source = source.reshape(spec["shape"])
    for special in spec["specials"]:
        value = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}[
            special["value"]
        ]
        if dtype_name in ("complex64", "complex128"):
            component = source.real if special["component"] == "real" else source.imag
            component.reshape(-1)[special["index"]] = value
        else:
            source.reshape(-1)[special["index"]] = value
    if spec["layout"] == "transpose":
        return source.transpose(0, 1)
    if spec["layout"] == "offset":
        return source.transpose(0, 1)[:, 1:]
    if spec["layout"] == "reverse":
        return source.flip([0])
    if spec["layout"] == "broadcast":
        return source.expand(4, 3)
    return source


def _logsumexp_error_code(error: Exception) -> str | None:
    message = str(error).lower().replace("non-zero", "nonzero")
    if isinstance(error, IndexError):
        return "invalid_axis"
    if isinstance(error, RuntimeError):
        if "appears multiple times" in message or "multiple times" in message:
            return "duplicate_axes"
        if "broadcast" in message and "shape" in message:
            return "empty_axes_drop_mismatch"
    return None


def _observe_logsumexp_request(
    torch: Any, source: Any, request: dict[str, Any], label: str
) -> dict[str, Any]:
    kind = request["kind"]
    keepdim = request["keepdim"]
    if kind == "whole":
        dimensions: int | tuple[int, ...] = tuple(range(source.ndim))
    elif kind == "axis":
        dimensions = request["axis"]
    else:
        dimensions = tuple(request["axes"])
    try:
        output = torch.logsumexp(source, dim=dimensions, keepdim=keepdim)
    except (IndexError, RuntimeError, TypeError) as error:
        code = _logsumexp_error_code(error)
        if code is None:
            raise ValueError(f"unexpected logsumexp rejection for {label}: {error}") from error
        return {"error": code}
    if output.is_complex():
        return _observe_complex(output)
    return _observe(output)


def logsumexp(torch: Any) -> dict[str, Any]:
    """Observe actual PyTorch CPU logsumexp values, layouts and rejection precedence."""
    inputs = _logsumexp_raw_inputs()
    expected: dict[str, dict[str, Any]] = {}
    for name, spec in inputs.items():
        source = _logsumexp_source(torch, spec)
        expected[name] = {
            request["id"]: _observe_logsumexp_request(
                torch, source, request, f"{name}.{request['id']}"
            )
            for request in spec["requests"]
        }
    return _case("tensor.logsumexp", "api:torch", inputs, expected, tolerance=1e-6)


def _logsumexp_gradient_requests(profile: str) -> list[dict[str, Any]]:
    basic_seeds = ["finite", "negative_zero", "positive_infinity", "nan", "ramp"]
    strided_seeds = [*basic_seeds, "strided"]

    def request(
        kind: str,
        dimensions: int | list[int] | None,
        keepdim: bool,
        seed_kinds: list[str] | None = None,
    ) -> dict[str, Any]:
        value = _logsumexp_request(kind, dimensions, keepdim)
        value["seed_kinds"] = list(basic_seeds if seed_kinds is None else seed_kinds)
        return value

    if profile in ("dense", "transpose", "offset", "broadcast", "nonfinite"):
        return [
            request("whole", None, False),
            request("whole", None, True),
            request("axis", 0, False, strided_seeds),
            request("axis", 0, True),
            request("axis", -1, False, strided_seeds),
            request("axis", -1, True),
            request("axes", [0, 1], False),
            request("axes", [1, 0], True),
            request("axes", [], True),
            request("axes", [], False, []),
            request("axis", 2, False, []),
            request("axes", [0, -2], False, []),
            request("axes", [0, 2, 0], False, []),
        ]
    if profile == "multi_axes":
        return [
            request("whole", None, False),
            request("whole", None, True),
            request("axis", 1, False, strided_seeds),
            request("axis", 1, True),
            request("axes", [0, 2], False, strided_seeds),
            request("axes", [2, 0], True),
            request("axes", [], True),
            request("axes", [], False, []),
            request("axis", 3, False, []),
            request("axes", [0, -3], False, []),
        ]
    if profile == "cube":
        return [
            request("whole", None, False),
            request("whole", None, True),
            request("axis", 0, False, strided_seeds),
            request("axis", 1, True),
            request("axis", -1, False),
            request("axes", [0, 2], False, strided_seeds),
            request("axes", [2, 0], True),
            request("axes", [], True),
            request("axes", [], False, []),
            request("axis", 3, False, []),
            request("axes", [0, -3], False, []),
            request("axes", [0, 3, 0], False, []),
        ]
    if profile in ("reverse", "rounded_forward", "complex_phase"):
        return [
            request("whole", None, False),
            request("whole", None, True),
            request("axis", 0, False),
            request("axis", 0, True),
            request("axes", [], True),
            request("axes", [], False, []),
            request("axis", 1, False, []),
            request("axes", [0, -1], False, []),
        ]
    if profile == "scalar":
        return [
            request("whole", None, False),
            request("whole", None, True),
            request("axis", 0, False),
            request("axis", 0, True),
            request("axes", [], False),
            request("axes", [], True),
            request("axis", 1, False, []),
        ]
    if profile == "empty":
        return [
            request("whole", None, False),
            request("whole", None, True),
            request("axis", 0, False),
            request("axis", 0, True),
            request("axis", 1, False),
            request("axis", 1, True),
            request("axes", [], False),
            request("axes", [], True),
            request("axis", 2, False, []),
            request("axes", [0, -2], False, []),
        ]
    raise ValueError(f"unknown logsumexp gradient request profile: {profile}")


def _logsumexp_gradient_raw_inputs() -> dict[str, dict[str, dict[str, Any]]]:
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype in ("float32", "float64", "complex64", "complex128"):
        is_complex = dtype in ("complex64", "complex128")
        cases: dict[str, dict[str, Any]] = {}
        inputs[dtype] = cases

        def add(
            name: str,
            shape: list[int],
            layout: str,
            real_values: list[int | float],
            complex_values: list[list[int | float]] | None = None,
            specials: list[dict[str, Any]] | None = None,
        ) -> None:
            if is_complex:
                if complex_values is None:
                    values: list[Any] = [[value, 0.0] for value in real_values]
                else:
                    values = complex_values
            else:
                values = real_values
            cases[name] = {
                "dtype": dtype,
                "shape": shape,
                "layout": layout,
                "values": values,
                "specials": [] if specials is None else specials,
                "requests": _logsumexp_gradient_requests(name),
                "invalid_seed": {"kind": "shape_mismatch"},
            }

        add(
            "dense",
            [2, 3],
            "identity",
            [1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
            [[1.0, 0.5], [-2.0, 1.0], [0.0, -1.0],
             [3.0, -0.5], [0.25, 2.0], [-1.0, 0.0]],
        )
        add(
            "multi_axes",
            [2, 3, 2],
            "identity",
            [index % 5 - 2 for index in range(12)],
            [[index % 5 - 2, (index % 3 - 1) * 0.5] for index in range(12)],
        )
        add(
            "cube",
            [2, 2, 2],
            "identity",
            [index % 5 - 2 for index in range(8)],
            [[index % 5 - 2, (index % 3 - 1) * 0.5] for index in range(8)],
        )
        add(
            "transpose",
            [2, 3],
            "transpose",
            [1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
            [[1.0, 0.5], [-2.0, 1.0], [0.0, -1.0],
             [3.0, -0.5], [0.25, 2.0], [-1.0, 0.0]],
        )
        add(
            "offset",
            [3, 4],
            "offset",
            [index % 7 - 3 for index in range(12)],
            [[index % 7 - 3, (index % 4 - 1) * 0.25] for index in range(12)],
        )
        add(
            "reverse",
            [5],
            "reverse",
            [2.0, -1.0, 3.0, 0.25, -2.0],
            [[2.0, 0.5], [-1.0, 1.0], [3.0, -0.75],
             [0.25, 2.0], [-2.0, -0.5]],
        )
        add(
            "broadcast",
            [1, 3],
            "broadcast",
            [0.5, -1.0, 2.0],
            [[0.5, 0.25], [-1.0, -0.5], [2.0, 1.0]],
        )
        add(
            "nonfinite",
            [2, 3],
            "identity",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [[0.0, 0.5], [1.0, -0.25], [2.0, 1.0],
             [3.0, -1.0], [4.0, 0.75], [5.0, -0.5]],
            specials=(
                [
                    {"index": 0, "component": "real", "value": "+inf"},
                    {"index": 1, "component": "imag", "value": "nan"},
                    {"index": 4, "component": "real", "value": "-inf"},
                ]
                if is_complex
                else [
                    {"index": 0, "value": "-inf"},
                    {"index": 1, "value": "+inf"},
                    {"index": 2, "value": "nan"},
                ]
            ),
        )
        add(
            "scalar",
            [],
            "identity",
            [-0.0],
            [[2.0, 7.0]],
        )
        add("empty", [0, 3], "identity", [])
        add(
            "rounded_forward",
            [2],
            "identity",
            [1000.0, 999.0],
            [[1000.0, 0.0], [999.0, 0.0]],
        )
        add(
            "complex_phase",
            [2],
            "identity",
            [2.0, 1.0],
            [[2.0, 7.0], [1.0, -3.0]],
        )
    return inputs


def _observe_logsumexp_gradient_tensor(tensor: Any) -> dict[str, Any]:
    if tensor.is_complex():
        return _observe_complex(tensor)
    observation = _observe(tensor)
    values = tensor.detach().reshape(-1).tolist()
    observation["negative_zero_indices"] = [
        index for index, value in enumerate(values)
        if value == 0 and math.copysign(1.0, float(value)) < 0.0
    ]
    return observation


def _logsumexp_gradient_seed(torch: Any, output: Any, dtype_name: str,
                             kind: str) -> Any:
    dtype = getattr(torch, dtype_name)
    count = output.numel()
    is_complex = dtype_name in ("complex64", "complex128")
    if kind == "finite":
        value = complex(2.0, -0.5) if is_complex else 2.0
        values = [value] * count
    elif kind == "negative_zero":
        value = complex(-0.0, -0.0) if is_complex else -0.0
        values = [value] * count
    elif kind == "positive_infinity":
        value = complex(float("inf"), 0.0) if is_complex else float("inf")
        values = [value] * count
    elif kind == "nan":
        value = complex(float("nan"), 0.0) if is_complex else float("nan")
        values = [value] * count
    elif kind in ("ramp", "strided"):
        values = (
            [complex(index % 5 - 2, (index % 3 - 1) * 0.5)
             for index in range(count)]
            if is_complex
            else [index % 5 - 2 for index in range(count)]
        )
        if kind == "strided":
            if output.ndim not in (1, 2) or count == 0:
                raise ValueError("invalid strided logsumexp gradient seed shape")
            backing = torch.zeros(count * 2 + 1, dtype=dtype, device="cpu")
            backing[1::2] = torch.tensor(values, dtype=dtype, device="cpu")
            return backing[1::2].reshape(output.shape)
    else:
        raise ValueError(f"unknown logsumexp gradient seed kind: {kind}")
    return torch.tensor(values, dtype=dtype, device="cpu").reshape(output.shape)


def _logsumexp_gradient_forward(torch: Any, source: Any, request: dict[str, Any]) -> Any:
    kind = request["kind"]
    if kind == "whole":
        dimensions: int | tuple[int, ...] = tuple(range(source.ndim))
    elif kind == "axis":
        dimensions = request["axis"]
    else:
        dimensions = tuple(request["axes"])
    return torch.logsumexp(source, dim=dimensions, keepdim=request["keepdim"])


def _observe_logsumexp_gradient_request(
    torch: Any, source: Any, dtype_name: str, request: dict[str, Any], label: str
) -> dict[str, Any]:
    request_id = request["id"]
    leaf = source.detach().requires_grad_(True)
    try:
        output = _logsumexp_gradient_forward(torch, leaf, request)
    except (IndexError, RuntimeError, TypeError) as error:
        code = _logsumexp_error_code(error)
        if code is None:
            raise ValueError(f"unexpected logsumexp gradient rejection for {label}: {error}") from error
        return {request_id: {"error": code}}

    observations: dict[str, Any] = {}
    for index, seed_kind in enumerate(request["seed_kinds"]):
        if index > 0:
            leaf = source.detach().requires_grad_(True)
            output = _logsumexp_gradient_forward(torch, leaf, request)
        seed = _logsumexp_gradient_seed(torch, output, dtype_name, seed_kind)
        output.backward(seed)
        if leaf.grad is None:
            raise ValueError(f"logsumexp gradient did not populate the input gradient for {label}")
        observations[f"{request_id}_seed_{seed_kind}"] = {
            "output": _observe_logsumexp_gradient_tensor(output),
            "seed": _observe_logsumexp_gradient_tensor(seed),
            "gradient": _observe_logsumexp_gradient_tensor(leaf.grad),
        }

    invalid_leaf = source.detach().requires_grad_(True)
    invalid_output = _logsumexp_gradient_forward(torch, invalid_leaf, request)
    invalid_seed = torch.zeros(
        invalid_output.numel() + 1, dtype=invalid_output.dtype, device="cpu"
    )
    try:
        invalid_output.backward(invalid_seed)
    except RuntimeError as error:
        message = str(error).lower()
        if "shape" not in message and "size" not in message:
            raise
        observations[f"{request_id}_invalid_seed"] = {"error": "seed_shape_mismatch"}
    else:
        raise ValueError(f"PyTorch accepted a mismatched logsumexp gradient seed for {label}")
    return observations


def logsumexp_gradient(torch: Any) -> dict[str, Any]:
    """Observe authentic CPU logsumexp forward values and seeded first-order VJPs."""
    inputs = _logsumexp_gradient_raw_inputs()
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype_name, cases in inputs.items():
        expected[dtype_name] = {}
        for name, spec in cases.items():
            source = _logsumexp_source(torch, spec)
            observations: dict[str, Any] = {}
            for request in spec["requests"]:
                observations.update(
                    _observe_logsumexp_gradient_request(
                        torch, source, dtype_name, request, f"{dtype_name}.{name}.{request['id']}"
                    )
                )
            expected[dtype_name][name] = observations
    return _case("autograd.logsumexp", "api:autograd", inputs, expected, tolerance=1e-6)


def _logcumsumexp_raw_inputs() -> dict[str, dict[str, dict[str, Any]]]:
    """Describe the complete CPU dtype/profile matrix from the pinned probe."""
    dtype_names = (
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
        "uint64", "float32", "float64", "complex64", "complex128",
    )
    integer_names = {
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    }
    supported_names = {"float32", "float64", "complex64", "complex128"}
    real_specials = [
        float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")
    ]
    complex_specials = [
        complex(float("-inf"), float("inf")),
        complex(float("-inf"), float("-inf")),
        complex(float("-inf"), float("nan")),
        complex(float("-inf"), -0.0),
        complex(float("inf"), float("inf")),
        complex(float("inf"), float("nan")),
        complex(float("inf"), -0.0),
        complex(float("nan"), 0.0),
        complex(0.0, float("nan")),
        complex(1.0, 0.0),
        complex(1.0, math.pi),
        complex(1.0, -math.pi),
    ]

    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype_name in dtype_names:
        is_integer = dtype_name in integer_names
        is_complex = dtype_name in ("complex64", "complex128")
        cases: dict[str, dict[str, Any]] = {}
        inputs[dtype_name] = cases

        def add(
            name: str,
            shape: list[int],
            layout: str,
            raw_values: Sequence[int | float | complex],
        ) -> None:
            specials: list[dict[str, Any]] = []

            def encode(value: float, index: int, component: str | None = None) -> float:
                if math.isnan(value):
                    special_value = "nan"
                elif math.isinf(value):
                    special_value = "+inf" if value > 0 else "-inf"
                else:
                    return float(value)
                special: dict[str, Any] = {"index": index, "value": special_value}
                if component is not None:
                    special["component"] = component
                specials.append(special)
                return 0.0

            if is_integer:
                payload_key = "words"
                payload = []
                for raw_value in raw_values:
                    if isinstance(raw_value, complex):
                        raise ValueError("integer logcumsumexp input cannot be complex")
                    payload.append(int(bool(raw_value)) if dtype_name == "bool" else int(raw_value))
            elif is_complex:
                payload_key = "values"
                payload = []
                for index, raw_value in enumerate(raw_values):
                    value = complex(raw_value)
                    payload.append([
                        encode(value.real, index, "real"),
                        encode(value.imag, index, "imag"),
                    ])
            else:
                payload_key = "values"
                payload = []
                for raw_value in raw_values:
                    if isinstance(raw_value, complex):
                        raise ValueError("real logcumsumexp input cannot be complex")
                    payload.append(encode(float(raw_value), len(payload)))

            axis_limit = max(len(shape), 1)
            cases[name] = {
                "dtype": dtype_name,
                "shape": shape,
                "layout": layout,
                payload_key: payload,
                "specials": specials,
                "requests": [
                    {"id": f"axis_{axis}", "axis": axis}
                    for axis in range(-axis_limit - 1, axis_limit + 1)
                ],
            }

        base = list(range(6))
        add("dense", [2, 3], "identity", base)
        add("transpose", [2, 3], "transpose", base)
        add("offset", [2, 3], "offset_columns", base)
        add("reverse", [2, 3], "reverse_both", base)
        add("broadcast", [1, 3], "broadcast", base[:3])
        add("cube", [2, 2, 3], "identity", base * 2)
        add("scalar", [], "identity", [0])
        add("empty", [2, 0, 3], "identity", [])

        if dtype_name in supported_names:
            add("zero_signs", [3], "identity", [-0.0, 0.0, -0.0])
            add("large", [3], "identity", [1000.0, 999.0, -1000.0])
            add(
                "nonfinite", [4], "identity",
                [float("-inf"), 1.0, float("inf"), float("nan")],
            )
            if is_complex:
                add(
                    "phases", [3], "identity",
                    [complex(2, 7), complex(1, -8), complex(-2, 0.5)],
                )
                add(
                    "complex_nonfinite", [3], "identity",
                    [
                        complex(float("-inf"), float("inf")),
                        complex(1, 0),
                        complex(float("inf"), -0.0),
                    ],
                )
            for left_index, left in enumerate(real_specials):
                for right_index, right in enumerate(real_specials):
                    add(
                        f"real_pair_{left_index}_{right_index}",
                        [2],
                        "identity",
                        [left, right],
                    )
            if is_complex:
                for left_index, left in enumerate(complex_specials):
                    for right_index, right in enumerate(complex_specials):
                        add(
                            f"complex_pair_{left_index}_{right_index}",
                            [2],
                            "identity",
                            [left, right],
                        )
    return inputs


def _logcumsumexp_source(torch: Any, spec: dict[str, Any]) -> Any:
    dtype_name = spec["dtype"]
    dtype = getattr(torch, dtype_name)
    if "words" in spec:
        source = torch.tensor(spec["words"], dtype=dtype, device="cpu")
    elif dtype_name in ("complex64", "complex128"):
        source = torch.tensor(
            [complex(real, imaginary) for real, imaginary in spec["values"]],
            dtype=dtype,
            device="cpu",
        )
    else:
        source = torch.tensor(spec["values"], dtype=dtype, device="cpu")
    source = source.reshape(spec["shape"])
    for special in spec["specials"]:
        value = {
            "nan": float("nan"),
            "+inf": float("inf"),
            "-inf": float("-inf"),
        }[special["value"]]
        if dtype_name in ("complex64", "complex128"):
            component = source.real if special["component"] == "real" else source.imag
            component.reshape(-1)[special["index"]] = value
        else:
            source.reshape(-1)[special["index"]] = value
    layout = spec["layout"]
    if layout == "transpose":
        return source.transpose(0, 1)
    if layout == "offset_columns":
        return torch.cat([source, source], dim=1)[:, 1:5]
    if layout == "reverse_both":
        return source.flip([0, 1])
    if layout == "broadcast":
        return source.expand(4, 3)
    return source


def _logcumsumexp_error_code(error: Exception, dtype_name: str) -> str | None:
    if isinstance(error, IndexError):
        return "invalid_axis"
    if isinstance(error, RuntimeError) and dtype_name in {
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    }:
        return "unsupported_dtype"
    return None


def _observe_logcumsumexp_request(
    torch: Any,
    source: Any,
    dtype_name: str,
    request: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    try:
        output = torch.logcumsumexp(source, dim=request["axis"])
    except (RuntimeError, IndexError) as error:
        code = _logcumsumexp_error_code(error, dtype_name)
        if code is None:
            raise ValueError(f"unexpected logcumsumexp rejection for {label}: {error}") from error
        return {"error": code}
    return _observe_logsumexp_gradient_tensor(output)


def logcumsumexp(torch: Any) -> dict[str, Any]:
    """Observe actual PyTorch CPU logcumsumexp outputs and dtype/axis rejections."""
    inputs = _logcumsumexp_raw_inputs()
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype_name, profiles in inputs.items():
        expected[dtype_name] = {}
        for profile, spec in profiles.items():
            source = _logcumsumexp_source(torch, spec)
            expected[dtype_name][profile] = {
                request["id"]: _observe_logcumsumexp_request(
                    torch,
                    source,
                    dtype_name,
                    request,
                    f"{dtype_name}.{profile}.{request['id']}",
                )
                for request in spec["requests"]
            }
    return _case(
        "tensor.logcumsumexp",
        "api:torch",
        inputs,
        expected,
        tolerance=1e-6,
    )


def _logcumsumexp_gradient_raw_inputs() -> dict[str, dict[str, dict[str, Any]]]:
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    seed_names = (
        "finite", "mixed", "zero", "negative_zero", "infinity",
        "negative_infinity", "nan", "strided",
    )
    for dtype_name in ("float32", "float64", "complex64", "complex128"):
        is_complex = dtype_name in ("complex64", "complex128")
        cases: dict[str, dict[str, Any]] = {}
        inputs[dtype_name] = cases

        def add(
            name: str,
            shape: list[int],
            layout: str,
            real_values: list[float],
            complex_values: list[list[float]] | None = None,
            specials: list[dict[str, Any]] | None = None,
        ) -> None:
            values: list[Any]
            if is_complex:
                values = (
                    complex_values
                    if complex_values is not None
                    else [[value, 0.0] for value in real_values]
                )
            else:
                values = real_values
            axis_limit = max(len(shape), 1)
            requests = [
                {
                    "id": f"axis_{axis}",
                    "axis": axis,
                    "seed_kinds": [
                        *seed_names,
                        *(["imaginary_infinity"] if is_complex else []),
                    ],
                }
                for axis in range(-axis_limit - 1, axis_limit + 1)
            ]
            cases[name] = {
                "dtype": dtype_name,
                "shape": shape,
                "layout": layout,
                "values": values,
                "specials": [] if specials is None else specials,
                "requests": requests,
                "invalid_seed": {"kind": "shape_mismatch"},
            }

        base_real = [0.0, 1.0, -2.0, 3.0, -1.0, 2.0]
        base_complex = [
            [0.0, 0.25], [1.0, -0.5], [-2.0, 7.0],
            [3.0, 1.0], [-1.0, -2.0], [2.0, 0.0],
        ]
        add("dense", [2, 3], "identity", base_real, base_complex)
        add("transpose", [2, 3], "transpose", base_real, base_complex)
        add("offset", [2, 3], "offset_columns", base_real, base_complex)
        add("reverse", [2, 3], "reverse_both", base_real, base_complex)
        add("broadcast", [1, 3], "broadcast", base_real[:3], base_complex[:3])
        add("cube", [2, 2, 3], "identity", base_real * 2, base_complex * 2)
        add(
            "scalar", [], "identity", [-0.0],
            [[2.0, 7.0]] if is_complex else None,
        )
        add(
            "scalar_nan", [], "identity", [0.0],
            [[0.0, 0.0]] if is_complex else None,
            (
                [
                    {"index": 0, "value": "nan", "component": "real"},
                    {"index": 0, "value": "+inf", "component": "imag"},
                ]
                if is_complex
                else [{"index": 0, "value": "nan"}]
            ),
        )
        add("empty", [2, 0, 3], "identity", [])
        add("large", [3], "identity", [1000.0, 999.0, -1000.0])
        add(
            "all_negative_infinity", [2], "identity", [0.0, 0.0],
            specials=[
                {"index": 0, "value": "-inf", **({"component": "real"} if is_complex else {})},
                {"index": 1, "value": "-inf", **({"component": "real"} if is_complex else {})},
            ],
        )
        add(
            "nonfinite", [4], "identity", [0.0, 1.0, 0.0, 0.0],
            specials=[
                {"index": 0, "value": "-inf", **({"component": "real"} if is_complex else {})},
                {"index": 2, "value": "+inf", **({"component": "real"} if is_complex else {})},
                {"index": 3, "value": "nan", **({"component": "real"} if is_complex else {})},
            ],
        )
        if is_complex:
            add(
                "complex_phase", [3], "identity", [],
                [[2.0, 7.0], [1.0, -8.0], [-2.0, 0.5]],
            )
            add(
                "complex_infinity", [3], "identity", [],
                [[0.0, 0.0], [1.0, 0.0], [0.0, -0.0]],
                [
                    {"index": 0, "value": "-inf", "component": "real"},
                    {"index": 0, "value": "+inf", "component": "imag"},
                    {"index": 2, "value": "+inf", "component": "real"},
                ],
            )
    return inputs


def _logcumsumexp_gradient_seed(
    torch: Any, output: Any, dtype_name: str, kind: str
) -> Any:
    dtype = getattr(torch, dtype_name)
    count = output.numel()
    is_complex = dtype_name in ("complex64", "complex128")
    if kind == "finite":
        constant = complex(2.0, -0.5) if is_complex else 2.0
    elif kind == "zero":
        constant = complex(0.0, 0.0) if is_complex else 0.0
    elif kind == "negative_zero":
        constant = complex(-0.0, -0.0) if is_complex else -0.0
    elif kind == "infinity":
        constant = complex(float("inf"), 0.0) if is_complex else float("inf")
    elif kind == "negative_infinity":
        constant = complex(float("-inf"), 0.0) if is_complex else float("-inf")
    elif kind == "nan":
        constant = complex(float("nan"), 0.0) if is_complex else float("nan")
    elif kind == "imaginary_infinity" and is_complex:
        constant = complex(1.0, float("inf"))
    elif kind in ("mixed", "strided"):
        values = (
            [
                complex(index % 5 - 2, (index % 3 - 1) * 0.5)
                for index in range(count)
            ]
            if is_complex
            else [index % 5 - 2 for index in range(count)]
        )
        seed = torch.tensor(values, dtype=dtype, device="cpu").reshape(output.shape)
        if kind == "strided":
            backing = torch.zeros(count * 2 + 1, dtype=dtype, device="cpu")
            backing[1::2] = seed.reshape(-1)
            seed = backing[1::2].reshape(output.shape)
        return seed
    else:
        raise ValueError(f"unknown logcumsumexp gradient seed kind: {kind}")
    return torch.tensor([constant] * count, dtype=dtype, device="cpu").reshape(output.shape)


def _logcumsumexp_gradient_axis_error(error: Exception) -> str | None:
    if isinstance(error, IndexError):
        return "invalid_axis"
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        if "dimension out of range" in message or "dim out of range" in message:
            return "invalid_axis"
    return None


def _observe_logcumsumexp_gradient_request(
    torch: Any, source: Any, dtype_name: str, request: dict[str, Any], label: str
) -> dict[str, Any]:
    request_id = request["id"]
    axis = request["axis"]
    axis_limit = max(source.ndim, 1)
    invalid_axis = axis < -axis_limit or axis >= axis_limit
    observations: dict[str, Any] = {}
    for seed_kind in request["seed_kinds"]:
        key = f"{request_id}_seed_{seed_kind}"
        leaf = source.detach().requires_grad_(True)
        try:
            output = torch.logcumsumexp(leaf, dim=axis)
        except (IndexError, RuntimeError) as error:
            code = _logcumsumexp_gradient_axis_error(error)
            if invalid_axis and code == "invalid_axis":
                observations[key] = {"error": code}
                continue
            raise ValueError(
                f"unexpected logcumsumexp gradient rejection for {label}.{seed_kind}: {error}"
            ) from error
        if invalid_axis:
            raise ValueError(f"PyTorch accepted an invalid logcumsumexp axis for {label}")
        seed = _logcumsumexp_gradient_seed(torch, output, dtype_name, seed_kind)
        gradient, = torch.autograd.grad(output, leaf, grad_outputs=seed)
        observations[key] = {
            "output": _observe_logsumexp_gradient_tensor(output),
            "seed": _observe_logsumexp_gradient_tensor(seed),
            "gradient": _observe_logsumexp_gradient_tensor(gradient),
        }

    if not invalid_axis:
        invalid_leaf = source.detach().requires_grad_(True)
        invalid_output = torch.logcumsumexp(invalid_leaf, dim=axis)
        invalid_seed = torch.zeros(
            source.numel() + 1, dtype=invalid_output.dtype, device="cpu"
        )
        try:
            torch.autograd.grad(invalid_output, invalid_leaf, grad_outputs=invalid_seed)
        except RuntimeError as error:
            message = str(error).lower()
            if "shape" not in message and "size" not in message:
                raise
            observations[f"{request_id}_invalid_seed"] = {
                "error": "seed_shape_mismatch"
            }
        else:
            raise ValueError(f"PyTorch accepted a mismatched logcumsumexp seed for {label}")
    return observations


def logcumsumexp_gradient(torch: Any) -> dict[str, Any]:
    """Observe actual CPU logcumsumexp forwards and seeded first-order VJPs."""
    inputs = _logcumsumexp_gradient_raw_inputs()
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for dtype_name, profiles in inputs.items():
        expected[dtype_name] = {}
        for profile, spec in profiles.items():
            source = _logcumsumexp_source(torch, spec)
            observations: dict[str, Any] = {}
            for request in spec["requests"]:
                observations.update(
                    _observe_logcumsumexp_gradient_request(
                        torch,
                        source,
                        dtype_name,
                        request,
                        f"{dtype_name}.{profile}.{request['id']}",
                    )
                )
            expected[dtype_name][profile] = observations
    return _case(
        "autograd.logcumsumexp",
        "api:autograd",
        inputs,
        expected,
        tolerance=1e-6,
    )




CASE_BUILDERS: tuple[Callable[[Any], dict[str, Any]], ...] = (
    transpose_alias, empty_shape, scalar_metadata, elementwise_shapes, broadcast_same_rank,
    broadcast_leading_dims, matmul_2d, mean_gradient, mse_mean,
    no_grad_exception, seeded_vjp, retained_vjp, independent_graphs, chunk_seed,
    slice_overlap, index_select_gradient, mask_select_gradient, axis_mean, reduction_scalars,
    axis_options, broadcast_multiply, broadcast_divide,
    mse_broadcast_mean, mse_broadcast_sum, mse_broadcast_none, matmul_ranks, singleton_views,
    unary_math, activation_math, reduction_edges, float64_reductions, product_gradient, extrema_gradient,
    cumsum_gradient, cumprod_gradient,
    integer_scans, integer_reductions, extrema, mean_dtype, sum_dtype, prod_dtype, cumsum_dtype, cumprod_dtype,
    integer_output_dtype, complex_scans, complex_scan_dtype, complex_mean_dtype, complex_sum_dtype,
    complex_prod_dtype, indexed_extrema, paired_extrema, paired_extrema_gradient, aminmax,
    logsumexp, logsumexp_gradient, logcumsumexp, logcumsumexp_gradient, cumulative_extrema,
    cumulative_extrema_gradient,
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
