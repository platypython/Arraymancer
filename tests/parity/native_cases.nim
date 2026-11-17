# Copyright 2026 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Observes fixture-driven CPU operations without replacing native semantics.
import std/[json, math]
import ../../src/arraymancer

proc getTensor(desc: JsonNode): Tensor[float64] =
  let shapeNode = desc["shape"]
  if shapeNode.len > LASER_MAXRANK: raise newException(ValueError, "input rank exceeds LASER_MAXRANK")
  var shape = newSeq[int](shapeNode.len)
  for i in 0 ..< shapeNode.len: shape[i] = shapeNode[i].getInt
  result = newTensorUninit[float64](toMetadata(shape))
  let values = desc["values"]
  if values.len != result.size: raise newException(ValueError, "input value count mismatch")
  var raw = result.unsafe_raw_offset
  for i in 0 ..< values.len: raw[i] = values[i].getFloat

proc floatNode(value: float64): JsonNode =
  case classify(value)
  of fcNan: result = %"nan"
  of fcInf: result = %"+inf"
  of fcNegInf: result = %"-inf"
  else: result = %value
proc tensorObservation[T](t: Tensor[T], includeValues = true): JsonNode =
  result = newJObject()
  var shape = newJArray()
  for dim in t.shape: shape.add newJInt(dim)
  result["shape"] = shape
  result["numel"] = newJInt(t.size.int)
  when T is float64: result["dtype"] = %"float64"
  elif T is float32: result["dtype"] = %"float32"
  elif T is bool: result["dtype"] = %"bool"
  else: {.error: "unsupported CPU oracle observation dtype".}
  result["device"] = %"cpu"
  if includeValues:
    var values = newJArray()
    for _, value in enumerate(t):
      when T is bool: values.add %value
      else: values.add floatNode(value.float64)
    result["values"] = values

proc runCase*(caseData: JsonNode): JsonNode =
  let id = caseData["id"].getStr
  let inputs = caseData["inputs"]
  result = newJObject()
  case id
  of "tensor.transpose_alias":
    var base = getTensor(inputs["a"])
    var view = base.transpose
    let mutation = inputs["mutation"]
    let index = mutation["index"]
    if index.len != 2: raise newException(ValueError, "transpose mutation index must have rank 2")
    view[index[0].getInt, index[1].getInt] = mutation["value"].getFloat
    result["base"] = tensorObservation(base)
    result["view"] = tensorObservation(view)
  of "tensor.empty_shape":
    var shape = newSeq[int](inputs["shape"].len)
    for i in 0 ..< inputs["shape"].len: shape[i] = inputs["shape"][i].getInt
    if shape.len > LASER_MAXRANK: raise newException(ValueError, "input rank exceeds LASER_MAXRANK")
    result["result"] = tensorObservation(newTensorUninit[float64](toMetadata(shape)))
  of "tensor.elementwise_shapes":
    for name, desc in inputs:
      var a = getTensor(desc)
      if name == "strided": a = a.transpose[_, 1..^1]
      elif name == "empty_transposed": a = a.transpose(0, 2)
      result[name] = %*{
        "exp": tensorObservation(a.exp), "sqrt": tensorObservation(a.sqrt),
        "pow": tensorObservation(a ^. 2.0), "add": tensorObservation(a +. 2.0),
        "scale": tensorObservation(a * 2.0), "divide": tensorObservation(a / 2.0),
        "reverse_divide": tensorObservation(2.0 /. a),
        "cast": tensorObservation(a.asType(float32)),
        "greater": tensorObservation(a >. 2.0), "isnan": tensorObservation(a.isNaN)}
  of "autograd.unary_math":
    for name, spec in inputs:
      result[name] = newJObject()
      var value = getTensor(spec["a"])
      if name == "strided": value = value.transpose[_, 1..^1]
      elif name == "empty": value = value.transpose(0, 2)
      for operation in ["exp", "ln", "sqrt"]:
        let ctx = newContext Tensor[float64]
        let x = ctx.variable(value, requires_grad = true)
        let output = case operation
                     of "exp": x.exp
                     of "ln": x.ln
                     else: x.sqrt
        output.backprop(getTensor(spec["seed"]))
        result[name][operation] = %*{"output": tensorObservation(output.value),
                                    "grad": tensorObservation(x.grad)}
  of "autograd.activation_math":
    for name, spec in inputs:
      result[name] = newJObject()
      var value = getTensor(spec["a"])
      if name == "strided": value = value.transpose[_, 1..^1]
      elif name == "empty": value = value.transpose(0, 2)
      if spec.hasKey("transform"):
        if spec["transform"].getStr == "ln": value = value.ln
        else: value = 1.0 /. value
      var seed = getTensor(spec["seed"])
      if spec.hasKey("seed_transform"): seed = seed.ln
      for operation in ["relu", "sigmoid", "tanh"]:
        let ctx = newContext Tensor[float64]
        let x = ctx.variable(value, requires_grad = true)
        let output = case operation
                     of "relu": x.relu
                     of "sigmoid": x.sigmoid
                     else: x.tanh
        output.backprop(seed)
        let negativeZero = output.value.map_inline(x.classify == fcNegZero)
        result[name][operation] = %*{"output": tensorObservation(output.value),
          "grad": tensorObservation(x.grad), "negative_zero": tensorObservation(negativeZero)}
  of "tensor.scalar_metadata":
    let scalar = newTensorUninit[float64]([])
    result["result"] = tensorObservation(scalar, false)
  of "tensor.broadcast_same_rank", "tensor.broadcast_leading_dims":
    let a = getTensor(inputs["a"])
    let b = getTensor(inputs["b"])
    result["result"] = tensorObservation(a +. b)
  of "tensor.matmul_2d":
    let a = getTensor(inputs["a"])
    let b = getTensor(inputs["b"])
    result["result"] = tensorObservation(a * b)
  of "autograd.mean_gradient":
    let ctx = newContext Tensor[float64]
    let x = ctx.variable(getTensor(inputs["x"]), requires_grad = true)
    let y = x.mean
    y.backprop
    result["grad"] = tensorObservation(x.grad)
  of "loss.mse_mean":
    let ctx = newContext Tensor[float64]
    let prediction = ctx.variable(getTensor(inputs["prediction"]), requires_grad = true)
    let target = getTensor(inputs["target"])
    let loss = prediction.mse_loss(target)
    loss.backprop
    result["loss"] = tensorObservation(loss.value)
    result["grad"] = tensorObservation(prediction.grad)
  of "loss.mse_mean_broadcast", "loss.mse_sum_broadcast", "loss.mse_none_broadcast":
    let ctx = newContext Tensor[float64]
    let prediction = ctx.variable(getTensor(inputs["prediction"]), requires_grad = true)
    let target = ctx.variable(getTensor(inputs["target"]), requires_grad = true)
    let reduction = if id == "loss.mse_mean_broadcast": mseMean
                    elif id == "loss.mse_sum_broadcast": mseSum else: mseNone
    let loss = prediction.mse_loss(target, reduction)
    loss.backprop(getTensor(inputs["seed"]))
    result["loss"] = tensorObservation(loss.value)
    result["grad_prediction"] = tensorObservation(prediction.grad)
    result["grad_target"] = tensorObservation(target.grad)
  of "autograd.matmul_ranks":
    for name, operands in inputs:
      let ctx = newContext Tensor[float64]
      let a = ctx.variable(getTensor(operands["a"]), requires_grad = true)
      let b = ctx.variable(getTensor(operands["b"]), requires_grad = true)
      let output = a * b
      output.backprop(getTensor(operands["seed"]))
      result[name] = %*{"output": tensorObservation(output.value),
                        "grad_a": tensorObservation(a.grad), "grad_b": tensorObservation(b.grad)}
  of "autograd.singleton_views":
    let ctx = newContext Tensor[float64]
    let x = ctx.variable(getTensor(inputs["x"]).transpose(0, 2), requires_grad = true)
    let output = x.squeeze(-2).unsqueeze(-1).squeeze(-1)
    output.backprop(getTensor(inputs["seed"]))
    let noop = x.squeeze(-1)
    noop.backprop(noop.value.ones_like)
    let scalar = ctx.variable(getTensor(inputs["scalar"]), requires_grad = true)
    let scalarOutput = scalar.squeeze.unsqueeze(-1).squeeze(0).squeeze(-1)
    scalarOutput.backprop(newTensorWith[float64]([], 3.0))
    result["output"] = tensorObservation(output.value)
    result["noop"] = tensorObservation(noop.value)
    result["grad"] = tensorObservation(x.grad)
    result["scalar"] = tensorObservation(scalarOutput.value)
    result["scalar_grad"] = tensorObservation(scalar.grad)
    result["shares_storage"] = %(output.value.get_offset_ptr == x.value.get_offset_ptr)
  of "autograd.no_grad_exception":
    let ctx = newContext Tensor[float64]
    let x = ctx.variable(getTensor(inputs["x"]), requires_grad = true)
    try:
      ctx.no_grad_mode:
        raise newException(ValueError, "parity probe")
    except ValueError:
      discard
    result["tracking_after_exception"] = %x.is_grad_needed
  of "autograd.broadcast_multiply", "autograd.broadcast_divide":
    let ctx = newContext Tensor[float64]
    let a = ctx.variable(getTensor(inputs["a"]), requires_grad = true)
    let b = ctx.variable(getTensor(inputs["b"]), requires_grad = true)
    let output = if id == "autograd.broadcast_divide": a /. b else: a *. b
    output.backprop(getTensor(inputs["seed"]))
    result["output"] = tensorObservation(output.value)
    result["grad_a"] = tensorObservation(a.grad)
    result["grad_b"] = tensorObservation(b.grad)
  of "autograd.axis_options":
    let ctx = newContext Tensor[float64]
    let x = ctx.variable(getTensor(inputs["x"]).transpose, requires_grad = true)
    let total = x.sum(axis = -1, keepdim = false)
    let average = x.mean(axis = -1, keepdim = false)
    total.backprop(getTensor(inputs["first"]))
    average.backprop(getTensor(inputs["second"]))
    result["sum"] = tensorObservation(total.value)
    result["mean"] = tensorObservation(average.value)
    result["grad"] = tensorObservation(x.grad)
  of "autograd.seeded_vjp", "autograd.retained_vjp", "autograd.independent_graphs",
      "autograd.chunk_seed", "autograd.slice_overlap", "autograd.index_select",
      "autograd.mask_select", "autograd.axis_mean", "autograd.reduction_scalars":
    let ctx = newContext Tensor[float64]
    let source = getTensor(inputs["x"])
    let x = ctx.variable(if id == "autograd.mask_select": source.transpose else: source,
                         requires_grad = true)
    case id
    of "autograd.seeded_vjp":
      let output = x *. x + x
      output.backprop(getTensor(inputs["seed"]))
      result["output"] = tensorObservation(output.value)
    of "autograd.retained_vjp":
      let squared = x *. x
      let output = squared *. squared
      output.backprop(getTensor(inputs["first"]), retain_graph = true)
      output.backprop(getTensor(inputs["second"]))
    of "autograd.independent_graphs":
      let unused = x /. ctx.variable(x.value.zeros_like)
      let earlier = x *. x
      let later = x + x
      earlier.backprop
      later.backprop
      discard unused
    of "autograd.chunk_seed":
      let parts = x.chunk(2, axis = 0)
      parts[1].backprop(getTensor(inputs["first"]), retain_graph = true)
      parts[0].backprop(getTensor(inputs["second"]))
    of "autograd.slice_overlap":
      let output = x[1..3].sum + x[4..0|-2].sum
      output.backprop
      result["output"] = tensorObservation(output.value)
    of "autograd.index_select":
      let output = x[_, [2, 0, 2]]
      output.backprop(getTensor(inputs["seed"]))
      result["output"] = tensorObservation(output.value)
    of "autograd.mask_select":
      let output = x[[[true, false], [false, true], [true, false]]]
      output.backprop(getTensor(inputs["seed"]))
      result["output"] = tensorObservation(output.value)
    of "autograd.axis_mean":
      let output = x.mean(axis = 1)
      output.backprop(output.value.ones_like * 6.0)
      result["output"] = tensorObservation(output.value)
    of "autograd.reduction_scalars":
      let average = x.mean
      let total = x.sum
      total.backprop([3.0].toTensor.reshape(newSeq[int]()))
      average.backprop([2.0].toTensor.reshape(newSeq[int]()))
      result["mean"] = tensorObservation(average.value)
      result["sum"] = tensorObservation(total.value)
    else: discard
    result["grad"] = tensorObservation(x.grad)
  else:
    raise newException(ValueError, "unknown parity case: " & id)
