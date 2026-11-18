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
  elif T is int64: result["dtype"] = %"int64"
  else: {.error: "unsupported CPU oracle observation dtype".}
  result["device"] = %"cpu"
  if includeValues:
    var values = newJArray()
    for _, value in enumerate(t):
      when T is bool: values.add %value
      elif T is int64: values.add %value
      else: values.add floatNode(value.float64)
    result["values"] = values

proc getIntegerTensor[T](desc: JsonNode): Tensor[T] =
  ## Signed words preserve unsigned bit patterns without range-checked casts.
  let shapeNode = desc["shape"]
  if shapeNode.len > LASER_MAXRANK: raise newException(ValueError, "input rank exceeds LASER_MAXRANK")
  var shape = newSeq[int](shapeNode.len)
  for i in 0 ..< shapeNode.len: shape[i] = shapeNode[i].getInt
  result = newTensorUninit[T](toMetadata(shape))
  let words = desc["words"]
  if words.len != result.size: raise newException(ValueError, "integer scan word count mismatch")
  var raw = result.unsafe_raw_offset
  for i in 0 ..< words.len:
    let word = words[i].getBiggestInt
    when T is bool: raw[i] = word != 0
    else: raw[i] = cast[T](word)

proc integerScanCase[T](spec: JsonNode): JsonNode =
  var source = getIntegerTensor[T](spec)
  case spec["layout"].getStr
  of "transpose": source = source.transpose
  of "offset": source = source.transpose[_, 1..^1]
  of "reverse": source = source[^1..0|-1]
  of "broadcast": source = source.broadcast(4, 3)
  else: discard
  result = newJObject()
  for operation in ["cumsum", "cumprod"]:
    var observations = newJObject()
    for axisNode in spec["axes"]:
      let axis = axisNode.getInt
      let output = if operation == "cumsum": source.cumsum(axis) else: source.cumprod(axis)
      observations[$axis] = tensorObservation(output)
    result[operation] = observations

proc integerScans(spec: JsonNode): JsonNode =
  result = newJObject()
  for name, input in spec:
    result[name] = case input["dtype"].getStr
      of "bool": integerScanCase[bool](input)
      of "int8": integerScanCase[int8](input)
      of "int16": integerScanCase[int16](input)
      of "int32": integerScanCase[int32](input)
      of "int64": integerScanCase[int64](input)
      of "uint8": integerScanCase[uint8](input)
      of "uint16": integerScanCase[uint16](input)
      of "uint32": integerScanCase[uint32](input)
      of "uint64": integerScanCase[uint64](input)
      else: raise newException(ValueError, "unsupported integer scan dtype: " & input["dtype"].getStr)

proc integerReductionCase[T](spec: JsonNode): JsonNode =
  var source = getIntegerTensor[T](spec)
  case spec["layout"].getStr
  of "transpose": source = source.transpose
  of "offset": source = source.transpose[_, 1..^1]
  of "reverse": source = source[^1..0|-1]
  of "broadcast": source = source.broadcast(4, 3)
  else: discard
  result = newJObject()
  for operation in ["sum", "prod"]:
    var observations = newJObject()
    for axisNode in spec["axes"]:
      let key = if axisNode.kind == JNull: "all" else: $axisNode.getInt
      if axisNode.kind == JNull:
        let output = if operation == "sum":
                       newTensorWith[int64]([], source.sum)
                     else:
                       newTensorWith[int64]([], source.product)
        observations[key] = tensorObservation(output)
      else:
        let axis = axisNode.getInt
        let output = if operation == "sum":
                       source.sum(axis)
                     else:
                       source.product(axis)
        observations[key] = tensorObservation(output)
    result[operation] = observations

proc integerReductions(spec: JsonNode): JsonNode =
  result = newJObject()
  for name, input in spec:
    result[name] = case input["dtype"].getStr
      of "bool": integerReductionCase[bool](input)
      of "int8": integerReductionCase[int8](input)
      of "int16": integerReductionCase[int16](input)
      of "int32": integerReductionCase[int32](input)
      of "int64": integerReductionCase[int64](input)
      of "uint8": integerReductionCase[uint8](input)
      of "uint16": integerReductionCase[uint16](input)
      of "uint32": integerReductionCase[uint32](input)
      of "uint64": integerReductionCase[uint64](input)
      else: raise newException(ValueError, "unsupported integer reduction dtype: " & input["dtype"].getStr)

proc realDtypeFloat32Tensor(desc: JsonNode): Tensor[float32] =
  let shapeNode = desc["shape"]
  if shapeNode.len > LASER_MAXRANK: raise newException(ValueError, "input rank exceeds LASER_MAXRANK")
  var shape = newSeq[int](shapeNode.len)
  for i in 0 ..< shapeNode.len: shape[i] = shapeNode[i].getInt
  result = newTensorUninit[float32](toMetadata(shape))
  let values = desc["values"]
  if values.len != result.size: raise newException(ValueError, "real dtype input value count mismatch")
  var raw = result.unsafe_raw_offset
  for i in 0 ..< values.len: raw[i] = values[i].getFloat.float32

proc realDtypeLayout[T](source: Tensor[T], layout: string): Tensor[T] =
  case layout
  of "transpose": result = source.transpose
  of "offset": result = source.transpose[_, 1..^1]
  of "reverse": result = source[^1..0|-1]
  of "broadcast": result = source.broadcast(4, 3)
  else: result = source

proc realDtypeScalar[T, U](source: Tensor[T], dtype: typedesc[U], operation: string): U =
  case operation
  of "sum": result = source.sum(dtype = dtype)
  of "mean": result = source.mean(dtype = dtype)
  else: raise newException(ValueError, "unsupported real dtype reduction: " & operation)

proc realDtypeAxis[T, U](source: Tensor[T], axis: int, dtype: typedesc[U],
                         operation: string): Tensor[U] =
  case operation
  of "sum": result = source.sum(axis = axis, dtype = dtype)
  of "mean": result = source.mean(axis = axis, dtype = dtype)
  else: raise newException(ValueError, "unsupported real dtype reduction: " & operation)

proc realDtypeOutput[T, U](source: Tensor[T], axes: JsonNode,
                           dtype: typedesc[U], operation: string): JsonNode =
  result = newJObject()
  result["all"] = tensorObservation(newTensorWith[U](
    [], realDtypeScalar(source, dtype, operation)))
  result["axes"] = newJObject()
  for axisNode in axes:
    let axis = axisNode.getInt
    result["axes"][$axis] = tensorObservation(realDtypeAxis(source, axis, dtype, operation))

proc meanDtypeDefaultRejections[T](source: Tensor[T], axes: JsonNode): JsonNode =
  result = newJObject()
  try:
    discard source.mean
    result["all"] = %*{"returned": true}
  except ValueError:
    result["all"] = %*{"error": "default_dtype_rejected"}
  result["axes"] = newJObject()
  for axisNode in axes:
    let axis = axisNode.getInt
    try:
      discard source.mean(axis)
      result["axes"][$axis] = %*{"returned": true}
    except ValueError:
      result["axes"][$axis] = %*{"error": "default_dtype_rejected"}

proc realDtypeCase[T](spec: JsonNode, operation: string): JsonNode =
  var source =
    when T is float32: realDtypeFloat32Tensor(spec)
    elif T is float64: getTensor(spec)
    else: getIntegerTensor[T](spec)
  source = realDtypeLayout(source, spec["layout"].getStr)
  let outputs32 = realDtypeOutput[T, float32](source, spec["axes"], float32, operation)
  let outputs64 = realDtypeOutput[T, float64](source, spec["axes"], float64, operation)
  result = newJObject()
  result["all"] = newJObject()
  result["all"]["float32"] = outputs32["all"]
  result["all"]["float64"] = outputs64["all"]
  result["axes"] = newJObject()
  for axisNode in spec["axes"]:
    let key = $axisNode.getInt
    result["axes"][key] = newJObject()
    result["axes"][key]["float32"] = outputs32["axes"][key]
    result["axes"][key]["float64"] = outputs64["axes"][key]
  when T is float32 or T is float64:
    discard
  else:
    if operation == "mean":
      result["default"] = meanDtypeDefaultRejections(source, spec["axes"])

proc realDtype(inputs: JsonNode, operation: string): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = case spec["dtype"].getStr
      of "bool": realDtypeCase[bool](spec, operation)
      of "int8": realDtypeCase[int8](spec, operation)
      of "int16": realDtypeCase[int16](spec, operation)
      of "int32": realDtypeCase[int32](spec, operation)
      of "int64": realDtypeCase[int64](spec, operation)
      of "uint8": realDtypeCase[uint8](spec, operation)
      of "uint16": realDtypeCase[uint16](spec, operation)
      of "uint32": realDtypeCase[uint32](spec, operation)
      of "uint64": realDtypeCase[uint64](spec, operation)
      of "float32": realDtypeCase[float32](spec, operation)
      of "float64": realDtypeCase[float64](spec, operation)
      else: raise newException(ValueError, "unsupported " & operation & " dtype input: " & spec["dtype"].getStr)


proc float64Layout(source: Tensor[float64], layout: string): Tensor[float64] =
  case layout
  of "transpose": result = source.transpose
  of "offset": result = source.transpose[_, 1..^1]
  of "reverse": result = source[^1..0|-1]
  of "broadcast": result = source.broadcast(4, 3)
  else: result = source

proc float64Specials(source: var Tensor[float64], specials: JsonNode) =
  for special in specials:
    let value = case special["value"].getStr
                of "nan": NaN
                of "-inf": NegInf
                else: Inf
    source.unsafe_raw_offset[special["index"].getInt] = value

proc float64ReductionObservations(specs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in specs:
    let source = float64Layout(getTensor(spec), spec["layout"].getStr)
    result[name] = newJObject()
    for operation in ["sum", "prod", "mean"]:
      let whole = case operation
                  of "sum": source.sum
                  of "prod": source.product
                  else: source.mean
      var byAxis = newJObject()
      byAxis["all"] = tensorObservation(newTensorWith[float64]([], whole))
      for axisNode in spec["axes"]:
        if axisNode.kind != JNull:
          let axis = axisNode.getInt
          let output = case operation
                       of "sum": source.sum(axis)
                       of "prod": source.product(axis)
                       else: source.mean(axis)
          byAxis[$axis] = tensorObservation(output)
      result[name][operation] = byAxis

proc float64ScanObservations(specs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in specs:
    let source = float64Layout(getTensor(spec), spec["layout"].getStr)
    result[name] = newJObject()
    for operation in ["cumsum", "cumprod"]:
      var byAxis = newJObject()
      for axisNode in spec["axes"]:
        let axis = axisNode.getInt
        let output = if operation == "cumsum": source.cumsum(axis) else: source.cumprod(axis)
        byAxis[$axis] = tensorObservation(output)
      result[name][operation] = byAxis

proc float64ReductionVjps(specs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in specs:
    var source = getTensor(spec["x"])
    float64Specials(source, spec["nonfinite"])
    source = float64Layout(source, spec["layout"].getStr)
    result[name] = newJArray()
    for operation in spec["operations"]:
      let context = newContext Tensor[float64]
      let x = context.variable(source, requires_grad = true)
      let axisNode = operation["axis"]
      let kind = operation["operation"].getStr
      let output =
        if axisNode.kind == JNull:
          case kind
          of "sum": x.sum
          of "prod": x.product
          else: x.mean
        else:
          let axis = axisNode.getInt
          let keepdim = operation["keepdim"].getBool
          case kind
          of "sum": x.sum(axis = axis, keepdim = keepdim)
          of "prod": x.product(axis, keepdim = keepdim)
          else: x.mean(axis, keepdim = keepdim)
      output.backprop(getTensor(operation["seed"]))
      result[name].add %*{"output": tensorObservation(output.value),
                         "grad": tensorObservation(x.grad)}

proc float64ScanVjps(specs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in specs:
    var source = getTensor(spec["x"])
    float64Specials(source, spec["nonfinite"])
    source = float64Layout(source, spec["layout"].getStr)
    result[name] = newJArray()
    for operation in spec["operations"]:
      let context = newContext Tensor[float64]
      let x = context.variable(source, requires_grad = true)
      let axis = operation["axis"].getInt
      let output = if operation["operation"].getStr == "cumsum":
                     x.cumsum(axis)
                   else:
                     x.cumprod(axis)
      output.backprop(getTensor(operation["seed"]))
      result[name].add %*{"output": tensorObservation(output.value),
                         "grad": tensorObservation(x.grad)}

proc float64Reductions(inputs: JsonNode): JsonNode =
  result = newJObject()
  result["reductions"] = float64ReductionObservations(inputs["reductions"])
  result["scans"] = float64ScanObservations(inputs["scans"])
  result["reduction_vjps"] = float64ReductionVjps(inputs["reduction_vjps"])
  result["scan_vjps"] = float64ScanVjps(inputs["scan_vjps"])


proc runCase*(caseData: JsonNode): JsonNode =
  let id = caseData["id"].getStr
  let inputs = caseData["inputs"]
  result = newJObject()
  case id
  of "tensor.float64_reductions":
    result = float64Reductions(inputs)
  of "tensor.integer_scans":
    result = integerScans(inputs)
  of "tensor.integer_reductions":
    result = integerReductions(inputs)
  of "tensor.mean_dtype":
    result = realDtype(inputs, "mean")
  of "tensor.sum_dtype":
    result = realDtype(inputs, "sum")
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
  of "autograd.product_gradient":
    for name, spec in inputs:
      var source = getTensor(spec["x"])
      for special in spec["nonfinite"]:
        source.unsafe_raw_offset[special["index"].getInt] =
          (if special["value"].getStr == "nan": NaN else: Inf)
      case spec["layout"].getStr
      of "transpose": source = source.transpose
      of "offset": source = source.transpose[_, 1..^1]
      of "reverse": source = source[^1..0|-1]
      of "broadcast": source = source.broadcast(4, 3)
      else: discard
      result[name] = newJArray()
      for operation in spec["operations"]:
        let context = newContext Tensor[float64]
        let x = context.variable(source, requires_grad = true)
        let output =
          if operation["axis"].kind == JNull: x.product
          else: x.product(operation["axis"].getInt, keepdim = operation["keepdim"].getBool)
        output.backprop(getTensor(operation["seed"]))
        result[name].add %*{"output": tensorObservation(output.value),
                           "grad": tensorObservation(x.grad)}
  of "autograd.cumsum_gradient", "autograd.cumprod_gradient":
    for name, spec in inputs:
      var source = getTensor(spec["x"])
      for special in spec["nonfinite"]:
        let value = case special["value"].getStr
                    of "nan": NaN
                    of "-inf": NegInf
                    else: Inf
        source.unsafe_raw_offset[special["index"].getInt] = value
      case spec["layout"].getStr
      of "transpose": source = source.transpose
      of "offset": source = source.transpose[_, 1..^1]
      of "reverse": source = source[^1..0|-1]
      of "broadcast": source = source.broadcast(4, 3)
      else: discard
      result[name] = newJArray()
      for operation in spec["operations"]:
        let context = newContext Tensor[float64]
        let x = context.variable(source, requires_grad = true)
        let output = if id == "autograd.cumsum_gradient":
                       x.cumsum(operation["axis"].getInt)
                     else:
                       x.cumprod(operation["axis"].getInt)
        var seed = getTensor(operation["seed"])
        for special in operation["seed_nonfinite"]:
          seed.unsafe_raw_offset[special["index"].getInt] =
            (case special["value"].getStr
             of "nan": NaN
             of "-inf": NegInf
             else: Inf)
        output.backprop(seed)
        result[name].add %*{"output": tensorObservation(output.value),
                           "grad": tensorObservation(x.grad)}
  of "tensor.reduction_edges":
    for name, spec in inputs:
      var a = getTensor(spec["a"])
      if name == "strided": a = a.transpose[_, 1..^1]
      result[name] = %*{
        "sum": tensorObservation(newTensorWith[float64]([], a.sum)),
        "prod": tensorObservation(newTensorWith[float64]([], a.product)),
        "mean": tensorObservation(newTensorWith[float64]([], a.mean))}
      for axisNode in spec["axes"]:
        let axis = axisNode.getInt
        result[name]["axis_" & $axis] = %*{
          "sum": tensorObservation(a.sum(axis)),
          "prod": tensorObservation(a.product(axis)),
          "mean": tensorObservation(a.mean(axis))}
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
