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
import std/[complex, json, math, strutils]
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
  elif T is float: result["dtype"] = %"float64"
  elif T is float32: result["dtype"] = %"float32"
  elif T is bool: result["dtype"] = %"bool"
  elif T is int8: result["dtype"] = %"int8"
  elif T is int16: result["dtype"] = %"int16"
  elif T is int32: result["dtype"] = %"int32"
  elif T is int64: result["dtype"] = %"int64"
  elif T is int: result["dtype"] = %"int64"
  elif T is uint8: result["dtype"] = %"uint8"
  elif T is uint16: result["dtype"] = %"uint16"
  elif T is uint32: result["dtype"] = %"uint32"
  elif T is uint64:
    result["dtype"] = %"uint64"
    result["word_encoding"] = %"signed_twos_complement"
  else: {.error: "unsupported CPU oracle observation dtype".}
  result["device"] = %"cpu"
  if includeValues:
    var values = newJArray()
    for _, value in enumerate(t):
      when T is bool: values.add %value
      elif T is int64: values.add %value
      elif T is uint64: values.add %(cast[int64](value))
      elif T is SomeInteger: values.add %(value.int64)
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
  of "reverse_both": result = source[^1..0|-1, ^1..0|-1]
  of "offset_columns":
    # Keep a skipped prefix and padding between rows in the oracle's view.
    var backing = newTensorUninit[T](2, 6)
    for row in 0 ..< 2:
      for column in 0 ..< 6:
        backing.unsafe_raw_offset[row * 6 + column] =
          source.atContiguousIndex(row * 3 + column mod 3)
    result = backing[_, 1..4]
  of "broadcast": result = source.broadcast(4, 3)
  else: result = source

proc realDtypeScalar[T, U](source: Tensor[T], dtype: typedesc[U], operation: string): U =
  case operation
  of "sum": result = source.sum(dtype = dtype)
  of "mean": result = source.mean(dtype = dtype)
  of "prod": result = source.product(dtype = dtype)
  else: raise newException(ValueError, "unsupported real dtype reduction: " & operation)

proc realDtypeAxis[T, U](source: Tensor[T], axis: int, dtype: typedesc[U],
                         operation: string): Tensor[U] =
  case operation
  of "sum": result = source.sum(axis = axis, dtype = dtype)
  of "mean": result = source.mean(axis = axis, dtype = dtype)
  of "prod": result = source.product(axis = axis, dtype = dtype)
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

proc realDtypeSpecials[T](source: var Tensor[T], specials: JsonNode) =
  for special in specials:
    when T is SomeFloat:
      let index = special["index"].getInt
      let value = case special["value"].getStr
                  of "nan": NaN
                  of "+inf": Inf
                  else: NegInf
      source.unsafe_raw_offset[index] = T(value)
    else:
      raise newException(ValueError, "nonfinite real dtype source must be floating point")

proc realDtypeScanAxis[T, U](source: Tensor[T], axis: int, dtype: typedesc[U],
                            operation: string, reverse: bool): Tensor[U] =
  case operation
  of "cumsum": result = source.cumsum(axis = axis, dtype = dtype, reverse = reverse)
  of "cumprod": result = source.cumprod(axis = axis, dtype = dtype, reverse = reverse)
  else: raise newException(ValueError, "unsupported real dtype scan: " & operation)

proc realDtypeScanDirections[T, U](source: Tensor[T], axis: int, dtype: typedesc[U],
                                  operation: string): JsonNode =
  result = newJObject()
  result["forward"] = tensorObservation(realDtypeScanAxis(source, axis, dtype, operation, false))
  result["reverse"] = tensorObservation(realDtypeScanAxis(source, axis, dtype, operation, true))

proc realDtypeScansCase[T](spec: JsonNode, operation: string): JsonNode =
  var source =
    when T is float32: realDtypeFloat32Tensor(spec)
    elif T is float64: getTensor(spec)
    else: getIntegerTensor[T](spec)
  if spec.hasKey("nonfinite"):
    realDtypeSpecials(source, spec["nonfinite"])
  source = realDtypeLayout(source, spec["layout"].getStr)
  result = newJObject()
  result["axes"] = newJObject()
  for axisNode in spec["axes"]:
    let axis = axisNode.getInt
    let key = $axis
    result["axes"][key] = newJObject()
    result["axes"][key]["float32"] =
      realDtypeScanDirections[T, float32](source, axis, float32, operation)
    result["axes"][key]["float64"] =
      realDtypeScanDirections[T, float64](source, axis, float64, operation)

proc realDtypeScans(inputs: JsonNode, operation: string): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = case spec["dtype"].getStr
      of "bool": realDtypeScansCase[bool](spec, operation)
      of "int8": realDtypeScansCase[int8](spec, operation)
      of "int16": realDtypeScansCase[int16](spec, operation)
      of "int32": realDtypeScansCase[int32](spec, operation)
      of "int64": realDtypeScansCase[int64](spec, operation)
      of "uint8": realDtypeScansCase[uint8](spec, operation)
      of "uint16": realDtypeScansCase[uint16](spec, operation)
      of "uint32": realDtypeScansCase[uint32](spec, operation)
      of "uint64": realDtypeScansCase[uint64](spec, operation)
      of "float32": realDtypeScansCase[float32](spec, operation)
      of "float64": realDtypeScansCase[float64](spec, operation)
      else: raise newException(ValueError, "unsupported " & operation & " dtype input: " & spec["dtype"].getStr)



proc integerOutputSource[T](spec: JsonNode): Tensor[T] =
  var source =
    when T is float32: realDtypeFloat32Tensor(spec)
    elif T is float64: getTensor(spec)
    else: getIntegerTensor[T](spec)
  if spec.hasKey("nonfinite"):
    realDtypeSpecials(source, spec["nonfinite"])
  result = realDtypeLayout(source, spec["layout"].getStr)

proc integerOutputUnsupported(): JsonNode =
  %*{"error": "unsupported_output_dtype"}

proc integerOutputReductionScalar[T, U](source: Tensor[T], dtype: typedesc[U],
                                         operation: string, reject: bool,
                                         label: string): JsonNode =
  if reject:
    try:
      discard realDtypeScalar(source, dtype, operation)
    except ValueError:
      return integerOutputUnsupported()
    raise newException(ValueError, label & " unexpectedly accepted unsupported output dtype")
  result = tensorObservation(newTensorWith[U]([], realDtypeScalar(source, dtype, operation)))

proc integerOutputReductionAxis[T, U](source: Tensor[T], axis: int,
                                       dtype: typedesc[U], operation: string,
                                       reject: bool, label: string): JsonNode =
  if reject:
    try:
      discard realDtypeAxis(source, axis, dtype, operation)
    except ValueError:
      return integerOutputUnsupported()
    raise newException(ValueError, label & " unexpectedly accepted unsupported output dtype")
  result = tensorObservation(realDtypeAxis(source, axis, dtype, operation))

proc integerOutputMeanError[T, U](source: Tensor[T], dtype: typedesc[U],
                                   label: string): JsonNode =
  try:
    discard source.mean(dtype = dtype)
  except ValueError:
    return %*{"error": "integer_mean_dtype_rejected"}
  raise newException(ValueError, label & " unexpectedly accepted an integer mean dtype")

proc integerOutputMeanAxisError[T, U](source: Tensor[T], axis: int,
                                       dtype: typedesc[U], label: string): JsonNode =
  try:
    discard source.mean(axis = axis, dtype = dtype)
  except ValueError:
    return %*{"error": "integer_mean_dtype_rejected"}
  raise newException(ValueError, label & " unexpectedly accepted an integer mean dtype")

proc integerOutputScan[T, U](source: Tensor[T], axis: int, dtype: typedesc[U],
                              operation: string, reverse, reject: bool,
                              errorCode, label: string): JsonNode =
  if reject:
    try:
      discard realDtypeScanAxis(source, axis, dtype, operation, reverse)
    except ValueError as error:
      if errorCode == "invalid_axis":
        let message = error.msg.toLowerAscii
        if not (message.contains("axis") or message.contains("dimension") or
                message.contains("index") or message.contains("range")):
          raise
      return %*{"error": errorCode}
    raise newException(ValueError, label & " unexpectedly accepted a rejected scan")
  result = tensorObservation(realDtypeScanAxis(source, axis, dtype, operation, reverse))

proc integerOutputDtypeCase[T, U](source: Tensor[T], spec: JsonNode,
                                  dtype: typedesc[U], dtypeName: string): JsonNode =
  let nonempty = source.size > 0
  let rejectsReduction = dtypeName in ["uint16", "uint32", "uint64"] and nonempty
  let rejectsScan = dtypeName in ["bool", "uint16", "uint32", "uint64"] and
    nonempty and source.shape.len > 0
  result = newJObject()

  var reductions = newJObject()
  for operation in ["sum", "prod"]:
    var byOperation = newJObject()
    byOperation["all"] = integerOutputReductionScalar(
      source, dtype, operation, rejectsReduction, dtypeName & "." & operation & ".all")
    var axes = newJObject()
    for axisNode in spec["axes"]:
      let axis = axisNode.getInt
      axes[$axis] = integerOutputReductionAxis(
        source, axis, dtype, operation, rejectsReduction,
        dtypeName & "." & operation & "." & $axis)
    byOperation["axes"] = axes
    reductions[operation] = byOperation
  result["reductions"] = reductions

  var scans = newJObject()
  for operation in ["cumsum", "cumprod"]:
    var byOperation = newJObject()
    var axes = newJObject()
    for axisNode in spec["axes"]:
      let axis = axisNode.getInt
      var byDtype = newJObject()
      for reverse in [false, true]:
        let direction = if reverse: "reverse" else: "forward"
        byDtype[direction] = integerOutputScan(source, axis, dtype, operation,
          reverse, rejectsScan, "unsupported_output_dtype",
          dtypeName & "." & operation & "." & $axis & "." & direction)
      axes[$axis] = byDtype
    byOperation["axes"] = axes
    scans[operation] = byOperation
  result["scans"] = scans

  var mean = newJObject()
  var meanAll = newJObject()
  meanAll[dtypeName] = integerOutputMeanError(source, dtype, dtypeName & ".mean.all")
  mean["all"] = meanAll
  var meanAxes = newJObject()
  for axisNode in spec["axes"]:
    let axis = axisNode.getInt
    var byDtype = newJObject()
    byDtype[dtypeName] = integerOutputMeanAxisError(
      source, axis, dtype, dtypeName & ".mean." & $axis)
    meanAxes[$axis] = byDtype
  mean["axes"] = meanAxes
  result["mean"] = mean

  if spec.hasKey("invalid_axes"):
    var invalid = newJObject()
    for operation in ["cumsum", "cumprod"]:
      var byOperation = newJObject()
      for axisNode in spec["invalid_axes"]:
        let axis = axisNode.getInt
        byOperation[$axis] = integerOutputScan(source, axis, dtype, operation,
          false, true, "invalid_axis",
          dtypeName & "." & operation & ".invalid_axis." & $axis)
      invalid[operation] = byOperation
    result["invalid_scan_axes"] = invalid

proc integerOutputForDtype[T](source: Tensor[T], spec: JsonNode,
                               dtypeName: string): JsonNode =
  case dtypeName
  of "bool": integerOutputDtypeCase[T, bool](source, spec, bool, dtypeName)
  of "uint8": integerOutputDtypeCase[T, uint8](source, spec, uint8, dtypeName)
  of "int8": integerOutputDtypeCase[T, int8](source, spec, int8, dtypeName)
  of "int16": integerOutputDtypeCase[T, int16](source, spec, int16, dtypeName)
  of "int32": integerOutputDtypeCase[T, int32](source, spec, int32, dtypeName)
  of "int64": integerOutputDtypeCase[T, int64](source, spec, int64, dtypeName)
  of "uint16": integerOutputDtypeCase[T, uint16](source, spec, uint16, dtypeName)
  of "uint32": integerOutputDtypeCase[T, uint32](source, spec, uint32, dtypeName)
  of "uint64": integerOutputDtypeCase[T, uint64](source, spec, uint64, dtypeName)
  else: raise newException(ValueError, "unsupported integer output dtype: " & dtypeName)

proc integerOutputCaseForSource[T](spec: JsonNode): JsonNode =
  let source = integerOutputSource[T](spec)
  let outputDtypes = if spec.hasKey("output_dtypes"): spec["output_dtypes"]
                     else: %*["bool", "uint8", "int8", "int16", "int32", "int64",
                              "uint16", "uint32", "uint64"]
  result = newJObject()
  result["reductions"] = newJObject()
  result["scans"] = newJObject()
  result["mean"] = %*{"all": newJObject(), "axes": newJObject()}
  if spec.hasKey("invalid_axes"): result["invalid_scan_axes"] = newJObject()
  for dtypeNode in outputDtypes:
    let dtype = dtypeNode.getStr
    let observations = integerOutputForDtype(source, spec, dtype)
    for operation in ["sum", "prod"]:
      if not result["reductions"].hasKey(operation):
        result["reductions"][operation] = %*{"all": newJObject(), "axes": newJObject()}
      result["reductions"][operation]["all"][dtype] = observations["reductions"][operation]["all"]
      for axisNode in spec["axes"]:
        let key = $axisNode.getInt
        if not result["reductions"][operation]["axes"].hasKey(key):
          result["reductions"][operation]["axes"][key] = newJObject()
        result["reductions"][operation]["axes"][key][dtype] =
          observations["reductions"][operation]["axes"][key]
    for operation in ["cumsum", "cumprod"]:
      if not result["scans"].hasKey(operation):
        result["scans"][operation] = %*{"axes": newJObject()}
      for axisNode in spec["axes"]:
        let key = $axisNode.getInt
        if not result["scans"][operation]["axes"].hasKey(key):
          result["scans"][operation]["axes"][key] = newJObject()
        result["scans"][operation]["axes"][key][dtype] =
          observations["scans"][operation]["axes"][key]
    result["mean"]["all"][dtype] = observations["mean"]["all"][dtype]
    for axisNode in spec["axes"]:
      let key = $axisNode.getInt
      if not result["mean"]["axes"].hasKey(key):
        result["mean"]["axes"][key] = newJObject()
      result["mean"]["axes"][key][dtype] = observations["mean"]["axes"][key][dtype]
    if spec.hasKey("invalid_axes"):
      for operation in ["cumsum", "cumprod"]:
        if not result["invalid_scan_axes"].hasKey(operation):
          result["invalid_scan_axes"][operation] = newJObject()
        for axisNode in spec["invalid_axes"]:
          let key = $axisNode.getInt
          if not result["invalid_scan_axes"][operation].hasKey(key):
            result["invalid_scan_axes"][operation][key] = newJObject()
          result["invalid_scan_axes"][operation][key][dtype] =
            observations["invalid_scan_axes"][operation][key]

proc integerOutputBySource(spec: JsonNode): JsonNode =
  case spec["dtype"].getStr
  of "bool": integerOutputCaseForSource[bool](spec)
  of "int8": integerOutputCaseForSource[int8](spec)
  of "int16": integerOutputCaseForSource[int16](spec)
  of "int32": integerOutputCaseForSource[int32](spec)
  of "int64": integerOutputCaseForSource[int64](spec)
  of "uint8": integerOutputCaseForSource[uint8](spec)
  of "uint16": integerOutputCaseForSource[uint16](spec)
  of "uint32": integerOutputCaseForSource[uint32](spec)
  of "uint64": integerOutputCaseForSource[uint64](spec)
  of "float32": integerOutputCaseForSource[float32](spec)
  of "float64": integerOutputCaseForSource[float64](spec)
  else: raise newException(ValueError, "unsupported integer output source dtype: " & spec["dtype"].getStr)

proc integerOutputDtype(inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = integerOutputBySource(spec)
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


proc getComplexTensor[T](desc: JsonNode): Tensor[T] =
  let shapeNode = desc["shape"]
  if shapeNode.len > LASER_MAXRANK:
    raise newException(ValueError, "input rank exceeds LASER_MAXRANK")
  var shape = newSeq[int](shapeNode.len)
  for index in 0 ..< shapeNode.len: shape[index] = shapeNode[index].getInt
  result = newTensorUninit[T](toMetadata(shape))
  let values = desc["values"]
  if values.len != result.size:
    raise newException(ValueError, "complex scan input value count mismatch")
  var raw = result.unsafe_raw_offset
  for index in 0 ..< values.len:
    let pair = values[index]
    when T is Complex[float32]:
      raw[index] = complex32(pair[0].getFloat.float32, pair[1].getFloat.float32)
    elif T is Complex[float64]:
      raw[index] = complex64(pair[0].getFloat, pair[1].getFloat)
    else:
      {.error: "unsupported complex scan input dtype".}

proc complexScanSpecials[T](source: var Tensor[T], specials: JsonNode) =
  for special in specials:
    let index = special["index"].getInt
    let value = case special["value"].getStr
      of "nan": NaN
      of "+inf": Inf
      else: NegInf
    let component = special["component"].getStr
    when T is Complex[float32]:
      case component
      of "real": source.unsafe_raw_offset[index].re = value.float32
      of "imag": source.unsafe_raw_offset[index].im = value.float32
      else: raise newException(ValueError, "unknown complex scan component")
    elif T is Complex[float64]:
      case component
      of "real": source.unsafe_raw_offset[index].re = value
      of "imag": source.unsafe_raw_offset[index].im = value
      else: raise newException(ValueError, "unknown complex scan component")
    else:
      {.error: "unsupported complex scan input dtype".}


proc negativeZero(value: float32): bool =
  value == 0'f32 and (cast[uint32](value) shr 31) == 1'u32

proc negativeZero(value: float64): bool =
  value == 0.0 and (cast[uint64](value) shr 63) == 1'u64

proc complexTensorObservation[T](t: Tensor[T]): JsonNode =
  result = newJObject()
  var shape = newJArray()
  for dimension in t.shape: shape.add newJInt(dimension)
  result["shape"] = shape
  result["numel"] = newJInt(t.size.int)
  when T is Complex[float32]: result["dtype"] = %"complex64"
  elif T is Complex[float64]: result["dtype"] = %"complex128"
  else: {.error: "unsupported complex scan observation dtype".}
  result["device"] = %"cpu"
  var values = newJArray()
  var negativeReal = newJArray()
  var negativeImag = newJArray()
  for index, value in enumerate(t):
    var components = newJObject()
    components["real"] = floatNode(value.re.float64)
    components["imag"] = floatNode(value.im.float64)
    values.add components
    if negativeZero(value.re): negativeReal.add newJInt(index)
    if negativeZero(value.im): negativeImag.add newJInt(index)
  result["values"] = values
  result["negative_zero"] = %*{"real": negativeReal, "imag": negativeImag}

proc complexScanAxis[T](source: Tensor[T], axis: int, operation: string,
                        reverse: bool): Tensor[T] =
  case operation
  of "cumsum": result = source.cumsum(axis = axis, reverse = reverse)
  of "cumprod": result = source.cumprod(axis = axis, reverse = reverse)
  else: raise newException(ValueError, "unsupported complex scan: " & operation)

proc complexInvalidScanAxis[T](source: Tensor[T], axis: int,
                               operation, label: string): JsonNode =
  try:
    discard complexScanAxis(source, axis, operation, false)
  except ValueError as error:
    let message = error.msg.toLowerAscii
    if not (message.contains("axis") or message.contains("dimension") or
            message.contains("index") or message.contains("range")):
      raise
    return %*{"error": "invalid_axis"}
  raise newException(ValueError, label & " unexpectedly accepted an invalid scan axis")

proc complexScanDirections[T](source: Tensor[T], axis: int,
                              operation: string): JsonNode =
  result = newJObject()
  result["forward"] = complexTensorObservation(
    complexScanAxis(source, axis, operation, false))
  result["reverse"] = complexTensorObservation(
    complexScanAxis(source, axis, operation, true))

proc complexScanCase[T](spec: JsonNode): JsonNode =
  var source = getComplexTensor[T](spec)
  if spec.hasKey("nonfinite"):
    complexScanSpecials(source, spec["nonfinite"])
  source = realDtypeLayout(source, spec["layout"].getStr)
  result = newJObject()
  for operation in ["cumsum", "cumprod"]:
    var axes = newJObject()
    for axisNode in spec["axes"]:
      let axis = axisNode.getInt
      axes[$axis] = complexScanDirections(source, axis, operation)
    var invalidAxes = newJObject()
    for axisNode in spec["invalid_axes"]:
      let axis = axisNode.getInt
      invalidAxes[$axis] = complexInvalidScanAxis(
        source, axis, operation, operation & ".invalid_axis." & $axis)
    result[operation] = %*{"axes": axes, "invalid_axes": invalidAxes}

proc complexScans(inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = case spec["dtype"].getStr
      of "complex64": complexScanCase[Complex[float32]](spec)
      of "complex128": complexScanCase[Complex[float64]](spec)
      else: raise newException(ValueError, "unsupported complex scan dtype")

proc complexScanDtypeAxis[T, U](source: Tensor[T], axis: int, dtype: typedesc[U],
                                operation: string, reverse: bool): Tensor[U] =
  case operation
  of "cumsum": result = source.cumsum(axis = axis, dtype = dtype, reverse = reverse)
  of "cumprod": result = source.cumprod(axis = axis, dtype = dtype, reverse = reverse)
  else: raise newException(ValueError, "unsupported complex scan dtype operation: " & operation)

proc complexScanDtypeObservation[T](t: Tensor[T]): JsonNode =
  when T is Complex[float32] or T is Complex[float64]:
    result = complexTensorObservation(t)
  elif T is float32 or T is float64:
    result = tensorObservation(t)
    var negative = newJArray()
    for index, value in enumerate(t):
      if negativeZero(value): negative.add newJInt(index)
    result["negative_zero"] = negative
  else:
    result = tensorObservation(t)

proc complexScanDtypeDirection[T, U](source: Tensor[T], axis: int,
                                     dtype: typedesc[U], operation: string,
                                     reverse, reject: bool, label: string): JsonNode =
  if reject:
    try:
      discard complexScanDtypeAxis(source, axis, dtype, operation, reverse)
    except ValueError as error:
      if error.msg.toLowerAscii.contains("cpu cumulative scans do not support this output dtype"):
        return %*{"error": "unsupported_output_dtype"}
      raise
    raise newException(ValueError, label & " unexpectedly accepted unsupported output dtype")
  result = complexScanDtypeObservation(
    complexScanDtypeAxis(source, axis, dtype, operation, reverse))

proc complexScanDtypeInvalidAxis[T, U](source: Tensor[T], axis: int,
                                       dtype: typedesc[U], operation,
                                       label: string): JsonNode =
  try:
    discard complexScanDtypeAxis(source, axis, dtype, operation, false)
  except ValueError as error:
    let message = error.msg.toLowerAscii
    if not (message.contains("axis") or message.contains("dimension") or
            message.contains("index") or message.contains("range")):
      raise
    return %*{"error": "invalid_axis"}
  raise newException(ValueError, label & " unexpectedly accepted an invalid scan axis")

proc complexScanDtypeTarget[T, U](source: Tensor[T], spec: JsonNode,
                                  dtype: typedesc[U], dtypeName: string): JsonNode =
  let reject = dtypeName in ["bool", "uint16", "uint32", "uint64"] and
    source.size > 0 and source.shape.len > 0
  result = newJObject()
  for operation in ["cumsum", "cumprod"]:
    var axes = newJObject()
    for axisNode in spec["axes"]:
      let axis = axisNode.getInt
      var directions = newJObject()
      for reverse in [false, true]:
        let direction = if reverse: "reverse" else: "forward"
        directions[direction] = complexScanDtypeDirection(
          source, axis, dtype, operation, reverse, reject,
          dtypeName & "." & operation & "." & $axis & "." & direction)
      axes[$axis] = directions
    var invalidAxes = newJObject()
    for axisNode in spec["invalid_axes"]:
      let axis = axisNode.getInt
      invalidAxes[$axis] = complexScanDtypeInvalidAxis(
        source, axis, dtype, operation,
        dtypeName & "." & operation & ".invalid_axis." & $axis)
    result[operation] = %*{"axes": axes, "invalid_axes": invalidAxes}

proc complexScanDtypeForOutput[T](source: Tensor[T], spec: JsonNode,
                                  dtypeName: string): JsonNode =
  case dtypeName
  of "bool": complexScanDtypeTarget[T, bool](source, spec, bool, dtypeName)
  of "uint8": complexScanDtypeTarget[T, uint8](source, spec, uint8, dtypeName)
  of "int8": complexScanDtypeTarget[T, int8](source, spec, int8, dtypeName)
  of "int16": complexScanDtypeTarget[T, int16](source, spec, int16, dtypeName)
  of "int32": complexScanDtypeTarget[T, int32](source, spec, int32, dtypeName)
  of "int64": complexScanDtypeTarget[T, int64](source, spec, int64, dtypeName)
  of "uint16": complexScanDtypeTarget[T, uint16](source, spec, uint16, dtypeName)
  of "uint32": complexScanDtypeTarget[T, uint32](source, spec, uint32, dtypeName)
  of "uint64": complexScanDtypeTarget[T, uint64](source, spec, uint64, dtypeName)
  of "float32": complexScanDtypeTarget[T, float32](source, spec, float32, dtypeName)
  of "float64": complexScanDtypeTarget[T, float64](source, spec, float64, dtypeName)
  of "complex64": complexScanDtypeTarget[T, Complex[float32]](
    source, spec, Complex[float32], dtypeName)
  of "complex128": complexScanDtypeTarget[T, Complex[float64]](
    source, spec, Complex[float64], dtypeName)
  else: raise newException(ValueError, "unsupported complex scan output dtype: " & dtypeName)

proc complexScanDtypeSource[T](spec: JsonNode): Tensor[T] =
  var source =
    when T is Complex[float32] or T is Complex[float64]: getComplexTensor[T](spec)
    elif T is float32: realDtypeFloat32Tensor(spec)
    elif T is float64: getTensor(spec)
    else: getIntegerTensor[T](spec)
  if spec.hasKey("nonfinite"):
    when T is Complex[float32] or T is Complex[float64]:
      complexScanSpecials(source, spec["nonfinite"])
    elif T is float32 or T is float64:
      realDtypeSpecials(source, spec["nonfinite"])
    else:
      raise newException(ValueError, "nonfinite scan dtype source must be floating point")
  result = realDtypeLayout(source, spec["layout"].getStr)


proc complexScanDtypeCase[T](spec: JsonNode): JsonNode =
  let source = complexScanDtypeSource[T](spec)
  result = newJObject()
  for dtypeNode in spec["output_dtypes"]:
    let dtypeName = dtypeNode.getStr
    result[dtypeName] = complexScanDtypeForOutput(source, spec, dtypeName)

proc complexScanDtypeBySource(spec: JsonNode): JsonNode =
  case spec["dtype"].getStr
  of "bool": complexScanDtypeCase[bool](spec)
  of "int8": complexScanDtypeCase[int8](spec)
  of "int16": complexScanDtypeCase[int16](spec)
  of "int32": complexScanDtypeCase[int32](spec)
  of "int64": complexScanDtypeCase[int64](spec)
  of "uint8": complexScanDtypeCase[uint8](spec)
  of "uint16": complexScanDtypeCase[uint16](spec)
  of "uint32": complexScanDtypeCase[uint32](spec)
  of "uint64": complexScanDtypeCase[uint64](spec)
  of "float32": complexScanDtypeCase[float32](spec)
  of "float64": complexScanDtypeCase[float64](spec)
  of "complex64": complexScanDtypeCase[Complex[float32]](spec)
  of "complex128": complexScanDtypeCase[Complex[float64]](spec)
  else: raise newException(ValueError, "unsupported complex scan input dtype")

proc complexScanDtype(inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = complexScanDtypeBySource(spec)
proc complexMeanIntegerError[T, U](source: Tensor[T], dtype: typedesc[U],
                                   label: string): JsonNode =
  try:
    discard source.mean(dtype = dtype)
  except ValueError as error:
    let message = error.msg.toLowerAscii
    if not (message.contains("mean") and message.contains("dtype")):
      raise
    return %*{"error": "integer_mean_dtype_rejected"}
  raise newException(ValueError, label & " unexpectedly accepted an integer mean dtype")

proc complexMeanIntegerAxisError[T, U](source: Tensor[T], axis: int,
                                       dtype: typedesc[U], label: string): JsonNode =
  try:
    discard source.mean(axis = axis, dtype = dtype)
  except ValueError as error:
    let message = error.msg.toLowerAscii
    if not (message.contains("mean") and message.contains("dtype")):
      raise
    return %*{"error": "integer_mean_dtype_rejected"}
  raise newException(ValueError, label & " unexpectedly accepted an integer mean dtype")

proc complexMeanInvalidAxis[T, U](source: Tensor[T], axis: int,
                                  dtype: typedesc[U], label: string): JsonNode =
  try:
    discard source.mean(axis = axis, dtype = dtype)
  except ValueError as error:
    let message = error.msg.toLowerAscii
    if not (message.contains("axis") or message.contains("dimension") or
            message.contains("index") or message.contains("range")):
      raise
    return %*{"error": "invalid_axis"}
  except IndexDefect as error:
    if not error.msg.toLowerAscii.contains("reduction axis out of bounds"): raise
    return %*{"error": "invalid_axis"}
  raise newException(ValueError, label & " unexpectedly accepted an invalid mean axis")

proc complexMeanInvalidDefaultAxis[T](source: Tensor[T], axis: int,
                                      label: string): JsonNode =
  try:
    discard source.mean(axis = axis)
  except ValueError as error:
    let message = error.msg.toLowerAscii
    if not (message.contains("axis") or message.contains("dimension") or
            message.contains("index") or message.contains("range")):
      raise
    return %*{"error": "invalid_axis"}
  except IndexDefect as error:
    if not error.msg.toLowerAscii.contains("reduction axis out of bounds"): raise
    return %*{"error": "invalid_axis"}
  raise newException(ValueError, label & " unexpectedly accepted an invalid mean axis")

proc complexMeanDtypeTarget[T, U](source: Tensor[T], spec: JsonNode,
                                  dtype: typedesc[U], dtypeName: string): JsonNode =
  let integerTarget = dtypeName in [
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
  ]
  result = newJObject()
  if integerTarget:
    result["all"] = complexMeanIntegerError(
      source, dtype, dtypeName & ".mean.all")
  else:
    result["all"] = complexScanDtypeObservation(
      newTensorWith[U]([], source.mean(dtype = dtype)))

  var axes = newJObject()
  for axisNode in spec["axes"]:
    let axis = axisNode.getInt
    if integerTarget:
      axes[$axis] = complexMeanIntegerAxisError(
        source, axis, dtype, dtypeName & ".mean." & $axis)
    else:
      axes[$axis] = complexScanDtypeObservation(
        source.mean(axis = axis, dtype = dtype))
  result["axes"] = axes

  var invalidAxes = newJObject()
  for axisNode in spec["invalid_axes"]:
    let axis = axisNode.getInt
    if integerTarget:
      invalidAxes[$axis] = complexMeanIntegerAxisError(
        source, axis, dtype, dtypeName & ".mean.invalid_axis." & $axis)
    else:
      invalidAxes[$axis] = complexMeanInvalidAxis(
        source, axis, dtype, dtypeName & ".mean.invalid_axis." & $axis)
  result["invalid_axes"] = invalidAxes

proc complexMeanDefaultTarget[T](source: Tensor[T], spec: JsonNode): JsonNode =
  result = newJObject()
  result["all"] = complexScanDtypeObservation(newTensorWith[T]([], source.mean))

  var axes = newJObject()
  for axisNode in spec["axes"]:
    let axis = axisNode.getInt
    axes[$axis] = complexScanDtypeObservation(source.mean(axis = axis))
  result["axes"] = axes

  var invalidAxes = newJObject()
  for axisNode in spec["invalid_axes"]:
    let axis = axisNode.getInt
    invalidAxes[$axis] = complexMeanInvalidDefaultAxis(
      source, axis, "default.mean.invalid_axis." & $axis)
  result["invalid_axes"] = invalidAxes

proc complexMeanDtypeForOutput[T](source: Tensor[T], spec: JsonNode,
                                  dtypeName: string): JsonNode =
  case dtypeName
  of "bool": complexMeanDtypeTarget[T, bool](source, spec, bool, dtypeName)
  of "uint8": complexMeanDtypeTarget[T, uint8](source, spec, uint8, dtypeName)
  of "int8": complexMeanDtypeTarget[T, int8](source, spec, int8, dtypeName)
  of "int16": complexMeanDtypeTarget[T, int16](source, spec, int16, dtypeName)
  of "int32": complexMeanDtypeTarget[T, int32](source, spec, int32, dtypeName)
  of "int64": complexMeanDtypeTarget[T, int64](source, spec, int64, dtypeName)
  of "uint16": complexMeanDtypeTarget[T, uint16](source, spec, uint16, dtypeName)
  of "uint32": complexMeanDtypeTarget[T, uint32](source, spec, uint32, dtypeName)
  of "uint64": complexMeanDtypeTarget[T, uint64](source, spec, uint64, dtypeName)
  of "float32": complexMeanDtypeTarget[T, float32](source, spec, float32, dtypeName)
  of "float64": complexMeanDtypeTarget[T, float64](source, spec, float64, dtypeName)
  of "complex64": complexMeanDtypeTarget[T, Complex[float32]](
    source, spec, Complex[float32], dtypeName)
  of "complex128": complexMeanDtypeTarget[T, Complex[float64]](
    source, spec, Complex[float64], dtypeName)
  else: raise newException(ValueError, "unsupported complex mean output dtype: " & dtypeName)

proc complexMeanDtypeCase[T](spec: JsonNode): JsonNode =
  let source = complexScanDtypeSource[T](spec)
  result = newJObject()
  for dtypeNode in spec["output_dtypes"]:
    let dtypeName = dtypeNode.getStr
    if dtypeName == "default":
      when T is Complex[float32] or T is Complex[float64]:
        result[dtypeName] = complexMeanDefaultTarget(source, spec)
      else:
        raise newException(ValueError, "default complex mean requires a complex input")
    else:
      result[dtypeName] = complexMeanDtypeForOutput(source, spec, dtypeName)

proc complexMeanDtypeBySource(spec: JsonNode): JsonNode =
  case spec["dtype"].getStr
  of "bool": complexMeanDtypeCase[bool](spec)
  of "int8": complexMeanDtypeCase[int8](spec)
  of "int16": complexMeanDtypeCase[int16](spec)
  of "int32": complexMeanDtypeCase[int32](spec)
  of "int64": complexMeanDtypeCase[int64](spec)
  of "uint8": complexMeanDtypeCase[uint8](spec)
  of "uint16": complexMeanDtypeCase[uint16](spec)
  of "uint32": complexMeanDtypeCase[uint32](spec)
  of "uint64": complexMeanDtypeCase[uint64](spec)
  of "float32": complexMeanDtypeCase[float32](spec)
  of "float64": complexMeanDtypeCase[float64](spec)
  of "complex64": complexMeanDtypeCase[Complex[float32]](spec)
  of "complex128": complexMeanDtypeCase[Complex[float64]](spec)
  else: raise newException(ValueError, "unsupported complex mean input dtype")

proc complexMeanDtype(inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = complexMeanDtypeBySource(spec)


proc complexSumUnsupported(message: string): bool =
  message == "CPU reductions do not support this unsigned output dtype"

proc complexSumDtypeScalar[T, U](source: Tensor[T], dtype: typedesc[U],
                                 reject: bool, label: string): JsonNode =
  if reject:
    try:
      discard source.sum(dtype = dtype)
    except ValueError as error:
      if not complexSumUnsupported(error.msg): raise
      return integerOutputUnsupported()
    raise newException(ValueError, label & " unexpectedly accepted an unsupported output dtype")
  result = complexScanDtypeObservation(newTensorWith[U]([], source.sum(dtype = dtype)))

proc complexSumDtypeAxis[T, U](source: Tensor[T], axis: int,
                               dtype: typedesc[U], reject: bool,
                               label: string): JsonNode =
  if reject:
    try:
      discard source.sum(axis = axis, dtype = dtype)
    except ValueError as error:
      if not complexSumUnsupported(error.msg): raise
      return integerOutputUnsupported()
    raise newException(ValueError, label & " unexpectedly accepted an unsupported output dtype")
  result = complexScanDtypeObservation(source.sum(axis = axis, dtype = dtype))

proc complexSumInvalidAxisError(message: string): bool =
  let lowered = message.toLowerAscii
  let namesAxis = lowered.contains("axis") or lowered.contains("dimension")
  let reportsInvalidRange = lowered.contains("invalid") or
    lowered.contains("out of range") or lowered.contains("out of bounds")
  namesAxis and reportsInvalidRange

proc complexSumInvalidAxis[T, U](source: Tensor[T], axis: int,
                                 dtype: typedesc[U], label: string): JsonNode =
  try:
    discard source.sum(axis = axis, dtype = dtype)
  except ValueError as error:
    if not complexSumInvalidAxisError(error.msg): raise
    return %*{"error": "invalid_axis"}
  except IndexDefect as error:
    if not error.msg.toLowerAscii.contains("reduction axis out of bounds"): raise
    return %*{"error": "invalid_axis"}
  raise newException(ValueError, label & " unexpectedly accepted an invalid sum axis")

proc complexSumInvalidDefaultAxis[T](source: Tensor[T], axis: int,
                                     label: string): JsonNode =
  try:
    discard source.sum(axis = axis)
  except ValueError as error:
    if not complexSumInvalidAxisError(error.msg): raise
    return %*{"error": "invalid_axis"}
  except IndexDefect as error:
    if not error.msg.toLowerAscii.contains("reduction axis out of bounds"): raise
    return %*{"error": "invalid_axis"}
  raise newException(ValueError, label & " unexpectedly accepted an invalid sum axis")

proc complexSumDefaultTarget[T](source: Tensor[T], spec: JsonNode): JsonNode =
  result = newJObject()
  result["all"] = complexScanDtypeObservation(newTensorWith[T]([], source.sum))

  var axes = newJObject()
  for axisNode in spec["axes"]:
    let axis = axisNode.getInt
    axes[$axis] = complexScanDtypeObservation(source.sum(axis = axis))
  result["axes"] = axes

  var invalidAxes = newJObject()
  for axisNode in spec["invalid_axes"]:
    let axis = axisNode.getInt
    invalidAxes[$axis] = complexSumInvalidDefaultAxis(
      source, axis, "default.sum.invalid_axis." & $axis)
  result["invalid_axes"] = invalidAxes


proc complexSumDtypeTarget[T, U](source: Tensor[T], spec: JsonNode,
                                 dtype: typedesc[U], dtypeName: string): JsonNode =
  let reject = dtypeName in ["uint16", "uint32", "uint64"] and source.size > 0
  result = newJObject()
  result["all"] = complexSumDtypeScalar(
    source, dtype, reject, dtypeName & ".sum.all")

  var axes = newJObject()
  for axisNode in spec["axes"]:
    let axis = axisNode.getInt
    axes[$axis] = complexSumDtypeAxis(
      source, axis, dtype, reject, dtypeName & ".sum." & $axis)
  result["axes"] = axes

  var invalidAxes = newJObject()
  for axisNode in spec["invalid_axes"]:
    let axis = axisNode.getInt
    invalidAxes[$axis] = complexSumInvalidAxis(
      source, axis, dtype, dtypeName & ".sum.invalid_axis." & $axis)
  result["invalid_axes"] = invalidAxes

proc complexSumDtypeForOutput[T](source: Tensor[T], spec: JsonNode,
                                 dtypeName: string): JsonNode =
  case dtypeName
  of "bool": complexSumDtypeTarget[T, bool](source, spec, bool, dtypeName)
  of "uint8": complexSumDtypeTarget[T, uint8](source, spec, uint8, dtypeName)
  of "int8": complexSumDtypeTarget[T, int8](source, spec, int8, dtypeName)
  of "int16": complexSumDtypeTarget[T, int16](source, spec, int16, dtypeName)
  of "int32": complexSumDtypeTarget[T, int32](source, spec, int32, dtypeName)
  of "int64": complexSumDtypeTarget[T, int64](source, spec, int64, dtypeName)
  of "uint16": complexSumDtypeTarget[T, uint16](source, spec, uint16, dtypeName)
  of "uint32": complexSumDtypeTarget[T, uint32](source, spec, uint32, dtypeName)
  of "uint64": complexSumDtypeTarget[T, uint64](source, spec, uint64, dtypeName)
  of "float32": complexSumDtypeTarget[T, float32](source, spec, float32, dtypeName)
  of "float64": complexSumDtypeTarget[T, float64](source, spec, float64, dtypeName)
  of "complex64": complexSumDtypeTarget[T, Complex[float32]](
    source, spec, Complex[float32], dtypeName)
  of "complex128": complexSumDtypeTarget[T, Complex[float64]](
    source, spec, Complex[float64], dtypeName)
  else: raise newException(ValueError, "unsupported complex sum output dtype: " & dtypeName)

proc complexSumDtypeCase[T](spec: JsonNode): JsonNode =
  let source = complexScanDtypeSource[T](spec)
  result = newJObject()
  for dtypeNode in spec["output_dtypes"]:
    let dtypeName = dtypeNode.getStr
    if dtypeName == "default":
      when T is Complex[float32] or T is Complex[float64]:
        result[dtypeName] = complexSumDefaultTarget(source, spec)
      else:
        raise newException(ValueError, "default complex sum requires a complex input")
    else:
      result[dtypeName] = complexSumDtypeForOutput(source, spec, dtypeName)

proc complexSumDtypeBySource(spec: JsonNode): JsonNode =
  case spec["dtype"].getStr
  of "bool": complexSumDtypeCase[bool](spec)
  of "int8": complexSumDtypeCase[int8](spec)
  of "int16": complexSumDtypeCase[int16](spec)
  of "int32": complexSumDtypeCase[int32](spec)
  of "int64": complexSumDtypeCase[int64](spec)
  of "uint8": complexSumDtypeCase[uint8](spec)
  of "uint16": complexSumDtypeCase[uint16](spec)
  of "uint32": complexSumDtypeCase[uint32](spec)
  of "uint64": complexSumDtypeCase[uint64](spec)
  of "float32": complexSumDtypeCase[float32](spec)
  of "float64": complexSumDtypeCase[float64](spec)
  of "complex64": complexSumDtypeCase[Complex[float32]](spec)
  of "complex128": complexSumDtypeCase[Complex[float64]](spec)
  else: raise newException(ValueError, "unsupported complex sum input dtype")

proc complexSumDtype(inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = complexSumDtypeBySource(spec)

proc complexProductDtypeScalar[T, U](source: Tensor[T], dtype: typedesc[U],
                                     reject: bool, label: string): JsonNode =
  if reject:
    try:
      discard source.product(dtype = dtype)
    except ValueError as error:
      if not complexSumUnsupported(error.msg): raise
      return integerOutputUnsupported()
    raise newException(ValueError, label & " unexpectedly accepted an unsupported output dtype")
  result = complexScanDtypeObservation(
    newTensorWith[U]([], source.product(dtype = dtype)))

proc complexProductDtypeAxis[T, U](source: Tensor[T], axis: int,
                                   dtype: typedesc[U], reject: bool,
                                   label: string): JsonNode =
  if reject:
    try:
      discard source.product(axis = axis, dtype = dtype)
    except ValueError as error:
      if not complexSumUnsupported(error.msg): raise
      return integerOutputUnsupported()
    raise newException(ValueError, label & " unexpectedly accepted an unsupported output dtype")
  result = complexScanDtypeObservation(
    source.product(axis = axis, dtype = dtype))

proc complexProductInvalidAxis[T, U](source: Tensor[T], axis: int,
                                     dtype: typedesc[U], label: string): JsonNode =
  try:
    discard source.product(axis = axis, dtype = dtype)
  except ValueError as error:
    if not complexSumInvalidAxisError(error.msg): raise
    return %*{"error": "invalid_axis"}
  except IndexDefect as error:
    if not error.msg.toLowerAscii.contains("reduction axis out of bounds"): raise
    return %*{"error": "invalid_axis"}
  raise newException(ValueError, label & " unexpectedly accepted an invalid product axis")

proc complexProductInvalidDefaultAxis[T](source: Tensor[T], axis: int,
                                         label: string): JsonNode =
  try:
    discard source.product(axis = axis)
  except ValueError as error:
    if not complexSumInvalidAxisError(error.msg): raise
    return %*{"error": "invalid_axis"}
  except IndexDefect as error:
    if not error.msg.toLowerAscii.contains("reduction axis out of bounds"): raise
    return %*{"error": "invalid_axis"}
  raise newException(ValueError, label & " unexpectedly accepted an invalid product axis")

proc complexProductDefaultTarget[T](source: Tensor[T], spec: JsonNode): JsonNode =
  result = newJObject()
  result["all"] = complexScanDtypeObservation(newTensorWith[T]([], source.product))

  var axes = newJObject()
  for axisNode in spec["axes"]:
    let axis = axisNode.getInt
    axes[$axis] = complexScanDtypeObservation(
      source.product(axis = axis))
  result["axes"] = axes

  var invalidAxes = newJObject()
  for axisNode in spec["invalid_axes"]:
    let axis = axisNode.getInt
    invalidAxes[$axis] = complexProductInvalidDefaultAxis(
      source, axis, "default.product.invalid_axis." & $axis)
  result["invalid_axes"] = invalidAxes

proc complexProductDtypeTarget[T, U](source: Tensor[T], spec: JsonNode,
                                     dtype: typedesc[U], dtypeName: string): JsonNode =
  let reject = dtypeName in ["uint16", "uint32", "uint64"] and source.size > 0
  result = newJObject()
  result["all"] = complexProductDtypeScalar(
    source, dtype, reject, dtypeName & ".product.all")

  var axes = newJObject()
  for axisNode in spec["axes"]:
    let axis = axisNode.getInt
    axes[$axis] = complexProductDtypeAxis(
      source, axis, dtype, reject, dtypeName & ".product." & $axis)
  result["axes"] = axes

  var invalidAxes = newJObject()
  for axisNode in spec["invalid_axes"]:
    let axis = axisNode.getInt
    invalidAxes[$axis] = complexProductInvalidAxis(
      source, axis, dtype, dtypeName & ".product.invalid_axis." & $axis)
  result["invalid_axes"] = invalidAxes

proc complexProductDtypeForOutput[T](source: Tensor[T], spec: JsonNode,
                                    dtypeName: string): JsonNode =
  case dtypeName
  of "bool": complexProductDtypeTarget[T, bool](source, spec, bool, dtypeName)
  of "uint8": complexProductDtypeTarget[T, uint8](source, spec, uint8, dtypeName)
  of "int8": complexProductDtypeTarget[T, int8](source, spec, int8, dtypeName)
  of "int16": complexProductDtypeTarget[T, int16](source, spec, int16, dtypeName)
  of "int32": complexProductDtypeTarget[T, int32](source, spec, int32, dtypeName)
  of "int64": complexProductDtypeTarget[T, int64](source, spec, int64, dtypeName)
  of "uint16": complexProductDtypeTarget[T, uint16](source, spec, uint16, dtypeName)
  of "uint32": complexProductDtypeTarget[T, uint32](source, spec, uint32, dtypeName)
  of "uint64": complexProductDtypeTarget[T, uint64](source, spec, uint64, dtypeName)
  of "float32": complexProductDtypeTarget[T, float32](source, spec, float32, dtypeName)
  of "float64": complexProductDtypeTarget[T, float64](source, spec, float64, dtypeName)
  of "complex64": complexProductDtypeTarget[T, Complex[float32]](
    source, spec, Complex[float32], dtypeName)
  of "complex128": complexProductDtypeTarget[T, Complex[float64]](
    source, spec, Complex[float64], dtypeName)
  else: raise newException(ValueError, "unsupported complex product output dtype: " & dtypeName)

proc complexProductDtypeCase[T](spec: JsonNode): JsonNode =
  let source = complexScanDtypeSource[T](spec)
  result = newJObject()
  for dtypeNode in spec["output_dtypes"]:
    let dtypeName = dtypeNode.getStr
    if dtypeName == "default":
      when T is Complex[float32] or T is Complex[float64]:
        result[dtypeName] = complexProductDefaultTarget(source, spec)
      else:
        raise newException(ValueError, "default complex product requires a complex input")
    else:
      result[dtypeName] = complexProductDtypeForOutput(source, spec, dtypeName)

proc complexProductDtypeBySource(spec: JsonNode): JsonNode =
  case spec["dtype"].getStr
  of "bool": complexProductDtypeCase[bool](spec)
  of "int8": complexProductDtypeCase[int8](spec)
  of "int16": complexProductDtypeCase[int16](spec)
  of "int32": complexProductDtypeCase[int32](spec)
  of "int64": complexProductDtypeCase[int64](spec)
  of "uint8": complexProductDtypeCase[uint8](spec)
  of "uint16": complexProductDtypeCase[uint16](spec)
  of "uint32": complexProductDtypeCase[uint32](spec)
  of "uint64": complexProductDtypeCase[uint64](spec)
  of "float32": complexProductDtypeCase[float32](spec)
  of "float64": complexProductDtypeCase[float64](spec)
  of "complex64": complexProductDtypeCase[Complex[float32]](spec)
  of "complex128": complexProductDtypeCase[Complex[float64]](spec)
  else: raise newException(ValueError, "unsupported complex product input dtype")

proc complexProductDtype(inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = complexProductDtypeBySource(spec)


proc extremaSpecials[T](source: var Tensor[T], specials: JsonNode) =
  for special in specials:
    when T is SomeFloat:
      let value = case special["value"].getStr
        of "nan": NaN
        of "+inf": Inf
        of "-inf": -Inf
        else: raise newException(ValueError, "unknown extrema nonfinite value")
      var raw = source.unsafe_raw_offset
      raw[special["index"].getInt] = T(value)
    else:
      raise newException(ValueError, "nonfinite extrema input must be floating point")


proc extremaInputTensor[T](spec: JsonNode): Tensor[T] =
  when T is Complex[float32] or T is Complex[float64]:
    result = getComplexTensor[T](spec)
  elif T is SomeFloat:
    when T is float32:
      result = realDtypeFloat32Tensor(spec)
    else:
      result = getTensor(spec)
  else:
    result = getIntegerTensor[T](spec)
  if spec["specials"].len > 0:
    extremaSpecials(result, spec["specials"])
  if spec["layout"].getStr == "permute_210":
    result = result.permute(2, 1, 0)
  else:
    result = realDtypeLayout(result, spec["layout"].getStr)


proc extremaOutput[T](source: Tensor[T], request: JsonNode): Tensor[T] =
  let operation = request["operation"].getStr
  let dimensions = request["dimensions"]
  let keepdim = request["keepdim"].getBool
  if dimensions.kind == JNull:
    case operation
    of "amin": result = source.amin(keepdim = keepdim)
    of "amax": result = source.amax(keepdim = keepdim)
    else: raise newException(ValueError, "unknown extrema operation")
  elif dimensions.kind == JInt:
    let axis = dimensions.getInt
    case operation
    of "amin": result = source.amin(axis, keepdim = keepdim)
    of "amax": result = source.amax(axis, keepdim = keepdim)
    else: raise newException(ValueError, "unknown extrema operation")
  elif dimensions.kind == JArray:
    var axes = newSeq[int](dimensions.len)
    for index in 0 ..< dimensions.len:
      axes[index] = dimensions[index].getInt
    case operation
    of "amin": result = source.amin(axes, keepdim = keepdim)
    of "amax": result = source.amax(axes, keepdim = keepdim)
    else: raise newException(ValueError, "unknown extrema operation")
  else:
    raise newException(ValueError, "invalid extrema dimensions")


proc extremaTensorObservation[T](t: Tensor[T]): JsonNode =
  when T is Complex[float32]:
    if t.size != 0:
      raise newException(ValueError, "nonempty complex extrema output was returned")
    result = newJObject()
    var shape = newJArray()
    for dimension in t.shape: shape.add newJInt(dimension)
    result["shape"] = shape
    result["numel"] = newJInt(t.size.int)
    result["dtype"] = %"complex64"
    result["device"] = %"cpu"
    result["values"] = newJArray()
  elif T is Complex[float64]:
    if t.size != 0:
      raise newException(ValueError, "nonempty complex extrema output was returned")
    result = newJObject()
    var shape = newJArray()
    for dimension in t.shape: shape.add newJInt(dimension)
    result["shape"] = shape
    result["numel"] = newJInt(t.size.int)
    result["dtype"] = %"complex128"
    result["device"] = %"cpu"
    result["values"] = newJArray()
  else:
    result = tensorObservation(t)


proc extremaErrorCode(message: string): string =
  let normalized = message.toLowerAscii
  if normalized.contains("extrema axis out of range"):
    return "axis_out_of_range"
  if normalized.contains("extrema empty reduced axis"):
    return "empty_reduced_axis"
  if normalized.contains("extrema empty reduction requires dimensions") or
      normalized.contains("indexed extrema empty reduction requires an axis"):
    return "empty_reduction"
  if normalized.contains("extrema duplicate reduction axis"):
    return "duplicate_axes"
  if normalized.contains("extrema unsupported cpu dtype"):
    return "unsupported_dtype"


proc extremaObservation[T](source: Tensor[T], request: JsonNode): JsonNode =
  try:
    result = extremaTensorObservation(extremaOutput(source, request))
  except IndexDefect as error:
    let code = extremaErrorCode(error.msg)
    if code notin ["axis_out_of_range", "empty_reduced_axis"]:
      raise
    result = %*{"error": code}
  except ValueError as error:
    let code = extremaErrorCode(error.msg)
    if code notin ["empty_reduction", "duplicate_axes", "unsupported_dtype"]:
      raise
    result = %*{"error": code}


proc extremaCase[T](spec: JsonNode): JsonNode =
  let source = extremaInputTensor[T](spec)
  result = newJObject()
  for request in spec["requests"]:
    result[request["id"].getStr] = extremaObservation(source, request)


proc extrema(inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    if name == "native_int":
      result[name] = extremaCase[int](spec)
    elif name == "native_float":
      result[name] = extremaCase[float](spec)
    else:
      result[name] = case spec["dtype"].getStr
        of "bool": extremaCase[bool](spec)
        of "int8": extremaCase[int8](spec)
        of "int16": extremaCase[int16](spec)
        of "int32": extremaCase[int32](spec)
        of "int64": extremaCase[int64](spec)
        of "uint8": extremaCase[uint8](spec)
        of "uint16": extremaCase[uint16](spec)
        of "uint32": extremaCase[uint32](spec)
        of "uint64": extremaCase[uint64](spec)
        of "float32": extremaCase[float32](spec)
        of "float64": extremaCase[float64](spec)
        of "complex64": extremaCase[Complex[float32]](spec)
        of "complex128": extremaCase[Complex[float64]](spec)
        else: raise newException(ValueError, "unsupported extrema input dtype")


proc indexedExtremaFloatValue[T: SomeFloat](node: JsonNode): T =
  if node.kind == JString:
    case node.getStr
    of "nan": result = T(NaN)
    of "+inf": result = T(Inf)
    of "-inf": result = T(NegInf)
    of "-0": result = T(-0.0)
    else: raise newException(ValueError, "unknown indexed extrema floating value")
  else:
    result = T(node.getFloat)


proc indexedExtremaFloatTensor[T: SomeFloat](spec: JsonNode): Tensor[T] =
  let shapeNode = spec["shape"]
  if shapeNode.len > LASER_MAXRANK:
    raise newException(ValueError, "input rank exceeds LASER_MAXRANK")
  var shape = newSeq[int](shapeNode.len)
  for index in 0 ..< shapeNode.len: shape[index] = shapeNode[index].getInt
  result = newTensorUninit[T](toMetadata(shape))
  let values = spec["values"]
  if values.len != result.size:
    raise newException(ValueError, "indexed extrema input value count mismatch")
  var raw = result.unsafe_raw_offset
  for index in 0 ..< values.len:
    raw[index] = indexedExtremaFloatValue[T](values[index])


proc indexedExtremaInputTensor[T](spec: JsonNode): Tensor[T] =
  when T is Complex[float32] or T is Complex[float64]:
    result = getComplexTensor[T](spec)
  elif T is SomeFloat:
    result = indexedExtremaFloatTensor[T](spec)
  else:
    result = getIntegerTensor[T](spec)
  let layout = spec["layout"].getStr
  if layout == "permute":
    result = result.permute(2, 1, 0)
  else:
    result = realDtypeLayout(result, layout)


proc indexedExtremaOutput[T](source: Tensor[T], operation: string,
    axis: JsonNode, keepdim: bool): Tensor[int64] =
  if axis.kind == JNull:
    case operation
    of "argmin":
      result = if keepdim: source.argmin(keepdim = true) else: source.argmin()
    of "argmax":
      result = if keepdim: source.argmax(keepdim = true) else: source.argmax()
    else: raise newException(ValueError, "unknown indexed extrema operation")
  else:
    let dimension = axis.getInt
    case operation
    of "argmin":
      result = if keepdim: source.argmin(dimension, keepdim = true) else: source.argmin(dimension)
    of "argmax":
      result = if keepdim: source.argmax(dimension, keepdim = true) else: source.argmax(dimension)
    else: raise newException(ValueError, "unknown indexed extrema operation")


proc indexedExtremaRequestId(operation: string, axis: JsonNode,
    keepdim: bool): string =
  let axisName = if axis.kind == JNull: "all" else: "axis_" & $axis.getInt
  result = operation & "_" & axisName & (if keepdim: "_keep" else: "_drop")


proc indexedExtremaObservation[T](source: Tensor[T], operation: string,
    axis: JsonNode, keepdim: bool): JsonNode =
  try:
    result = tensorObservation(indexedExtremaOutput(source, operation, axis, keepdim))
  except IndexDefect as error:
    let code = extremaErrorCode(error.msg)
    if code notin ["axis_out_of_range", "empty_reduced_axis", "empty_reduction"]:
      raise
    result = %*{"error": code}
  except ValueError as error:
    let code = extremaErrorCode(error.msg)
    if code notin ["empty_reduction", "unsupported_dtype"]:
      raise
    result = %*{"error": code}


proc indexedExtremaCase[T](spec: JsonNode): JsonNode =
  let source = indexedExtremaInputTensor[T](spec)
  result = newJObject()
  for operation in ["argmin", "argmax"]:
    for axis in spec["axes"]:
      for keepdim in [false, true]:
        let requestId = indexedExtremaRequestId(operation, axis, keepdim)
        result[requestId] = indexedExtremaObservation(source, operation, axis, keepdim)


proc indexedExtremaDtype[T](inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = indexedExtremaCase[T](spec)


proc indexedExtrema(inputs: JsonNode): JsonNode =
  result = newJObject()
  for dtype, cases in inputs:
    result[dtype] = case dtype
      of "bool": indexedExtremaDtype[bool](cases)
      of "int8": indexedExtremaDtype[int8](cases)
      of "int16": indexedExtremaDtype[int16](cases)
      of "int32": indexedExtremaDtype[int32](cases)
      of "int64": indexedExtremaDtype[int64](cases)
      of "uint8": indexedExtremaDtype[uint8](cases)
      of "uint16": indexedExtremaDtype[uint16](cases)
      of "uint32": indexedExtremaDtype[uint32](cases)
      of "uint64": indexedExtremaDtype[uint64](cases)
      of "float32": indexedExtremaDtype[float32](cases)
      of "float64": indexedExtremaDtype[float64](cases)
      of "complex64": indexedExtremaDtype[Complex[float32]](cases)
      of "complex128": indexedExtremaDtype[Complex[float64]](cases)
      else: raise newException(ValueError, "unsupported indexed extrema input dtype")


proc extremaGradientNumber[T: SomeFloat](node: JsonNode): T =
  if node.kind == JString:
    case node.getStr
    of "nan": result = T(NaN)
    of "+inf": result = T(Inf)
    of "-inf": result = T(NegInf)
    of "-0": result = T(-0.0)
    else: raise newException(ValueError, "unknown extrema gradient special value")
  else:
    result = T(node.getFloat)


proc extremaGradientTensor[T: SomeFloat](shape: Metadata, values: JsonNode): Tensor[T] =
  result = newTensorUninit[T](shape)
  if values.kind != JArray or values.len != result.size:
    raise newException(ValueError, "extrema gradient tensor value count mismatch")
  var raw = result.unsafe_raw_offset
  for index in 0 ..< values.len:
    raw[index] = extremaGradientNumber[T](values[index])


proc extremaGradientInput[T: SomeFloat](spec: JsonNode): Tensor[T] =
  var source: Tensor[T]
  when T is float32:
    source = realDtypeFloat32Tensor(spec["x"])
  else:
    source = getTensor(spec["x"])
  extremaSpecials(source, spec["nonfinite"])
  case spec["layout"].getStr
  of "transpose", "offset", "reverse", "broadcast":
    source = realDtypeLayout(source, spec["layout"].getStr)
  of "permute":
    source = source.permute(2, 1, 0)
  of "identity": discard
  else: raise newException(ValueError, "unknown extrema gradient input layout")
  result = source


proc extremaGradientSeed[T: SomeFloat](shape: Metadata, spec: JsonNode): Tensor[T] =
  let kind = spec["kind"].getStr
  case kind
  of "constant":
    result = newTensorUninit[T](shape)
    let value = extremaGradientNumber[T](spec["value"])
    var raw = result.unsafe_raw_offset
    for index in 0 ..< result.size: raw[index] = value
  of "values":
    result = extremaGradientTensor[T](shape, spec["values"])
  of "strided_sequence":
    if shape.len notin [1, 2] or shape[^1] == 0:
      raise newException(ValueError, "invalid strided extrema gradient seed shape")
    var backingShape = shape
    backingShape[^1] *= 2
    var backing = newTensorUninit[T](backingShape)
    var raw = backing.unsafe_raw_offset
    for index in 0 ..< backing.size: raw[index] = T(index)
    if shape.len == 1:
      result = backing[0..^1|2]
    else:
      result = backing[_, 0..^1|2]
  else:
    raise newException(ValueError, "unknown extrema gradient seed kind")


proc extremaGradientObservation[T: SomeFloat](t: Tensor[T]): JsonNode =
  result = tensorObservation(t)
  var indices = newJArray()
  for index, value in enumerate(t):
    if value == T(0):
      when T is float32:
        if (cast[uint32](value) shr 31) == 1'u32: indices.add newJInt(index)
      else:
        if (cast[uint64](value) shr 63) == 1'u64: indices.add newJInt(index)
  result["negative_zero_indices"] = indices

proc pairedExtremaErrorCode(message: string): string =
  result = extremaErrorCode(message)
  if result in ["axis_out_of_range", "empty_reduced_axis", "unsupported_dtype"]:
    return
  let normalized = message.toLowerAscii
  if normalized.contains("axis out of range") or normalized.contains("dimension out of range"):
    result = "axis_out_of_range"
  elif normalized.contains("empty reduced axis") or normalized.contains("nonzero size"):
    result = "empty_reduced_axis"
  elif normalized.contains("unsupported") and normalized.contains("dtype"):
    result = "unsupported_dtype"


proc pairedExtremaObservation[T](source: Tensor[T], operation: string,
    axis: JsonNode, keepdim: bool): JsonNode =
  try:
    let dimension = axis.getInt
    let output =
      if operation == "min":
        source.minWithIndices(dimension, keepdim = keepdim)
      else:
        source.maxWithIndices(dimension, keepdim = keepdim)
    result = newJObject()
    when T is Complex[float32] or T is Complex[float64]:
      raise newException(ValueError, "paired extrema unexpectedly accepted complex input")
    elif T is SomeFloat:
      result["values"] = extremaGradientObservation(output.values)
    else:
      result["values"] = tensorObservation(output.values)
    result["indices"] = tensorObservation(output.indices)
  except IndexDefect as error:
    let code = pairedExtremaErrorCode(error.msg)
    if code notin ["axis_out_of_range", "empty_reduced_axis", "unsupported_dtype"]:
      raise
    result = %*{"error": code}
  except ValueError as error:
    let code = pairedExtremaErrorCode(error.msg)
    if code notin ["axis_out_of_range", "empty_reduced_axis", "unsupported_dtype"]:
      raise
    result = %*{"error": code}


proc pairedExtremaCase[T](spec: JsonNode): JsonNode =
  let source = indexedExtremaInputTensor[T](spec)
  result = newJObject()
  for operation in ["min", "max"]:
    for axis in spec["axes"]:
      for keepdim in [false, true]:
        let requestId = indexedExtremaRequestId(operation, axis, keepdim)
        result[requestId] = pairedExtremaObservation(source, operation, axis, keepdim)


proc pairedExtremaDtype[T](inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = pairedExtremaCase[T](spec)


proc pairedExtrema(inputs: JsonNode): JsonNode =
  result = newJObject()
  for dtype, cases in inputs:
    result[dtype] = case dtype
      of "bool": pairedExtremaDtype[bool](cases)
      of "int8": pairedExtremaDtype[int8](cases)
      of "int16": pairedExtremaDtype[int16](cases)
      of "int32": pairedExtremaDtype[int32](cases)
      of "int64": pairedExtremaDtype[int64](cases)
      of "uint8": pairedExtremaDtype[uint8](cases)
      of "uint16": pairedExtremaDtype[uint16](cases)
      of "uint32": pairedExtremaDtype[uint32](cases)
      of "uint64": pairedExtremaDtype[uint64](cases)
      of "float32": pairedExtremaDtype[float32](cases)
      of "float64": pairedExtremaDtype[float64](cases)
      of "complex64": pairedExtremaDtype[Complex[float32]](cases)
      of "complex128": pairedExtremaDtype[Complex[float64]](cases)
      else: raise newException(ValueError, "unsupported paired extrema input dtype")



proc extremaGradientCase[T: SomeFloat](spec: JsonNode): JsonNode =
  let source = extremaGradientInput[T](spec)
  result = newJArray()
  for request in spec["operations"]:
    let context = newContext Tensor[T]
    let x = context.variable(source, requires_grad = true)
    let dimensions = spec["axes"]
    let keepdim = request["keepdim"].getBool
    let output = case request["operation"].getStr
      of "amin":
        if dimensions.kind == JNull:
          x.amin(keepdim = keepdim)
        elif dimensions.kind == JInt:
          x.amin(dimensions.getInt, keepdim = keepdim)
        elif dimensions.kind == JArray:
          var axes = newSeq[int](dimensions.len)
          for index in 0 ..< dimensions.len: axes[index] = dimensions[index].getInt
          x.amin(axes, keepdim = keepdim)
        else:
          raise newException(ValueError, "invalid extrema gradient dimensions")
      of "amax":
        if dimensions.kind == JNull:
          x.amax(keepdim = keepdim)
        elif dimensions.kind == JInt:
          x.amax(dimensions.getInt, keepdim = keepdim)
        elif dimensions.kind == JArray:
          var axes = newSeq[int](dimensions.len)
          for index in 0 ..< dimensions.len: axes[index] = dimensions[index].getInt
          x.amax(axes, keepdim = keepdim)
        else:
          raise newException(ValueError, "invalid extrema gradient dimensions")
      else:
        raise newException(ValueError, "unknown extrema gradient operation")
    let seed = extremaGradientSeed[T](output.value.shape, spec["seed"])
    output.backprop(seed)
    result.add %*{
      "output": tensorObservation(output.value),
      "grad": extremaGradientObservation(x.grad),
    }


proc extremaGradientDtype[T: SomeFloat](inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = extremaGradientCase[T](spec)


proc extremaGradient(inputs: JsonNode): JsonNode =
  result = newJObject()
  result["float32"] = extremaGradientDtype[float32](inputs)
  result["float64"] = extremaGradientDtype[float64](inputs)

proc pairedExtremaGradientInput[T: SomeFloat](spec: JsonNode): Tensor[T] =
  var source = indexedExtremaFloatTensor[T](spec["x"])
  extremaSpecials(source, spec["nonfinite"])
  case spec["layout"].getStr
  of "transpose", "offset":
    source = realDtypeLayout(source, spec["layout"].getStr)
  of "reverse":
    source = source[_, ^1..0|-1]
  of "broadcast":
    source = source.broadcast(2, 3)
  of "identity": discard
  else: raise newException(ValueError, "unknown paired extrema gradient input layout")
  result = source


proc pairedExtremaGradientSeed[T: SomeFloat](shape: Metadata, kind: string): Tensor[T] =
  var count = 1
  for dimension in shape: count *= dimension
  if kind == "strided":
    if shape.len notin [1, 2] or shape[^1] <= 0:
      raise newException(ValueError, "invalid strided paired extrema gradient seed shape")
    var backingShape = shape
    backingShape[^1] *= 2
    var backing = newTensorUninit[T](backingShape)
    var raw = backing.unsafe_raw_offset
    for index in 0 ..< backing.size: raw[index] = T(0)
    let width = shape[^1]
    for index in 0 ..< count:
      let storageIndex =
        if shape.len == 1:
          index * 2 + 1
        else:
          (index div width) * (2 * width) + (index mod width) * 2 + 1
      raw[storageIndex] = T(index mod 5 - 2)
    if shape.len == 1:
      result = backing[1..^1|2]
    else:
      result = backing[_, 1..^1|2]
    return

  result = newTensorUninit[T](shape)
  var raw = result.unsafe_raw_offset
  for index in 0 ..< count:
    case kind
    of "finite": raw[index] = T(index mod 5 - 2)
    of "negative_zero": raw[index] = T(-0.0)
    of "nonfinite": raw[index] = if index mod 2 == 0: T(Inf) else: T(NaN)
    else: raise newException(ValueError, "unknown paired extrema gradient seed kind")


proc pairedExtremaGradientInvalidSeed[T: SomeFloat](shape: Metadata): Tensor[T] =
  var count = 1
  for dimension in shape: count *= dimension
  result = newTensorUninit[T](toMetadata(@[count + 1]))
  var raw = result.unsafe_raw_offset
  for index in 0 ..< result.size: raw[index] = T(0)


proc pairedExtremaGradientObservation[T: SomeFloat](source: Tensor[T],
    operation: string, axis: int, keepdim: bool, seedKind: string,
    invalidSeed = false): JsonNode =
  let context = newContext Tensor[T]
  let x = context.variable(source, requires_grad = true)
  let output =
    if operation == "min":
      x.minWithIndices(axis, keepdim = keepdim)
    else:
      x.maxWithIndices(axis, keepdim = keepdim)
  if invalidSeed:
    try:
      output.values.backprop(pairedExtremaGradientInvalidSeed[T](output.values.value.shape))
    except IndexDefect:
      return %*{"error": "seed_shape_mismatch"}
    except ValueError:
      return %*{"error": "seed_shape_mismatch"}
    raise newException(ValueError, "paired extrema accepted a mismatched gradient seed shape")
  let seed = pairedExtremaGradientSeed[T](output.values.value.shape, seedKind)
  output.values.backprop(seed)
  result = %*{
    "values": extremaGradientObservation(output.values.value),
    "indices": tensorObservation(output.indices),
    "seed": extremaGradientObservation(seed),
    "gradient": extremaGradientObservation(x.grad),
    "indices_require_grad": false,
  }


proc pairedExtremaGradientCase[T: SomeFloat](spec: JsonNode): JsonNode =
  let source = pairedExtremaGradientInput[T](spec)
  result = newJObject()
  for operation in ["min", "max"]:
    for axisNode in spec["axes"]:
      let axis = axisNode.getInt
      for keepdim in [false, true]:
        let requestId = indexedExtremaRequestId(operation, %axis, keepdim)
        for seedKindNode in spec["seed_kinds"]:
          let seedKind = seedKindNode.getStr
          result[requestId & "_seed_" & seedKind] =
            pairedExtremaGradientObservation(source, operation, axis, keepdim, seedKind)
        result[requestId & "_invalid_seed"] =
          pairedExtremaGradientObservation(source, operation, axis, keepdim, "", invalidSeed = true)


proc pairedExtremaGradientDtype[T: SomeFloat](inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    if spec["invalid_seed"]["kind"].getStr != "shape_mismatch":
      raise newException(ValueError, "unknown paired extrema invalid seed kind")
    result[name] = pairedExtremaGradientCase[T](spec)


proc pairedExtremaGradient(inputs: JsonNode): JsonNode =
  result = newJObject()
  result["float32"] = pairedExtremaGradientDtype[float32](inputs)
  result["float64"] = pairedExtremaGradientDtype[float64](inputs)




proc aminmaxHexWord(bits: uint64, digits: int): string =
  const HexDigits = "0123456789abcdef"
  result = newString(digits + 2)
  result[0] = '0'
  result[1] = 'x'
  var remaining = bits
  for index in 0 ..< digits:
    result[digits + 1 - index] = HexDigits[int(remaining and 0xf)]
    remaining = remaining shr 4


proc aminmaxParseHexBits(node: JsonNode, digits: int): uint64 =
  let value = node.getStr
  if value.len != digits + 2 or value[0 .. 1] != "0x":
    raise newException(ValueError, "invalid aminmax bit pattern")
  for index in 2 ..< value.len:
    var nibble = 0'u64
    case value[index]
    of '0' .. '9': nibble = uint64(ord(value[index]) - ord('0'))
    of 'a' .. 'f': nibble = uint64(ord(value[index]) - ord('a') + 10)
    else: raise newException(ValueError, "invalid aminmax bit pattern")
    result = (result shl 4) or nibble


proc aminmaxBitsTensor[T: SomeFloat](spec: JsonNode): Tensor[T] =
  let shapeNode = spec["shape"]
  if shapeNode.len > LASER_MAXRANK:
    raise newException(ValueError, "aminmax input rank exceeds LASER_MAXRANK")
  var shape = newSeq[int](shapeNode.len)
  for index in 0 ..< shapeNode.len: shape[index] = shapeNode[index].getInt
  result = newTensorUninit[T](toMetadata(shape))
  let bits = spec["bits"]
  if bits.len != result.size:
    raise newException(ValueError, "aminmax bit pattern count mismatch")
  var raw = result.unsafe_raw_offset
  for index in 0 ..< bits.len:
    when T is float32:
      raw[index] = cast[float32](uint32(aminmaxParseHexBits(bits[index], 8)))
    else:
      raw[index] = cast[float64](aminmaxParseHexBits(bits[index], 16))


proc aminmaxInputTensor[T](spec: JsonNode): Tensor[T] =
  when T is Complex[float32] or T is Complex[float64]:
    result = getComplexTensor[T](spec)
  elif T is float32:
    if spec.hasKey("bits"): result = aminmaxBitsTensor[float32](spec)
    else: result = realDtypeFloat32Tensor(spec)
  elif T is float64:
    if spec.hasKey("bits"):
      result = aminmaxBitsTensor[float64](spec)
    else:
      result = getTensor(spec)
  else:
    result = getIntegerTensor[T](spec)
  result = realDtypeLayout(result, spec["layout"].getStr)


proc aminmaxTensorObservation[T](tensor: Tensor[T]): JsonNode =
  when T is float32 or T is float64:
    result = tensorObservation(tensor)
    var bits = newJArray()
    let raw = tensor.unsafe_raw_offset
    for index in 0 ..< tensor.size:
      when T is float32:
        bits.add %aminmaxHexWord(uint64(cast[uint32](raw[index])), 8)
      else:
        bits.add %aminmaxHexWord(cast[uint64](raw[index]), 16)
    result["bits"] = bits
  elif T is Complex[float32] or T is Complex[float64]:
    if tensor.size != 0:
      raise newException(ValueError, "nonempty complex aminmax output was returned")
    result = newJObject()
    var shape = newJArray()
    for dimension in tensor.shape: shape.add newJInt(dimension)
    result["shape"] = shape
    result["numel"] = newJInt(tensor.size.int)
    when T is Complex[float32]: result["dtype"] = %"complex64"
    else: result["dtype"] = %"complex128"
    result["device"] = %"cpu"
    result["values"] = newJArray()
  else:
    result = tensorObservation(tensor)


proc aminmaxExpectedNativeError(spec, request: JsonNode,
                                shape: Metadata): string =
  let dtype = spec["dtype"].getStr
  if dtype in ["complex64", "complex128"]: return "unsupported_dtype"
  let dimensions = request["dim"]
  if dimensions.kind == JNull:
    for dimension in shape:
      if dimension == 0: return "empty_reduction"
    if dtype in ["uint16", "uint32", "uint64"]: return "unsupported_dtype"
    return ""
  let axis = dimensions.getInt
  let axisLimit = max(1, shape.len)
  if axis < -axisLimit or axis >= axisLimit: return "axis_out_of_range"
  if shape.len > 0:
    let normalized = if axis < 0: axis + shape.len else: axis
    if shape[normalized] == 0: return "empty_reduced_axis"
    if dtype in ["uint16", "uint32", "uint64"]: return "unsupported_dtype"

proc aminmaxOutput[T](source: Tensor[T], dim: JsonNode,
                      keepdim: bool): tuple[min: Tensor[T], max: Tensor[T]] =
  if dim.kind == JNull: result = source.aminmax(keepdim = keepdim)
  else: result = source.aminmax(dim.getInt, keepdim = keepdim)

proc aminmaxObservation[T](source: Tensor[T], spec, request: JsonNode): JsonNode =
  var output: tuple[min: Tensor[T], max: Tensor[T]]
  try:
    output = aminmaxOutput(source, request["dim"], request["keepdim"].getBool)
  except IndexDefect:
    let code = aminmaxExpectedNativeError(spec, request, source.shape)
    if code notin ["axis_out_of_range", "empty_reduced_axis"]: raise
    return %*{"error": code}
  except ValueError:
    let code = aminmaxExpectedNativeError(spec, request, source.shape)
    if code notin ["empty_reduction", "unsupported_dtype"]: raise
    return %*{"error": code}
  result = newJObject()
  result["min"] = aminmaxTensorObservation(output.min)
  result["max"] = aminmaxTensorObservation(output.max)
  let outputsAreDistinct = if output.min.size == 0: true
    else: cast[pointer](output.min.unsafe_raw_offset) !=
      cast[pointer](output.max.unsafe_raw_offset)
  result["distinct"] = %outputsAreDistinct



proc aminmaxCase[T](spec: JsonNode): JsonNode =
  let source = aminmaxInputTensor[T](spec)
  result = newJObject()
  for request in spec["requests"]:
    result[request["id"].getStr] = aminmaxObservation(source, spec, request)


proc aminmax(inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = case spec["dtype"].getStr
      of "bool": aminmaxCase[bool](spec)
      of "int8": aminmaxCase[int8](spec)
      of "int16": aminmaxCase[int16](spec)
      of "int32": aminmaxCase[int32](spec)
      of "int64": aminmaxCase[int64](spec)
      of "uint8": aminmaxCase[uint8](spec)
      of "uint16": aminmaxCase[uint16](spec)
      of "uint32": aminmaxCase[uint32](spec)
      of "uint64": aminmaxCase[uint64](spec)
      of "float32": aminmaxCase[float32](spec)
      of "float64": aminmaxCase[float64](spec)
      of "complex64": aminmaxCase[Complex[float32]](spec)
      of "complex128": aminmaxCase[Complex[float64]](spec)
      else: raise newException(ValueError, "unsupported aminmax dtype: " & spec["dtype"].getStr)


proc logsumexpInputTensor[T](spec: JsonNode): Tensor[T] =
  var source =
    when T is Complex[float32] or T is Complex[float64]: getComplexTensor[T](spec)
    elif T is float32: realDtypeFloat32Tensor(spec)
    elif T is float64: getTensor(spec)
    else: getIntegerTensor[T](spec)
  if spec["specials"].len > 0:
    when T is Complex[float32] or T is Complex[float64]:
      complexScanSpecials(source, spec["specials"])
    elif T is SomeFloat:
      realDtypeSpecials(source, spec["specials"])
    else:
      raise newException(ValueError, "integer logsumexp input cannot contain nonfinite values")
  result = realDtypeLayout(source, spec["layout"].getStr)


proc logsumexpExpectedNativeError[T](source: Tensor[T],
                                     request: JsonNode): string =
  let rank = source.shape.len
  let axisLimit = max(1, rank)
  let kind = request["kind"].getStr
  if kind == "axis":
    let axis = request["axis"].getInt
    if axis < -axisLimit or axis >= axisLimit: return "invalid_axis"
  elif kind == "axes":
    let axes = request["axes"]
    for index in 0 ..< axes.len:
      let axisNode = axes[index]
      let axis = axisNode.getInt
      if axis < -axisLimit or axis >= axisLimit: return "invalid_axis"
      let normalized = if axis < 0: axis + axisLimit else: axis
      for prior in 0 ..< index:
        let previous = axes[prior].getInt
        if (if previous < 0: previous + axisLimit else: previous) == normalized:
          return "duplicate_axes"
    if axes.len == 0 and not request["keepdim"].getBool and
        rank > 0 and source.size > 0:
      return "empty_axes_drop_mismatch"


proc logsumexpApply[T](source: Tensor[T], request: JsonNode): JsonNode =
  let kind = request["kind"].getStr
  let keepdim = request["keepdim"].getBool
  when T is bool or T is SomeInteger or T is SomeFloat:
    case kind
    of "whole":
      result = tensorObservation(source.logsumexp(keepdim = keepdim))
    of "axis":
      result = tensorObservation(source.logsumexp(
        axis = request["axis"].getInt, keepdim = keepdim))
    of "axes":
      var axes = newSeq[int](request["axes"].len)
      for index in 0 ..< axes.len: axes[index] = request["axes"][index].getInt
      result = tensorObservation(source.logsumexp(axes = axes, keepdim = keepdim))
    else:
      raise newException(ValueError, "unknown logsumexp request kind: " & kind)
  elif T is Complex[float32] or T is Complex[float64]:
    case kind
    of "whole":
      result = complexTensorObservation(source.logsumexp(keepdim = keepdim))
    of "axis":
      result = complexTensorObservation(source.logsumexp(
        axis = request["axis"].getInt, keepdim = keepdim))
    of "axes":
      var axes = newSeq[int](request["axes"].len)
      for index in 0 ..< axes.len: axes[index] = request["axes"][index].getInt
      result = complexTensorObservation(source.logsumexp(axes = axes, keepdim = keepdim))
    else:
      raise newException(ValueError, "unknown logsumexp request kind: " & kind)
  else:
    {.error: "unsupported logsumexp input dtype".}


proc logsumexpObservation[T](source: Tensor[T], request: JsonNode): JsonNode =
  try:
    return logsumexpApply(source, request)
  except IndexDefect:
    let code = logsumexpExpectedNativeError(source, request)
    if code != "invalid_axis": raise
    return %*{"error": code}
  except ValueError:
    let code = logsumexpExpectedNativeError(source, request)
    if code notin ["duplicate_axes", "empty_axes_drop_mismatch"]: raise
    return %*{"error": code}


proc logsumexpCase[T](spec: JsonNode): JsonNode =
  let source = logsumexpInputTensor[T](spec)
  result = newJObject()
  for request in spec["requests"]:
    result[request["id"].getStr] = logsumexpObservation(source, request)


proc logsumexp(inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    result[name] = case spec["dtype"].getStr
      of "bool": logsumexpCase[bool](spec)
      of "int8": logsumexpCase[int8](spec)
      of "int16": logsumexpCase[int16](spec)
      of "int32": logsumexpCase[int32](spec)
      of "int64": logsumexpCase[int64](spec)
      of "uint8": logsumexpCase[uint8](spec)
      of "uint16": logsumexpCase[uint16](spec)
      of "uint32": logsumexpCase[uint32](spec)
      of "uint64": logsumexpCase[uint64](spec)
      of "float32": logsumexpCase[float32](spec)
      of "float64": logsumexpCase[float64](spec)
      of "complex64": logsumexpCase[Complex[float32]](spec)
      of "complex128": logsumexpCase[Complex[float64]](spec)
      else: raise newException(ValueError, "unsupported logsumexp dtype: " & spec["dtype"].getStr)


proc logsumexpGradientZero[T: SomeFloat or Complex](): T =
  when T is Complex[float32]: result = complex32(0'f32, 0'f32)
  elif T is Complex[float64]: result = complex64(0.0, 0.0)
  else: result = T(0)


proc logsumexpGradientSeedValue[T: SomeFloat or Complex](index: int,
    kind: string): T =
  when T is Complex:
    type Component = typeof(default(T).re)
    case kind
    of "finite": result = T(re: Component(2), im: Component(-0.5))
    of "negative_zero": result = T(re: Component(-0.0), im: Component(-0.0))
    of "positive_infinity": result = T(re: Component(Inf), im: Component(0))
    of "nan": result = T(re: Component(NaN), im: Component(0))
    of "ramp", "strided":
      result = T(re: Component(index mod 5 - 2),
        im: Component(index mod 3 - 1) * Component(0.5))
    else: raise newException(ValueError, "unknown logsumexp gradient seed kind")
  else:
    case kind
    of "finite": result = T(2)
    of "negative_zero": result = T(-0.0)
    of "positive_infinity": result = T(Inf)
    of "nan": result = T(NaN)
    of "ramp", "strided": result = T(index mod 5 - 2)
    else: raise newException(ValueError, "unknown logsumexp gradient seed kind")


proc logsumexpGradientSeed[T: SomeFloat or Complex](shape: Metadata,
    kind: string): Tensor[T] =
  var count = 1
  for dimension in shape: count *= dimension
  if kind == "strided":
    if shape.len notin [1, 2] or shape[^1] <= 0:
      raise newException(ValueError, "invalid strided logsumexp gradient seed shape")
    var backingShape = shape
    backingShape[^1] *= 2
    var backing = newTensorUninit[T](backingShape)
    var raw = backing.unsafe_raw_offset
    for index in 0 ..< backing.size: raw[index] = logsumexpGradientZero[T]()
    let width = shape[^1]
    for index in 0 ..< count:
      let storageIndex =
        if shape.len == 1:
          index * 2 + 1
        else:
          (index div width) * (2 * width) + (index mod width) * 2 + 1
      raw[storageIndex] = logsumexpGradientSeedValue[T](index, kind)
    if shape.len == 1: result = backing[1..^1|2]
    else: result = backing[_, 1..^1|2]
    return

  result = newTensorUninit[T](shape)
  var raw = result.unsafe_raw_offset
  for index in 0 ..< count:
    raw[index] = logsumexpGradientSeedValue[T](index, kind)


proc logsumexpGradientInvalidSeed[T: SomeFloat or Complex](
    shape: Metadata): Tensor[T] =
  var count = 1
  for dimension in shape: count *= dimension
  result = newTensorUninit[T](toMetadata(@[count + 1]))
  var raw = result.unsafe_raw_offset
  for index in 0 ..< result.size: raw[index] = logsumexpGradientZero[T]()


proc logsumexpGradientApply[T: SomeFloat or Complex](
    input: Variable[Tensor[T]], request: JsonNode): Variable[Tensor[T]] =
  let kind = request["kind"].getStr
  let keepdim = request["keepdim"].getBool
  case kind
  of "whole":
    result = input.logsumexp(keepdim = keepdim)
  of "axis":
    result = input.logsumexp(axis = request["axis"].getInt, keepdim = keepdim)
  of "axes":
    var axes = newSeq[int](request["axes"].len)
    for index in 0 ..< axes.len: axes[index] = request["axes"][index].getInt
    result = input.logsumexp(axes = axes, keepdim = keepdim)
  else:
    raise newException(ValueError, "unknown logsumexp gradient request kind")


proc logsumexpGradientForward[T: SomeFloat or Complex](
    input: Variable[Tensor[T]], source: Tensor[T],
    request: JsonNode): tuple[output: Variable[Tensor[T]], error: string] =
  try:
    result.output = logsumexpGradientApply(input, request)
  except IndexDefect:
    result.error = logsumexpExpectedNativeError(source, request)
    if result.error != "invalid_axis": raise
  except ValueError:
    result.error = logsumexpExpectedNativeError(source, request)
    if result.error notin ["duplicate_axes", "empty_axes_drop_mismatch"]: raise


proc logsumexpGradientTensorObservation[T: SomeFloat or Complex](
    tensor: Tensor[T]): JsonNode =
  when T is SomeFloat: result = extremaGradientObservation(tensor)
  elif T is Complex[float32] or T is Complex[float64]:
    result = complexTensorObservation(tensor)
  else:
    {.error: "unsupported logsumexp gradient observation dtype".}


proc logsumexpGradientObservation[T: SomeFloat or Complex](
    source: Tensor[T], request: JsonNode, seedKind: string,
    invalidSeed = false): JsonNode =
  let context = newContext Tensor[T]
  let input = context.variable(source, requires_grad = true)
  let attempt = logsumexpGradientForward(input, source, request)
  if attempt.error.len > 0: return %*{"error": attempt.error}
  let output = attempt.output
  if invalidSeed:
    try:
      output.backprop(logsumexpGradientInvalidSeed[T](output.value.shape))
    except IndexDefect:
      return %*{"error": "seed_shape_mismatch"}
    except ValueError:
      return %*{"error": "seed_shape_mismatch"}
    raise newException(ValueError, "logsumexp accepted a mismatched gradient seed shape")
  let seed = logsumexpGradientSeed[T](output.value.shape, seedKind)
  output.backprop(seed)
  result = %*{
    "output": logsumexpGradientTensorObservation(output.value),
    "seed": logsumexpGradientTensorObservation(seed),
    "gradient": logsumexpGradientTensorObservation(input.grad),
  }


proc logsumexpGradientCase[T: SomeFloat or Complex](spec: JsonNode): JsonNode =
  let source = logsumexpInputTensor[T](spec)
  result = newJObject()
  for request in spec["requests"]:
    let requestId = request["id"].getStr
    if logsumexpExpectedNativeError(source, request).len > 0:
      result[requestId] = logsumexpGradientObservation(source, request, "")
    else:
      for seedKindNode in request["seed_kinds"]:
        let seedKind = seedKindNode.getStr
        result[requestId & "_seed_" & seedKind] =
          logsumexpGradientObservation(source, request, seedKind)
      result[requestId & "_invalid_seed"] =
        logsumexpGradientObservation(source, request, "", invalidSeed = true)


proc logsumexpGradientDtype[T: SomeFloat or Complex](inputs: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in inputs:
    if spec["invalid_seed"]["kind"].getStr != "shape_mismatch":
      raise newException(ValueError, "unknown logsumexp gradient invalid seed kind")
    result[name] = logsumexpGradientCase[T](spec)


proc logsumexpGradient(inputs: JsonNode): JsonNode =
  result = newJObject()
  result["float32"] = logsumexpGradientDtype[float32](inputs["float32"])
  result["float64"] = logsumexpGradientDtype[float64](inputs["float64"])
  result["complex64"] = logsumexpGradientDtype[Complex[float32]](inputs["complex64"])
  result["complex128"] = logsumexpGradientDtype[Complex[float64]](inputs["complex128"])

proc logcumsumexpDtype[T](profiles: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in profiles:
    let source = logsumexpInputTensor[T](spec)
    var observations = newJObject()
    for request in spec["requests"]:
      let id = request["id"].getStr
      let axis = request["axis"].getInt
      let axisLimit = max(1, source.shape.len)
      let invalidAxis = axis < -axisLimit or axis >= axisLimit
      try:
        let output = source.logcumsumexp(axis)
        when T is SomeFloat or T is Complex:
          observations[id] = logsumexpGradientTensorObservation(output)
        else:
          observations[id] = tensorObservation(output)
      except IndexDefect:
        if not invalidAxis: raise
        observations[id] = %*{"error": "invalid_axis"}
      except ValueError:
        when T is SomeFloat or T is Complex:
          raise
        else:
          if invalidAxis: raise
          observations[id] = %*{"error": "unsupported_dtype"}
    result[name] = observations


proc logcumsumexp(inputs: JsonNode): JsonNode =
  result = newJObject()
  result["bool"] = logcumsumexpDtype[bool](inputs["bool"])
  result["int8"] = logcumsumexpDtype[int8](inputs["int8"])
  result["int16"] = logcumsumexpDtype[int16](inputs["int16"])
  result["int32"] = logcumsumexpDtype[int32](inputs["int32"])
  result["int64"] = logcumsumexpDtype[int64](inputs["int64"])
  result["uint8"] = logcumsumexpDtype[uint8](inputs["uint8"])
  result["uint16"] = logcumsumexpDtype[uint16](inputs["uint16"])
  result["uint32"] = logcumsumexpDtype[uint32](inputs["uint32"])
  result["uint64"] = logcumsumexpDtype[uint64](inputs["uint64"])
  result["float32"] = logcumsumexpDtype[float32](inputs["float32"])
  result["float64"] = logcumsumexpDtype[float64](inputs["float64"])
  result["complex64"] = logcumsumexpDtype[Complex[float32]](inputs["complex64"])
  result["complex128"] = logcumsumexpDtype[Complex[float64]](inputs["complex128"])


proc logcumsumexpGradientSeedValue[T: SomeFloat or Complex](index: int,
    kind: string): T =
  if kind == "zero": return logsumexpGradientZero[T]()
  when T is Complex:
    type Component = typeof(default(T).re)
    if kind == "negative_infinity":
      return T(re: Component(-Inf), im: Component(0))
    if kind == "imaginary_infinity":
      return T(re: Component(1), im: Component(Inf))
  else:
    if kind == "negative_infinity": return T(-Inf)
  let reductionKind = case kind
    of "mixed": "ramp"
    of "infinity": "positive_infinity"
    else: kind
  logsumexpGradientSeedValue[T](index, reductionKind)


proc logcumsumexpGradientSeed[T: SomeFloat or Complex](shape: Metadata,
    kind: string): Tensor[T] =
  ## Explicit view metadata keeps offset/step-two seeds strided at every rank.
  var count = 1
  for dimension in shape: count *= dimension
  if kind == "strided":
    var backing = newTensor[T](count * 2 + 1)
    for index in 0 ..< count:
      backing.unsafe_raw_offset[index * 2 + 1] =
        logcumsumexpGradientSeedValue[T](index, kind)
    result = backing
    result.offset += 1
    result.shape = shape
    result.strides.len = shape.len
    var stride = 2
    for dimension in countdown(int(shape.len) - 1, 0):
      result.strides[dimension] = stride
      stride *= shape[dimension]
    return
  result = newTensorUninit[T](shape)
  for index in 0 ..< count:
    result.unsafe_raw_offset[index] = logcumsumexpGradientSeedValue[T](index, kind)


proc logcumsumexpGradientObservation[T: SomeFloat or Complex](
    source: Tensor[T], axis: int, seedKind: string, invalidSeed = false): JsonNode =
  let context = newContext Tensor[T]
  let input = context.variable(source, requires_grad = true)
  let axisLimit = max(1, source.rank)
  let invalidAxis = axis < -axisLimit or axis >= axisLimit
  var output: Variable[Tensor[T]]
  try:
    output = input.logcumsumexp(axis)
  except IndexDefect:
    if not invalidAxis: raise
    return %*{"error": "invalid_axis"}
  if invalidAxis:
    raise newException(ValueError, "logcumsumexp accepted an invalid gradient axis")
  if invalidSeed:
    try:
      output.backprop(logsumexpGradientInvalidSeed[T](output.value.shape))
    except ValueError as error:
      if error.msg != "backprop gradient shape must match the output shape": raise
      return %*{"error": "seed_shape_mismatch"}
    raise newException(ValueError, "logcumsumexp accepted a mismatched gradient seed shape")
  let seed = logcumsumexpGradientSeed[T](output.value.shape, seedKind)
  output.backprop(seed)
  %*{
    "output": logsumexpGradientTensorObservation(output.value),
    "seed": logsumexpGradientTensorObservation(seed),
    "gradient": logsumexpGradientTensorObservation(input.grad),
  }


proc logcumsumexpGradientDtype[T: SomeFloat or Complex](profiles: JsonNode): JsonNode =
  result = newJObject()
  for name, spec in profiles:
    if spec["invalid_seed"]["kind"].getStr != "shape_mismatch":
      raise newException(ValueError, "unknown logcumsumexp gradient invalid seed kind")
    let source = logsumexpInputTensor[T](spec)
    var observations = newJObject()
    for request in spec["requests"]:
      let id = request["id"].getStr
      let axis = request["axis"].getInt
      for seedKindNode in request["seed_kinds"]:
        let seedKind = seedKindNode.getStr
        observations[id & "_seed_" & seedKind] =
          logcumsumexpGradientObservation(source, axis, seedKind)
      let axisLimit = max(1, source.rank)
      if axis >= -axisLimit and axis < axisLimit:
        observations[id & "_invalid_seed"] =
          logcumsumexpGradientObservation(source, axis, "", invalidSeed = true)
    result[name] = observations


proc logcumsumexpGradient(inputs: JsonNode): JsonNode =
  result = newJObject()
  result["float32"] = logcumsumexpGradientDtype[float32](inputs["float32"])
  result["float64"] = logcumsumexpGradientDtype[float64](inputs["float64"])
  result["complex64"] = logcumsumexpGradientDtype[Complex[float32]](inputs["complex64"])
  result["complex128"] = logcumsumexpGradientDtype[Complex[float64]](inputs["complex128"])



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
  of "tensor.extrema":
    result = extrema(inputs)
  of "tensor.indexed_extrema":
    result = indexedExtrema(inputs)
  of "tensor.paired_extrema":
    result = pairedExtrema(inputs)
  of "tensor.aminmax":
    result = aminmax(inputs)
  of "tensor.logsumexp":
    result = logsumexp(inputs)
  of "tensor.logcumsumexp":
    result = logcumsumexp(inputs)
  of "tensor.integer_output_dtype":
    result = integerOutputDtype(inputs)
  of "tensor.mean_dtype":
    result = realDtype(inputs, "mean")
  of "tensor.sum_dtype":
    result = realDtype(inputs, "sum")
  of "tensor.prod_dtype":
    result = realDtype(inputs, "prod")
  of "tensor.cumsum_dtype":
    result = realDtypeScans(inputs, "cumsum")
  of "tensor.cumprod_dtype":
    result = realDtypeScans(inputs, "cumprod")
  of "tensor.complex_scans":
    result = complexScans(inputs)
  of "tensor.complex_scan_dtype":
    result = complexScanDtype(inputs)
  of "tensor.complex_mean_dtype":
    result = complexMeanDtype(inputs)
  of "tensor.complex_sum_dtype":
    result = complexSumDtype(inputs)
  of "tensor.complex_prod_dtype":
    result = complexProductDtype(inputs)
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
  of "autograd.extrema":
    result = extremaGradient(inputs)
  of "autograd.paired_extrema":
    result = pairedExtremaGradient(inputs)
  of "autograd.logsumexp":
    result = logsumexpGradient(inputs)
  of "autograd.logcumsumexp":
    result = logcumsumexpGradient(inputs)
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
