# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating CPU logcumsumexp forward and explicitly seeded backward actions.
import std/[algorithm, complex, json, math, monotimes, os, strutils, times]
import arraymancer

const
  rows = 128
  columns = 128
  elementCount = rows * columns
  cubeShape = [16, 32, 32]
  warmupCount = 50
  repeatCount = 100
  callsPerSample = 50
  profiles = [
    "rows", "rows_head", "rows_tail", "columns",
    "cube_axis0", "cube_axis1", "cube_axis2", "vector",
  ]
  actions = [
    "rows", "rows_head", "rows_tail", "columns",
    "cube_axis0", "cube_axis1", "cube_axis2", "vector",
  ]
  manifestFields = [
    "input_dtype", "output_dtype", "rows", "columns", "shape", "values",
    "cube_shape", "cube_values", "vector_shape", "vector_values",
    "rows_head_values", "rows_tail_values", "profiles", "actions", "seeds",
    "warmup", "repeats", "iterations",
  ]

type
  BoundAction[T] = object
    input: Tensor[T]
    seed: Tensor[T]
    axis: int
  TrainingResult[T] = tuple[output: Tensor[T], gradient: Tensor[T]]

proc dtypeName[T](): string =
  when T is float32: "float32"
  elif T is float64: "float64"
  elif T is Complex[float32]: "complex64"
  elif T is Complex[float64]: "complex128"
  else: {.error: "Unsupported logcumsumexp gradient benchmark dtype".}

proc expectedValue[T](index, profile: int): T =
  let column = index mod columns
  when T is Complex:
    type Component = typeof(default(T).re)
    if profile == 1:
      if column == 0: return T(re: Component(2.0), im: Component(0.125))
      if column == columns - 1: return T(re: Component(-2.0), im: Component(-0.125))
    elif profile == 2:
      if column == 0: return T(re: Component(-2.0), im: Component(-0.125))
      if column == columns - 1: return T(re: Component(2.0), im: Component(0.125))
    T(re: Component(float64(index mod 31 - 15) / 16.0),
      im: Component(float64(index mod 9 - 4) / 64.0))
  else:
    if profile == 1:
      if column == 0: return T(4.0)
      if column == columns - 1: return T(-4.0)
    elif profile == 2:
      if column == 0: return T(-4.0)
      if column == columns - 1: return T(4.0)
    T(float64(index mod 63 - 31) / 8.0)

proc expectedSeed[T](index: int): T =
  when T is Complex:
    type Component = typeof(default(T).re)
    T(re: Component(index mod 5 - 2),
      im: Component(float64(index mod 3 - 1) / 4.0))
  else:
    T(index mod 5 - 2)

proc numberValue(node: JsonNode): float64 =
  case node.kind
  of JInt: float64(node.getInt)
  of JFloat: node.getFloat
  else: raise newException(ValueError, "Expected an exact JSON number")

proc loadMatrix[T](values: JsonNode; label: string; profile: int): Tensor[T] =
  if values.kind != JArray or values.len != elementCount:
    raise newException(ValueError, label & " must contain exactly 16,384 values")
  result = newTensorUninit[T](rows, columns)
  let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
  for index in 0 ..< elementCount:
    var value: T
    when T is Complex:
      let pair = values[index]
      if pair.kind != JArray or pair.len != 2 or
          pair[0].kind notin {JInt, JFloat} or pair[1].kind notin {JInt, JFloat}:
        raise newException(ValueError, label & " complex values must be [real, imag] JSON number pairs")
      type Component = typeof(default(T).re)
      value = T(re: Component(numberValue(pair[0])), im: Component(numberValue(pair[1])))
    else:
      if values[index].kind notin {JInt, JFloat}:
        raise newException(ValueError, label & " floating values must be JSON numbers")
      value = T(numberValue(values[index]))
    if value != expectedValue[T](index, profile):
      raise newException(ValueError, label & " values do not match the deterministic typed profile")
    data[index] = value

proc loadSeed[T](seedNode: JsonNode; expected: Metadata; dtype, name: string): Tensor[T] =
  if seedNode.kind != JObject or seedNode.len != 3 or
      not seedNode.hasKey("dtype") or not seedNode.hasKey("shape") or not seedNode.hasKey("values") or
      seedNode["dtype"].kind != JString or seedNode["dtype"].getStr != dtype:
    raise newException(ValueError, "Malformed seed record for " & name)
  let shapeNode = seedNode["shape"]
  if shapeNode.kind != JArray or shapeNode.len != expected.len:
    raise newException(ValueError, name & " seed shape does not match input shape")
  for index in 0 ..< expected.len:
    if shapeNode[index].kind != JInt or shapeNode[index].getInt != expected[index]:
      raise newException(ValueError, name & " seed shape does not match input shape")
  let values = seedNode["values"]
  result = newTensorUninit[T](expected)
  if values.kind != JArray or values.len != result.size:
    raise newException(ValueError, name & " seed values do not match its shape")
  let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
  for index in 0 ..< result.size:
    var value: T
    when T is Complex:
      let pair = values[index]
      if pair.kind != JArray or pair.len != 2 or
          pair[0].kind notin {JInt, JFloat} or pair[1].kind notin {JInt, JFloat}:
        raise newException(ValueError, name & " complex seed values must be [real, imag] number pairs")
      type Component = typeof(default(T).re)
      value = T(re: Component(numberValue(pair[0])), im: Component(numberValue(pair[1])))
    else:
      if values[index].kind notin {JInt, JFloat}:
        raise newException(ValueError, name & " floating seed values must be JSON numbers")
      value = T(numberValue(values[index]))
    if value != expectedSeed[T](index):
      raise newException(ValueError, name & " seed does not match the deterministic signed sequence")
    when T is Complex:
      if value.re.classify in {fcNan, fcInf, fcNegInf} or
          value.im.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, name & " seed values must be finite")
    else:
      if value.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, name & " seed values must be finite")
    data[index] = value

proc bindAction[T](name, dtype: string; base, cube, rowsHead, rowsTail,
                   vector: Tensor[T]; seedNode: JsonNode): BoundAction[T] =
  case name
  of "rows":
    result.input = base
    result.axis = 1
  of "rows_head":
    result.input = rowsHead
    result.axis = 1
  of "rows_tail":
    result.input = rowsTail
    result.axis = 1
  of "columns":
    result.input = base
    result.axis = 0
  of "cube_axis0":
    result.input = cube
    result.axis = 0
  of "cube_axis1":
    result.input = cube
    result.axis = 1
  of "cube_axis2":
    result.input = cube
    result.axis = 2
  of "vector":
    result.input = vector
    result.axis = 0
  else:
    raise newException(ValueError, "Unknown logcumsumexp gradient action: " & name)
  result.seed = loadSeed[T](seedNode, result.input.shape, dtype, name)

proc execute[T](action: BoundAction[T]): TrainingResult[T] =
  ## Every call owns a fresh context and leaf; inputs and seeds are resident.
  let context = newContext(Tensor[T])
  let variable = context.variable(action.input, requires_grad = true)
  let output = variable.logcumsumexp(action.axis)
  output.backprop(action.seed)
  result = (output.value, variable.grad)

proc shapeJson[T](value: Tensor[T]): JsonNode =
  result = newJArray()
  for dimension in value.shape:
    result.add %dimension

proc jsonFloat(value: float64): JsonNode =
  if value.isNaN: newJString("NaN")
  elif value == Inf: newJString("+Inf")
  elif value == NegInf: newJString("-Inf")
  else: %value

proc valuesJson[T](value: Tensor[T]): JsonNode =
  result = newJArray()
  for item in value:
    when T is Complex:
      result.add %*{"real": jsonFloat(float64(item.re)),
                    "imag": jsonFloat(float64(item.im))}
    else:
      result.add jsonFloat(float64(item))

proc sameFloat[T: SomeFloat](left, right: T): bool =
  let leftClass = left.classify
  let rightClass = right.classify
  if leftClass == fcNan: return rightClass == fcNan
  if rightClass == fcNan: return false
  if left != right: return false
  if left == T(0):
    when T is float32:
      return (cast[uint32](left) shr 31) == (cast[uint32](right) shr 31)
    else:
      return (cast[uint64](left) shr 63) == (cast[uint64](right) shr 63)
  true

proc sameValue[T](left, right: T): bool =
  when T is Complex:
    sameFloat(left.re, right.re) and sameFloat(left.im, right.im)
  else:
    sameFloat(left, right)

proc sameValues[T](left, right: Tensor[T]): bool =
  if left.shape != right.shape or left.size != right.size:
    return false
  let leftData = cast[ptr UncheckedArray[T]](left.get_offset_ptr)
  let rightData = cast[ptr UncheckedArray[T]](right.get_offset_ptr)
  for index in 0 ..< left.size:
    if not sameValue(leftData[index], rightData[index]):
      return false
  true

proc addressRange[T](value: Tensor[T]): tuple[first, last: uint] =
  result.first = cast[uint](value.get_offset_ptr)
  result.last = result.first + uint(value.size) * uint(sizeof(T))

proc rangesDisjoint[LeftT, RightT](left: Tensor[LeftT]; right: Tensor[RightT]): bool =
  let leftRange = addressRange(left)
  let rightRange = addressRange(right)
  leftRange.last <= rightRange.first or rightRange.last <= leftRange.first

proc disjointFromInputs[OutputT, InputT](value: Tensor[OutputT];
    base, rowsHead, rowsTail, cube, vector: Tensor[InputT]): bool =
  rangesDisjoint(value, base) and rangesDisjoint(value, rowsHead) and
    rangesDisjoint(value, rowsTail) and rangesDisjoint(value, cube) and
    rangesDisjoint(value, vector)

proc disjointFromSeeds[T](value: Tensor[T]; actions: seq[BoundAction[T]]): bool =
  for action in actions:
    if not rangesDisjoint(value, action.seed):
      return false
  true

proc allSeedsIndependent[T](actions: seq[BoundAction[T]]): bool =
  for leftIndex in 0 ..< actions.len:
    for rightIndex in leftIndex + 1 ..< actions.len:
      if not rangesDisjoint(actions[leftIndex].seed, actions[rightIndex].seed):
        return false
  true

proc observe[T](action: BoundAction[T]; output, repeated: TrainingResult[T];
    base, rowsHead, rowsTail, cube, vector: Tensor[T];
    allActions: seq[BoundAction[T]]): JsonNode =
  let expected = action.input.shape
  if output.output.shape != expected or repeated.output.shape != expected or
      output.gradient.shape != expected or repeated.gradient.shape != expected or
      not output.output.is_C_contiguous or not repeated.output.is_C_contiguous or
      not output.gradient.is_C_contiguous or not repeated.gradient.is_C_contiguous:
    raise newException(ValueError, "logcumsumexp gradient output dtype, shape, or contiguous layout mismatch")
  if not sameValues(output.output, repeated.output) or
      not sameValues(output.gradient, repeated.gradient):
    raise newException(ValueError, "Repeated logcumsumexp gradient outputs differ, including signed-zero or nonfinite categories")
  let outputDisjoint = disjointFromInputs(output.output, base, rowsHead, rowsTail, cube, vector)
  let gradientDisjoint = disjointFromInputs(output.gradient, base, rowsHead, rowsTail, cube, vector)
  let outputSeedsDisjoint = disjointFromSeeds(output.output, allActions)
  let gradientSeedsDisjoint = disjointFromSeeds(output.gradient, allActions)
  let outputGradientDisjoint = rangesDisjoint(output.output, output.gradient)
  let outputRepeatDisjoint = rangesDisjoint(output.output, repeated.output)
  let gradientRepeatDisjoint = rangesDisjoint(output.gradient, repeated.gradient)
  let repeatCrossDisjoint = rangesDisjoint(output.output, repeated.gradient) and
    rangesDisjoint(output.gradient, repeated.output)
  let outputRepeatedGradientDisjoint = rangesDisjoint(repeated.output, repeated.gradient)
  let repeatedOutputDisjoint = disjointFromInputs(repeated.output, base, rowsHead, rowsTail, cube, vector)
  let repeatedGradientDisjoint = disjointFromInputs(repeated.gradient, base, rowsHead, rowsTail, cube, vector)
  let repeatedOutputSeedsDisjoint = disjointFromSeeds(repeated.output, allActions)
  let repeatedGradientSeedsDisjoint = disjointFromSeeds(repeated.gradient, allActions)
  if not outputDisjoint or not gradientDisjoint or not outputSeedsDisjoint or
      not gradientSeedsDisjoint or not outputGradientDisjoint or not outputRepeatDisjoint or
      not gradientRepeatDisjoint or not repeatCrossDisjoint or not outputRepeatedGradientDisjoint or
      not repeatedOutputDisjoint or not repeatedGradientDisjoint or
      not repeatedOutputSeedsDisjoint or not repeatedGradientSeedsDisjoint:
    raise newException(ValueError, "logcumsumexp gradient outputs must be independent allocations disjoint from every input and seed")
  %*{
    "input_dtype": dtypeName[T](),
    "output_dtype": dtypeName[T](),
    "gradient_dtype": dtypeName[T](),
    "output_shape": shapeJson(output.output),
    "gradient_shape": shapeJson(output.gradient),
    "output_values": valuesJson(output.output),
    "gradient_values": valuesJson(output.gradient),
    "output_data_ptr": $cast[uint](output.output.get_offset_ptr),
    "gradient_data_ptr": $cast[uint](output.gradient.get_offset_ptr),
    "output_is_contiguous": output.output.is_C_contiguous,
    "gradient_is_contiguous": output.gradient.is_C_contiguous,
    "output_disjoint_from_inputs": outputDisjoint,
    "gradient_disjoint_from_inputs": gradientDisjoint,
    "output_disjoint_from_seeds": outputSeedsDisjoint,
    "gradient_disjoint_from_seeds": gradientSeedsDisjoint,
    "output_gradient_disjoint": outputGradientDisjoint,
    "output_distinct_from_repeat": outputRepeatDisjoint,
    "gradient_distinct_from_repeat": gradientRepeatDisjoint,
    "output_repeated_values_match": true,
    "gradient_repeated_values_match": true,
  }

proc consume[T](output: TrainingResult[T]) =
  ## Keep both forward and full input-gradient allocations observable without extra work.
  let outputAddress = cast[uint](output.output.get_offset_ptr)
  let gradientAddress = cast[uint](output.gradient.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`outputAddress`), "r"(`gradientAddress`)
    : "memory"
  """

proc run[T](manifest: JsonNode; preflightOnly: bool): JsonNode =
  if manifest.kind != JObject or manifest.len != manifestFields.len:
    raise newException(ValueError, "Logcumsumexp gradient manifest has missing or unknown fields")
  for field in manifestFields:
    if not manifest.hasKey(field):
      raise newException(ValueError, "Logcumsumexp gradient manifest has missing or unknown fields")
  if manifest["input_dtype"].kind != JString or manifest["input_dtype"].getStr != dtypeName[T]() or
      manifest["output_dtype"].kind != JString or manifest["output_dtype"].getStr != dtypeName[T]():
    raise newException(ValueError, "Logcumsumexp gradient dtype fields do not match dispatched types")
  if manifest["rows"].kind != JInt or manifest["columns"].kind != JInt or
      manifest["rows"].getInt != rows or manifest["columns"].getInt != columns or
      manifest["shape"] != %*[rows, columns] or manifest["cube_shape"] != %*[16, 32, 32] or
      manifest["vector_shape"] != %*[elementCount]:
    raise newException(ValueError, "Logcumsumexp gradient input shape metadata mismatch")
  if manifest["cube_values"] != manifest["values"] or
      manifest["vector_values"] != manifest["values"]:
    raise newException(ValueError, "Logcumsumexp gradient cube/vector values must equal base matrix values")
  for (field, expected) in [("warmup", warmupCount), ("repeats", repeatCount),
                            ("iterations", callsPerSample)]:
    if manifest[field].kind != JInt or manifest[field].getInt != expected:
      raise newException(ValueError, "Logcumsumexp gradient timing protocol mismatch")
  let suppliedProfiles = manifest["profiles"]
  if suppliedProfiles.kind != JArray or suppliedProfiles.len != profiles.len:
    raise newException(ValueError, "Logcumsumexp gradient profile manifest mismatch")
  for index, profile in profiles:
    if suppliedProfiles[index].kind != JString or suppliedProfiles[index].getStr != profile:
      raise newException(ValueError, "Logcumsumexp gradient ordered profile manifest mismatch")
  let suppliedActions = manifest["actions"]
  if suppliedActions.kind != JArray or suppliedActions.len != actions.len:
    raise newException(ValueError, "Logcumsumexp gradient manifest must contain exactly 8 actions")
  for index, action in actions:
    if suppliedActions[index].kind != JString or suppliedActions[index].getStr != action:
      raise newException(ValueError, "Logcumsumexp gradient ordered action manifest mismatch")

  let base = loadMatrix[T](manifest["values"], "values", 0)
  let rowsHead = loadMatrix[T](manifest["rows_head_values"], "rows_head_values", 1)
  let rowsTail = loadMatrix[T](manifest["rows_tail_values"], "rows_tail_values", 2)
  let cube = base.reshape(cubeShape[0], cubeShape[1], cubeShape[2])
  let vector = base.reshape(elementCount)
  if not base.is_C_contiguous or not rowsHead.is_C_contiguous or not rowsTail.is_C_contiguous or
      not cube.is_C_contiguous or not vector.is_C_contiguous or
      not rangesDisjoint(base, rowsHead) or not rangesDisjoint(base, rowsTail) or
      not rangesDisjoint(rowsHead, rowsTail) or
      cast[uint](cube.get_offset_ptr) != cast[uint](base.get_offset_ptr) or
      cast[uint](vector.get_offset_ptr) != cast[uint](base.get_offset_ptr) or
      cube.size != base.size or vector.size != base.size:
    raise newException(ValueError, "Logcumsumexp gradient inputs must be independent resident matrices with base-sharing cube/vector views")
  let seeds = manifest["seeds"]
  if seeds.kind != JObject or seeds.len != actions.len:
    raise newException(ValueError, "Logcumsumexp gradient seed manifest must contain exactly 8 seeds")
  var boundActions = newSeq[BoundAction[T]](actions.len)
  for index, name in actions:
    if not seeds.hasKey(name):
      raise newException(ValueError, "Missing seed for " & name)
    boundActions[index] = bindAction(
      name, dtypeName[T](), base, cube, rowsHead, rowsTail, vector, seeds[name])
  for action in boundActions:
    if not disjointFromInputs(action.seed, base, rowsHead, rowsTail, cube, vector):
      raise newException(ValueError, "Logcumsumexp gradient seeds must be disjoint from every input")
  if not allSeedsIndependent(boundActions):
    raise newException(ValueError, "Logcumsumexp gradient action seeds must own independent allocations")

  var report = newJObject()
  for index, name in actions:
    try:
      let action = boundActions[index]
      let output = execute(action)
      let repeated = execute(action)
      let observation = observe(
        action, output, repeated, base, rowsHead, rowsTail, cube, vector, boundActions)
      if preflightOnly:
        report[name] = observation
      else:
        for _ in 0 ..< warmupCount:
          consume(execute(action))
        var samples = newSeq[float64](repeatCount)
        for sample in 0 ..< repeatCount:
          let start = getMonoTime()
          for _ in 0 ..< callsPerSample:
            consume(execute(action))
          samples[sample] = float64((getMonoTime() - start).inNanoseconds) / 1e6 /
            float64(callsPerSample)
        let ordered = sorted(samples)
        let middle = samples.len div 2
        report[name] = %*{
          "observation": observation,
          "samples_ms": samples,
          "median_ms": (ordered[middle - 1] + ordered[middle]) / 2.0,
          "p95_ms": ordered[int(ceil(0.95 * float64(samples.len))) - 1],
        }
    except CatchableError as error:
      report[name] = %*{"error": error.msg}
  report

proc runForDtype(manifest: JsonNode; preflightOnly: bool): JsonNode =
  if manifest.kind != JObject or not manifest.hasKey("input_dtype") or
      manifest["input_dtype"].kind != JString:
    raise newException(ValueError, "Logcumsumexp gradient manifest must declare a string input dtype")
  case manifest["input_dtype"].getStr
  of "float32": run[float32](manifest, preflightOnly)
  of "float64": run[float64](manifest, preflightOnly)
  of "complex64": run[Complex[float32]](manifest, preflightOnly)
  of "complex128": run[Complex[float64]](manifest, preflightOnly)
  else: raise newException(ValueError, "Unsupported logcumsumexp gradient benchmark dtype")

when isMainModule:
  if paramCount() < 1 or paramCount() > 2:
    echo "Usage: logcumsumexp_gradient_runner INPUT.json [--preflight]"
    quit(if paramCount() == 0: 0 else: 2)
  let preflightOnly = paramCount() == 2 and paramStr(2) == "--preflight"
  if paramCount() == 2 and not preflightOnly:
    echo "Usage: logcumsumexp_gradient_runner INPUT.json [--preflight]"
    quit(2)
  let manifest = parseFile(paramStr(1))
  echo runForDtype(manifest, preflightOnly)
