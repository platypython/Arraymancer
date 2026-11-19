# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating CPU logsumexp forward and explicitly seeded backward actions.
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
    "whole_matrix", "whole_cube",
    "rows", "rows_head", "rows_tail", "columns",
    "cube_axis0", "cube_axis1", "cube_axis2",
  ]
  actions = [
    "flat_whole_matrix", "flat_whole_cube",
    "flat_rows", "flat_rows_head", "flat_rows_tail", "flat_columns",
    "flat_cube_axis0", "flat_cube_axis1", "flat_cube_axis2",
    "keepdim_whole_matrix", "keepdim_whole_cube",
    "keepdim_rows", "keepdim_rows_head", "keepdim_rows_tail", "keepdim_columns",
    "keepdim_cube_axis0", "keepdim_cube_axis1", "keepdim_cube_axis2",
  ]
  manifestFields = [
    "input_dtype", "output_dtype", "rows", "columns", "shape", "values",
    "cube_shape", "cube_values", "rows_head_values", "rows_tail_values",
    "profiles", "actions", "seeds", "warmup", "repeats", "iterations",
  ]

type
  BoundAction[T] = object
    input: Tensor[T]
    seed: Tensor[T]
    axis: int
    whole: bool
    keepdim: bool
  TrainingResult[T] = tuple[output: Tensor[T], gradient: Tensor[T]]

proc dtypeName[T](): string =
  when T is float32: "float32"
  elif T is float64: "float64"
  elif T is Complex[float32]: "complex64"
  elif T is Complex[float64]: "complex128"
  else: {.error: "Unsupported logsumexp gradient benchmark dtype".}

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
    T(re: Component(index mod 7 - 3) + Component(0.25), im: Component(-0.5))
  else:
    T(index mod 7 - 3) + T(0.25)

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
    raise newException(ValueError, name & " seed shape does not match output shape")
  for index in 0 ..< expected.len:
    if shapeNode[index].kind != JInt or shapeNode[index].getInt != expected[index]:
      raise newException(ValueError, name & " seed shape does not match output shape")
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
      raise newException(ValueError, name & " seed must match the finite nonzero deterministic sequence")
    when T is Complex:
      if value.re.classify in {fcNan, fcInf, fcNegInf} or
          value.im.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, name & " seed values must be finite")
    else:
      if value.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, name & " seed values must be finite")
    data[index] = value

proc expectedShape[T](action: BoundAction[T]): Metadata =
  let inputShape = action.input.shape
  if action.whole:
    if action.keepdim:
      result.len = inputShape.len
      for index in 0 ..< result.len:
        result[index] = 1
  else:
    for index, dimension in inputShape:
      if index == action.axis:
        if action.keepdim:
          result.add 1
      else:
        result.add dimension

proc bindAction[T](name, dtype: string; base, cube, rowsHead, rowsTail: Tensor[T];
                   seedNode: JsonNode): BoundAction[T] =
  let pieces = name.split('_')
  if pieces.len < 2 or pieces[0] notin ["flat", "keepdim"]:
    raise newException(ValueError, "Malformed logsumexp gradient action: " & name)
  result.keepdim = pieces[0] == "keepdim"
  let profile = pieces[1 .. ^1].join("_")
  case profile
  of "whole_matrix":
    result.input = base
    result.whole = true
  of "whole_cube":
    result.input = cube
    result.whole = true
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
  else:
    raise newException(ValueError, "Unknown logsumexp gradient profile: " & profile)
  result.seed = loadSeed[T](seedNode, expectedShape(result), dtype, name)

proc execute[T](action: BoundAction[T]): TrainingResult[T] =
  ## Every training call owns a fresh context and leaf; inputs and seeds are resident.
  let context = newContext(Tensor[T])
  let variable = context.variable(action.input, requires_grad = true)
  let output =
    if action.whole:
      variable.logsumexp(keepdim = action.keepdim)
    else:
      variable.logsumexp(action.axis, keepdim = action.keepdim)
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

proc addressRange[T](value: Tensor[T]): tuple[first, last: uint] =
  result.first = cast[uint](value.get_offset_ptr)
  result.last = result.first + uint(value.size) * uint(sizeof(T))

proc rangesDisjoint[LeftT, RightT](left: Tensor[LeftT]; right: Tensor[RightT]): bool =
  let leftRange = addressRange(left)
  let rightRange = addressRange(right)
  leftRange.last <= rightRange.first or rightRange.last <= leftRange.first

proc disjointFromInputs[OutputT, InputT](value: Tensor[OutputT];
    base, rowsHead, rowsTail, cube: Tensor[InputT]): bool =
  rangesDisjoint(value, base) and rangesDisjoint(value, rowsHead) and
    rangesDisjoint(value, rowsTail) and rangesDisjoint(value, cube)

proc observe[T](action: BoundAction[T]; output, repeated: TrainingResult[T];
    base, rowsHead, rowsTail, cube: Tensor[T]): JsonNode =
  let expected = expectedShape(action)
  if output.output.shape != expected or repeated.output.shape != expected or
      output.gradient.shape != action.input.shape or repeated.gradient.shape != action.input.shape or
      not output.output.is_C_contiguous or not repeated.output.is_C_contiguous or
      not output.gradient.is_C_contiguous or not repeated.gradient.is_C_contiguous:
    raise newException(ValueError, "logsumexp gradient output dtype, shape, or contiguous layout mismatch")
  let outputDisjoint = disjointFromInputs(output.output, base, rowsHead, rowsTail, cube)
  let gradientDisjoint = disjointFromInputs(output.gradient, base, rowsHead, rowsTail, cube)
  let outputSeedDisjoint = rangesDisjoint(output.output, action.seed)
  let gradientSeedDisjoint = rangesDisjoint(output.gradient, action.seed)
  let outputGradientDisjoint = rangesDisjoint(output.output, output.gradient)
  let outputRepeatDisjoint = rangesDisjoint(output.output, repeated.output)
  let gradientRepeatDisjoint = rangesDisjoint(output.gradient, repeated.gradient)
  let repeatCrossDisjoint = rangesDisjoint(output.output, repeated.gradient) and
    rangesDisjoint(output.gradient, repeated.output)
  let outputRepeatedGradientDisjoint = rangesDisjoint(repeated.output, repeated.gradient)
  let repeatedOutputDisjoint = disjointFromInputs(repeated.output, base, rowsHead, rowsTail, cube)
  let repeatedGradientDisjoint = disjointFromInputs(repeated.gradient, base, rowsHead, rowsTail, cube)
  let repeatedOutputSeedDisjoint = rangesDisjoint(repeated.output, action.seed)
  let repeatedGradientSeedDisjoint = rangesDisjoint(repeated.gradient, action.seed)
  if not outputDisjoint or not gradientDisjoint or not outputSeedDisjoint or
      not gradientSeedDisjoint or not outputGradientDisjoint or not outputRepeatDisjoint or
      not gradientRepeatDisjoint or not repeatCrossDisjoint or not outputRepeatedGradientDisjoint or
      not repeatedOutputDisjoint or not repeatedGradientDisjoint or not repeatedOutputSeedDisjoint or
      not repeatedGradientSeedDisjoint:
    raise newException(ValueError, "logsumexp gradient outputs must be independent allocations disjoint from inputs and seed")
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
    "output_disjoint_from_seed": outputSeedDisjoint,
    "gradient_disjoint_from_seed": gradientSeedDisjoint,
    "output_gradient_disjoint": outputGradientDisjoint,
    "output_distinct_from_repeat": outputRepeatDisjoint,
    "gradient_distinct_from_repeat": gradientRepeatDisjoint,
  }

proc consume[T](output: TrainingResult[T]) =
  ## Keep both forward and full input-gradient allocations observable, with no extra reduction.
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
    raise newException(ValueError, "Logsumexp gradient manifest has missing or unknown fields")
  for field in manifestFields:
    if not manifest.hasKey(field):
      raise newException(ValueError, "Logsumexp gradient manifest has missing or unknown fields")
  if manifest["input_dtype"].kind != JString or manifest["input_dtype"].getStr != dtypeName[T]() or
      manifest["output_dtype"].kind != JString or manifest["output_dtype"].getStr != dtypeName[T]():
    raise newException(ValueError, "Logsumexp gradient dtype fields do not match dispatched types")
  if manifest["rows"].kind != JInt or manifest["columns"].kind != JInt or
      manifest["rows"].getInt != rows or manifest["columns"].getInt != columns or
      manifest["shape"] != %*[rows, columns] or manifest["cube_shape"] != %*[16, 32, 32]:
    raise newException(ValueError, "Logsumexp gradient input shape metadata mismatch")
  if manifest["cube_values"] != manifest["values"]:
    raise newException(ValueError, "Logsumexp gradient cube values must equal base matrix values")
  for (field, expected) in [("warmup", warmupCount), ("repeats", repeatCount),
                            ("iterations", callsPerSample)]:
    if manifest[field].kind != JInt or manifest[field].getInt != expected:
      raise newException(ValueError, "Logsumexp gradient timing protocol mismatch")
  let suppliedProfiles = manifest["profiles"]
  if suppliedProfiles.kind != JArray or suppliedProfiles.len != profiles.len:
    raise newException(ValueError, "Logsumexp gradient profile manifest mismatch")
  for index, profile in profiles:
    if suppliedProfiles[index].kind != JString or suppliedProfiles[index].getStr != profile:
      raise newException(ValueError, "Logsumexp gradient ordered profile manifest mismatch")
  let suppliedActions = manifest["actions"]
  if suppliedActions.kind != JArray or suppliedActions.len != actions.len:
    raise newException(ValueError, "Logsumexp gradient manifest must contain exactly 18 actions")
  for index, action in actions:
    if suppliedActions[index].kind != JString or suppliedActions[index].getStr != action:
      raise newException(ValueError, "Logsumexp gradient ordered action manifest mismatch")

  let base = loadMatrix[T](manifest["values"], "values", 0)
  let rowsHead = loadMatrix[T](manifest["rows_head_values"], "rows_head_values", 1)
  let rowsTail = loadMatrix[T](manifest["rows_tail_values"], "rows_tail_values", 2)
  let cube = base.reshape(cubeShape[0], cubeShape[1], cubeShape[2])
  if not base.is_C_contiguous or not rowsHead.is_C_contiguous or not rowsTail.is_C_contiguous or
      not cube.is_C_contiguous or not rangesDisjoint(base, rowsHead) or
      not rangesDisjoint(base, rowsTail) or not rangesDisjoint(rowsHead, rowsTail) or
      cast[uint](cube.get_offset_ptr) != cast[uint](base.get_offset_ptr) or cube.size != base.size:
    raise newException(ValueError, "Logsumexp gradient inputs must be independent resident matrices with a base-sharing cube view")
  let seeds = manifest["seeds"]
  if seeds.kind != JObject or seeds.len != actions.len:
    raise newException(ValueError, "Logsumexp gradient seed manifest must contain exactly 18 seeds")
  var boundActions = newSeq[BoundAction[T]](actions.len)
  for index, name in actions:
    if not seeds.hasKey(name):
      raise newException(ValueError, "Missing seed for " & name)
    boundActions[index] = bindAction(name, dtypeName[T](), base, cube, rowsHead, rowsTail, seeds[name])

  var report = newJObject()
  for index, name in actions:
    let action = boundActions[index]
    let output = execute(action)
    let repeated = execute(action)
    let observation = observe(action, output, repeated, base, rowsHead, rowsTail, cube)
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
  report

proc runForDtype(manifest: JsonNode; preflightOnly: bool): JsonNode =
  if manifest.kind != JObject or not manifest.hasKey("input_dtype") or
      manifest["input_dtype"].kind != JString:
    raise newException(ValueError, "Logsumexp gradient manifest must declare a string input dtype")
  case manifest["input_dtype"].getStr
  of "float32": run[float32](manifest, preflightOnly)
  of "float64": run[float64](manifest, preflightOnly)
  of "complex64": run[Complex[float32]](manifest, preflightOnly)
  of "complex128": run[Complex[float64]](manifest, preflightOnly)
  else: raise newException(ValueError, "Unsupported logsumexp gradient benchmark dtype")

when isMainModule:
  if paramCount() < 1 or paramCount() > 2:
    echo "Usage: logsumexp_gradient_runner INPUT.json [--preflight]"
    quit(if paramCount() == 0: 0 else: 2)
  let preflightOnly = paramCount() == 2 and paramStr(2) == "--preflight"
  if paramCount() == 2 and not preflightOnly:
    echo "Usage: logsumexp_gradient_runner INPUT.json [--preflight]"
    quit(2)
  let manifest = parseFile(paramStr(1))
  echo runForDtype(manifest, preflightOnly)
