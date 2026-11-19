# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating CPU paired extrema forward and explicitly seeded backward actions.
import std/[algorithm, json, math, monotimes, os, strutils, times]
import arraymancer

const
  rows = 128
  columns = 128
  elementCount = rows * columns
  cubeShape = [16, 32, 32]
  warmupCount = 50
  repeatCount = 100
  callsPerSample = 50
  actions = [
    "min_flat_rows", "min_flat_rows_head", "min_flat_rows_tail",
    "min_flat_columns", "min_flat_cube_axis0", "min_flat_cube_axis1",
    "min_flat_cube_axis2",
    "min_keepdim_rows", "min_keepdim_rows_head", "min_keepdim_rows_tail",
    "min_keepdim_columns", "min_keepdim_cube_axis0", "min_keepdim_cube_axis1",
    "min_keepdim_cube_axis2",
    "max_flat_rows", "max_flat_rows_head", "max_flat_rows_tail",
    "max_flat_columns", "max_flat_cube_axis0", "max_flat_cube_axis1",
    "max_flat_cube_axis2",
    "max_keepdim_rows", "max_keepdim_rows_head", "max_keepdim_rows_tail",
    "max_keepdim_columns", "max_keepdim_cube_axis0", "max_keepdim_cube_axis1",
    "max_keepdim_cube_axis2",
  ]
  manifestFields = [
    "input_dtype", "values_dtype", "indices_dtype", "rows", "columns",
    "shape", "values", "cube_shape", "cube_values", "rows_head_values",
    "rows_tail_values", "actions", "seeds", "warmup", "repeats", "iterations",
  ]

type
  Operation = enum
    opMin, opMax
  BoundAction[T] = object
    input: Tensor[T]
    seed: Tensor[T]
    operation: Operation
    axis: int
    keepdim: bool
  TrainingResult[T] = tuple[values: Tensor[T], indices: Tensor[int64], gradient: Tensor[T]]

proc dtypeName[T](): string =
  when T is float32: "float32"
  elif T is float64: "float64"
  else: {.error: "Unsupported paired-extrema gradient benchmark dtype".}

proc expectedValue[T: SomeFloat](index, profile: int): T =
  let column = index mod columns
  if profile == 1:
    if column == 0: return T(1024)
    if column == columns - 1: return T(-1024)
  elif profile == 2:
    if column == 0: return T(-1024)
    if column == columns - 1: return T(1024)
  T(index mod 63 + 1)

proc loadMatrix[T: SomeFloat](values: JsonNode; label: string; profile: int): Tensor[T] =
  if values.kind != JArray or values.len != elementCount:
    raise newException(ValueError, label & " must contain exactly 16,384 values")
  result = newTensorUninit[T](rows, columns)
  let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
  for index in 0 ..< elementCount:
    if values[index].kind != JFloat:
      raise newException(ValueError, label & " values must be exact JSON floats")
    let supplied = values[index].getFloat
    if supplied.classify in {fcNan, fcInf, fcNegInf}:
      raise newException(ValueError, label & " values must be finite")
    let value = T(supplied)
    if value.classify in {fcNan, fcInf, fcNegInf} or value != expectedValue[T](index, profile):
      raise newException(ValueError, label & " values do not match the deterministic profile")
    data[index] = value

proc expectedShape(inputShape: Metadata; axis: int; keepdim: bool): Metadata =
  result.len = inputShape.len
  var outputIndex = 0
  for index in 0 ..< inputShape.len:
    if index == axis:
      if keepdim:
        result[outputIndex] = 1
        inc outputIndex
    else:
      result[outputIndex] = inputShape[index]
      inc outputIndex
  result.len = outputIndex

proc loadSeed[T: SomeFloat](seedNode: JsonNode; expected: Metadata; dtype: string;
                            name: string): Tensor[T] =
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
    if values[index].kind != JFloat:
      raise newException(ValueError, name & " seed values must be exact JSON floats")
    let supplied = values[index].getFloat
    if supplied.classify in {fcNan, fcInf, fcNegInf}:
      raise newException(ValueError, name & " seed values must be finite")
    let value = T(supplied)
    let expectedValue = T(index mod 7 - 3) + T(0.25)
    if value.classify in {fcNan, fcInf, fcNegInf} or value == T(0) or value != expectedValue:
      raise newException(ValueError, name & " seed must match the finite nonzero deterministic sequence")
    data[index] = value

proc bindAction[T: SomeFloat](name, dtype: string; base, cube, rowsHead, rowsTail: Tensor[T];
                              seedNode: JsonNode): BoundAction[T] =
  let pieces = name.split('_')
  if pieces.len < 3 or pieces[1] notin ["flat", "keepdim"]:
    raise newException(ValueError, "Malformed paired-extrema gradient action: " & name)
  case pieces[0]
  of "min": result.operation = opMin
  of "max": result.operation = opMax
  else: raise newException(ValueError, "Unknown paired-extrema operation: " & pieces[0])
  result.keepdim = pieces[1] == "keepdim"
  let profile = pieces[2 .. ^1].join("_")
  case profile
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
  else: raise newException(ValueError, "Unknown paired-extrema profile: " & profile)
  result.seed = loadSeed[T](seedNode, expectedShape(result.input.shape, result.axis, result.keepdim),
                            dtype, name)

proc execute[T: SomeFloat](action: BoundAction[T]): TrainingResult[T] =
  ## Each call owns a fresh context and leaf; immutable inputs/seeds are shared.
  let context = newContext(Tensor[T])
  let variable = context.variable(action.input, requires_grad = true)
  let pair =
    case action.operation
    of opMin: variable.minWithIndices(action.axis, keepdim = action.keepdim)
    of opMax: variable.maxWithIndices(action.axis, keepdim = action.keepdim)
  pair.values.backprop(action.seed)
  result = (pair.values.value, pair.indices, variable.grad)

proc shapeJson[T](value: Tensor[T]): JsonNode =
  result = newJArray()
  for dimension in value.shape:
    result.add %dimension

proc observe[T: SomeFloat](output: TrainingResult[T]; input: Tensor[T]; dtype: string): JsonNode =
  var values = newJArray()
  for item in output.values:
    values.add %(float64(item))
  var indices = newJArray()
  for item in output.indices:
    indices.add %item
  var gradient = newJArray()
  for item in output.gradient:
    gradient.add %(float64(item))
  let valuesAddress = cast[uint](output.values.get_offset_ptr)
  let indicesAddress = cast[uint](output.indices.get_offset_ptr)
  let gradientAddress = cast[uint](output.gradient.get_offset_ptr)
  let inputAddress = cast[uint](input.get_offset_ptr)
  %*{
    "input_dtype": dtype,
    "values_dtype": dtype,
    "indices_dtype": "int64",
    "gradient_dtype": dtype,
    "values_shape": shapeJson(output.values),
    "indices_shape": shapeJson(output.indices),
    "gradient_shape": shapeJson(output.gradient),
    "values": values,
    "indices": indices,
    "gradient": gradient,
    "values_indices_distinct": valuesAddress != indicesAddress,
    "values_gradient_distinct": valuesAddress != gradientAddress,
    "indices_gradient_distinct": indicesAddress != gradientAddress,
    "values_input_distinct": valuesAddress != inputAddress,
    "indices_input_distinct": indicesAddress != inputAddress,
    "gradient_input_distinct": gradientAddress != inputAddress,
  }

proc consume[T](output: TrainingResult[T]) =
  ## Keep every independently allocated output observable through the timed call.
  let valuesAddress = cast[uint](output.values.get_offset_ptr)
  let indicesAddress = cast[uint](output.indices.get_offset_ptr)
  let gradientAddress = cast[uint](output.gradient.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`valuesAddress`), "r"(`indicesAddress`), "r"(`gradientAddress`)
    : "memory"
  """

proc run[T: SomeFloat](manifest: JsonNode; dtype: string): JsonNode =
  if manifest.kind != JObject or manifest.len != manifestFields.len:
    raise newException(ValueError, "Paired-extrema gradient manifest has missing or unknown fields")
  for field in manifestFields:
    if not manifest.hasKey(field):
      raise newException(ValueError, "Paired-extrema gradient manifest has missing or unknown fields")
  if manifest["input_dtype"].kind != JString or manifest["input_dtype"].getStr != dtype or
      manifest["values_dtype"].kind != JString or manifest["values_dtype"].getStr != dtype or
      manifest["indices_dtype"].kind != JString or manifest["indices_dtype"].getStr != "int64":
    raise newException(ValueError, "Paired-extrema gradient dtype metadata mismatch")
  if manifest["rows"].kind != JInt or manifest["columns"].kind != JInt or
      manifest["rows"].getInt != rows or manifest["columns"].getInt != columns or
      manifest["shape"] != %*[rows, columns] or manifest["cube_shape"] != %*[16, 32, 32]:
    raise newException(ValueError, "Paired-extrema gradient input shape metadata mismatch")
  if manifest["cube_values"] != manifest["values"]:
    raise newException(ValueError, "Paired-extrema gradient cube must share the base values")
  for (field, expected) in [("warmup", warmupCount), ("repeats", repeatCount),
                            ("iterations", callsPerSample)]:
    if manifest[field].kind != JInt or manifest[field].getInt != expected:
      raise newException(ValueError, "Paired-extrema gradient timing protocol mismatch")
  let supplied = manifest["actions"]
  if supplied.kind != JArray or supplied.len != actions.len:
    raise newException(ValueError, "Paired-extrema gradient manifest must contain exactly 28 actions")
  for index, action in actions:
    if supplied[index].kind != JString or supplied[index].getStr != action:
      raise newException(ValueError, "Paired-extrema gradient ordered action manifest mismatch")

  let base = loadMatrix[T](manifest["values"], "values", 0)
  let rowsHead = loadMatrix[T](manifest["rows_head_values"], "rows_head_values", 1)
  let rowsTail = loadMatrix[T](manifest["rows_tail_values"], "rows_tail_values", 2)
  let cube = base.reshape(cubeShape[0], cubeShape[1], cubeShape[2])
  let seeds = manifest["seeds"]
  if seeds.kind != JObject or seeds.len != actions.len:
    raise newException(ValueError, "Paired-extrema gradient seed manifest must contain exactly 28 seeds")
  var boundActions = newSeq[BoundAction[T]](actions.len)
  for index, name in actions:
    if not seeds.hasKey(name):
      raise newException(ValueError, "Missing seed for " & name)
    boundActions[index] = bindAction(name, dtype, base, cube, rowsHead, rowsTail, seeds[name])

  var report = newJObject()
  for index, name in actions:
    let action = boundActions[index]
    let output = execute(action)
    let observation = observe(output, action.input, dtype)
    consume(output)
    for _ in 0 ..< warmupCount:
      consume(execute(action))
    var samples = newSeq[float64](repeatCount)
    for sample in 0 ..< repeatCount:
      let start = getMonoTime()
      for _ in 0 ..< callsPerSample:
        consume(execute(action))
      samples[sample] = float64((getMonoTime() - start).inNanoseconds) / 1e6 /
        float64(callsPerSample)
    var ordered = samples
    ordered.sort()
    let middle = samples.len div 2
    report[name] = %*{
      "observation": observation,
      "samples_ms": samples,
      "median_ms": (ordered[middle - 1] + ordered[middle]) / 2.0,
      "p95_ms": ordered[int(ceil(0.95 * float64(samples.len))) - 1],
    }
  report

when isMainModule:
  if paramCount() != 1:
    echo "Usage: paired_extrema_gradient_runner INPUT.json"
    quit(if paramCount() == 0: 0 else: 2)
  let manifest = parseFile(paramStr(1))
  if manifest.kind != JObject or not manifest.hasKey("input_dtype") or
      manifest["input_dtype"].kind != JString:
    raise newException(ValueError, "Paired-extrema gradient manifest must declare a string input dtype")
  case manifest["input_dtype"].getStr
  of "float32": echo run[float32](manifest, "float32")
  of "float64": echo run[float64](manifest, "float64")
  else: raise newException(ValueError, "Unsupported paired-extrema gradient input dtype")
