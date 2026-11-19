# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures public allocating CPU min/max-with-indices calls for the pinned benchmark.
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
    "rows_tail_values", "actions", "warmup", "repeats", "iterations",
  ]

type
  Operation = enum
    opMin, opMax
  BoundAction[T] = object
    input: Tensor[T]
    operation: Operation
    axis: int
    keepdim: bool

proc dtypeName[T](): string =
  when T is int8: "int8"
  elif T is int16: "int16"
  elif T is int32: "int32"
  elif T is int64: "int64"
  elif T is uint8: "uint8"
  elif T is float32: "float32"
  elif T is float64: "float64"
  elif T is bool: "bool"
  else: {.error: "Unsupported paired-extrema benchmark dtype".}

proc lowFence[T](): T =
  when T is SomeFloat: T(-1024)
  else: low(T)

proc highFence[T](): T =
  when T is SomeFloat: T(1024)
  else: high(T)

proc expectedValue[T](index, profile: int): T =
  let column = index mod columns
  when T is bool:
    case profile
    of 1: column != columns - 1
    of 2: column == columns - 1
    else: index mod 3 != 0
  else:
    if profile == 1:
      if column == 0: return highFence[T]()
      if column == columns - 1: return lowFence[T]()
    elif profile == 2:
      if column == 0: return lowFence[T]()
      if column == columns - 1: return highFence[T]()
    T(index mod 63 + 1)

proc loadMatrix[T](values: JsonNode; label: string; profile: int): Tensor[T] =
  if values.kind != JArray or values.len != elementCount:
    raise newException(ValueError, label & " must contain exactly 16,384 values")
  result = newTensorUninit[T](rows, columns)
  let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
  for index in 0 ..< elementCount:
    var value: T
    when T is bool:
      if values[index].kind != JBool:
        raise newException(ValueError, label & " bool values must be exact JSON booleans")
      value = values[index].getBool
    elif T is SomeInteger:
      if values[index].kind != JInt:
        raise newException(ValueError, label & " integer values must be exact JSON integers")
      let exact = values[index].getInt
      if exact < int64(low(T)) or exact > int64(high(T)):
        raise newException(ValueError, label & " integer value is outside the input dtype")
      value = T(exact)
    else:
      if values[index].kind notin {JInt, JFloat}:
        raise newException(ValueError, label & " float values must be JSON numbers")
      let exact = values[index].getFloat
      if exact.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, label & " values must be finite")
      value = T(exact)
      if value.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, label & " value is not finite in the input dtype")
    if value != expectedValue[T](index, profile):
      raise newException(ValueError, label & " values do not match the deterministic profile")
    data[index] = value

proc bindAction[T](name: string; base, cube, rowsHead, rowsTail: Tensor[T]): BoundAction[T] =
  let pieces = name.split('_')
  if pieces.len < 3 or pieces[1] notin ["flat", "keepdim"]:
    raise newException(ValueError, "Malformed paired-extrema action: " & name)
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

proc execute[T](action: BoundAction[T]): tuple[values: Tensor[T], indices: Tensor[int64]] =
  case action.operation
  of opMin:
    action.input.minWithIndices(action.axis, keepdim = action.keepdim)
  of opMax:
    action.input.maxWithIndices(action.axis, keepdim = action.keepdim)

proc shapeJson[T](value: Tensor[T]): JsonNode =
  result = newJArray()
  for dimension in value.shape:
    result.add %dimension

proc observe[T](pair: tuple[values: Tensor[T], indices: Tensor[int64]];
                inputDtype: string): JsonNode =
  var values = newJArray()
  for item in pair.values:
    when T is bool:
      values.add %item
    elif T is SomeInteger:
      values.add %(int64(item))
    else:
      values.add %(float64(item))
  var indices = newJArray()
  for item in pair.indices:
    indices.add %item
  %*{
    "input_dtype": inputDtype,
    "values_dtype": inputDtype,
    "indices_dtype": "int64",
    "values_shape": shapeJson(pair.values),
    "indices_shape": shapeJson(pair.indices),
    "values": values,
    "indices": indices,
    "values_indices_distinct": cast[uint](pair.values.get_offset_ptr) !=
      cast[uint](pair.indices.get_offset_ptr),
  }

proc consume[T](pair: tuple[values: Tensor[T], indices: Tensor[int64]]) =
  ## Keep both independently allocated outputs observable across the timed call.
  let valuesAddress = cast[uint](pair.values.get_offset_ptr)
  let indicesAddress = cast[uint](pair.indices.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`valuesAddress`), "r"(`indicesAddress`)
    : "memory"
  """

proc run[T](manifest: JsonNode): JsonNode =
  if manifest.kind != JObject or manifest.len != manifestFields.len:
    raise newException(ValueError, "Paired-extrema manifest has missing or unknown fields")
  for field in manifestFields:
    if not manifest.hasKey(field):
      raise newException(ValueError, "Paired-extrema manifest has missing or unknown fields")
  if manifest["input_dtype"].kind != JString or manifest["input_dtype"].getStr != dtypeName[T]() or
      manifest["values_dtype"].kind != JString or manifest["values_dtype"].getStr != dtypeName[T]() or
      manifest["indices_dtype"].kind != JString or manifest["indices_dtype"].getStr != "int64":
    raise newException(ValueError, "Paired-extrema dtype fields do not match dispatched input and output types")
  if manifest["rows"].kind != JInt or manifest["columns"].kind != JInt or
      manifest["rows"].getInt != rows or manifest["columns"].getInt != columns or
      manifest["shape"] != %*[rows, columns] or manifest["cube_shape"] != %*[16, 32, 32]:
    raise newException(ValueError, "Paired-extrema input shape metadata mismatch")
  if manifest["cube_values"] != manifest["values"]:
    raise newException(ValueError, "Paired-extrema cube values must equal the base matrix values")
  for (field, expected) in [("warmup", warmupCount), ("repeats", repeatCount),
                            ("iterations", callsPerSample)]:
    if manifest[field].kind != JInt or manifest[field].getInt != expected:
      raise newException(ValueError, "Paired-extrema timing protocol mismatch")
  let supplied = manifest["actions"]
  if supplied.kind != JArray or supplied.len != actions.len:
    raise newException(ValueError, "Paired-extrema manifest must contain exactly 28 actions")
  for index, action in actions:
    if supplied[index].kind != JString or supplied[index].getStr != action:
      raise newException(ValueError, "Paired-extrema ordered action manifest mismatch")

  let base = loadMatrix[T](manifest["values"], "values", 0)
  let rowsHead = loadMatrix[T](manifest["rows_head_values"], "rows_head_values", 1)
  let rowsTail = loadMatrix[T](manifest["rows_tail_values"], "rows_tail_values", 2)
  let cube = base.reshape(cubeShape[0], cubeShape[1], cubeShape[2])
  var boundActions = newSeq[BoundAction[T]](actions.len)
  for index, name in actions:
    boundActions[index] = bindAction(name, base, cube, rowsHead, rowsTail)

  var report = newJObject()
  for index, name in actions:
    let action = boundActions[index]
    let observation = observe(execute(action), dtypeName[T]())
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
    let median = (ordered[middle - 1] + ordered[middle]) / 2.0
    report[name] = %*{
      "observation": observation,
      "samples_ms": samples,
      "median_ms": median,
      "p95_ms": ordered[int(ceil(0.95 * float64(samples.len))) - 1],
    }
  report

proc runForDtype(manifest: JsonNode): JsonNode =
  if manifest.kind != JObject or not manifest.hasKey("input_dtype") or
      manifest["input_dtype"].kind != JString:
    raise newException(ValueError, "Paired-extrema manifest must declare a string input dtype")
  case manifest["input_dtype"].getStr
  of "float32": run[float32](manifest)
  of "float64": run[float64](manifest)
  of "int8": run[int8](manifest)
  of "int16": run[int16](manifest)
  of "int32": run[int32](manifest)
  of "int64": run[int64](manifest)
  of "uint8": run[uint8](manifest)
  of "bool": run[bool](manifest)
  else: raise newException(ValueError, "Unsupported paired-extrema benchmark dtype")

when isMainModule:
  if paramCount() != 1:
    echo "Usage: paired_extrema_runner INPUT.json"
    quit(if paramCount() == 0: 0 else: 2)
  let manifest = parseFile(paramStr(1))
  echo runForDtype(manifest)
