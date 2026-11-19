# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures public allocating CPU joint aminmax calls for the pinned benchmark.
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
    "input_dtype", "min_dtype", "max_dtype", "rows", "columns",
    "shape", "values", "cube_shape", "cube_values", "rows_head_values",
    "rows_tail_values", "profiles", "actions", "warmup", "repeats", "iterations",
  ]

type
  BoundAction[T] = object
    input: Tensor[T]
    axis: int
    whole: bool
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
  else: {.error: "Unsupported aminmax benchmark dtype".}

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

proc addressRange[T](value: Tensor[T]): tuple[first, last: uint] =
  result.first = cast[uint](value.get_offset_ptr)
  result.last = result.first + uint(value.size) * uint(sizeof(T))

proc rangesDisjoint[T](left, right: Tensor[T]): bool =
  let leftRange = addressRange(left)
  let rightRange = addressRange(right)
  leftRange.last <= rightRange.first or rightRange.last <= leftRange.first

proc disjointFromInputs[T](value: Tensor[T]; base, rowsHead, rowsTail, cube: Tensor[T]): bool =
  rangesDisjoint(value, base) and rangesDisjoint(value, rowsHead) and
    rangesDisjoint(value, rowsTail) and rangesDisjoint(value, cube)

proc bindAction[T](name: string; base, cube, rowsHead, rowsTail: Tensor[T]): BoundAction[T] =
  let pieces = name.split('_')
  if pieces.len < 2 or pieces[0] notin ["flat", "keepdim"]:
    raise newException(ValueError, "Malformed aminmax action: " & name)
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
    raise newException(ValueError, "Unknown aminmax profile: " & profile)

proc execute[T](action: BoundAction[T]): tuple[min, max: Tensor[T]] =
  if action.whole:
    result = action.input.aminmax(keepdim = action.keepdim)
  else:
    result = action.input.aminmax(action.axis, keepdim = action.keepdim)

proc expectedShape[T](action: BoundAction[T]): seq[int] =
  let inputShape = action.input.shape
  if action.whole:
    if action.keepdim:
      result = newSeq[int](inputShape.len)
      for index in 0 ..< result.len:
        result[index] = 1
  else:
    for index, dimension in inputShape:
      if index == action.axis:
        if action.keepdim:
          result.add 1
      else:
        result.add dimension

proc shapeJson[T](value: Tensor[T]): JsonNode =
  result = newJArray()
  for dimension in value.shape:
    result.add %dimension

proc valuesJson[T](value: Tensor[T]): JsonNode =
  result = newJArray()
  for item in value:
    when T is bool:
      result.add %item
    elif T is SomeInteger:
      result.add %(int64(item))
    else:
      result.add %(float64(item))

proc observe[T](action: BoundAction[T]; pair: tuple[min, max: Tensor[T]];
                inputDtype: string; base, rowsHead, rowsTail, cube: Tensor[T]): JsonNode =
  let expected = expectedShape(action)
  if pair.min.shape != expected or pair.max.shape != expected or
      pair.min.size != pair.max.size:
    raise newException(ValueError, "aminmax output shape does not match action profile")
  let outputsDistinct = rangesDisjoint(pair.min, pair.max)
  let minDisjoint = disjointFromInputs(pair.min, base, rowsHead, rowsTail, cube)
  let maxDisjoint = disjointFromInputs(pair.max, base, rowsHead, rowsTail, cube)
  if not outputsDistinct or not minDisjoint or not maxDisjoint:
    raise newException(ValueError, "aminmax outputs must be independent allocations disjoint from every input")
  %*{
    "input_dtype": inputDtype,
    "min_dtype": inputDtype,
    "max_dtype": inputDtype,
    "min_shape": shapeJson(pair.min),
    "max_shape": shapeJson(pair.max),
    "min_values": valuesJson(pair.min),
    "max_values": valuesJson(pair.max),
    "min_max_distinct": outputsDistinct,
    "min_disjoint_from_inputs": minDisjoint,
    "max_disjoint_from_inputs": maxDisjoint,
  }

proc consume[T](pair: tuple[min, max: Tensor[T]]) =
  ## Keep both independently allocated outputs observable across the timed call.
  let minAddress = cast[uint](pair.min.get_offset_ptr)
  let maxAddress = cast[uint](pair.max.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`minAddress`), "r"(`maxAddress`)
    : "memory"
  """

proc run[T](manifest: JsonNode; preflightOnly: bool): JsonNode =
  if manifest.kind != JObject or manifest.len != manifestFields.len:
    raise newException(ValueError, "Aminmax manifest has missing or unknown fields")
  for field in manifestFields:
    if not manifest.hasKey(field):
      raise newException(ValueError, "Aminmax manifest has missing or unknown fields")
  if manifest["input_dtype"].kind != JString or manifest["input_dtype"].getStr != dtypeName[T]() or
      manifest["min_dtype"].kind != JString or manifest["min_dtype"].getStr != dtypeName[T]() or
      manifest["max_dtype"].kind != JString or manifest["max_dtype"].getStr != dtypeName[T]():
    raise newException(ValueError, "Aminmax dtype fields do not match dispatched input and output types")
  if manifest["rows"].kind != JInt or manifest["columns"].kind != JInt or
      manifest["rows"].getInt != rows or manifest["columns"].getInt != columns or
      manifest["shape"] != %*[rows, columns] or manifest["cube_shape"] != %*[16, 32, 32]:
    raise newException(ValueError, "Aminmax input shape metadata mismatch")
  if manifest["cube_values"] != manifest["values"]:
    raise newException(ValueError, "Aminmax cube values must equal base matrix values")
  for (field, expected) in [("warmup", warmupCount), ("repeats", repeatCount),
                            ("iterations", callsPerSample)]:
    if manifest[field].kind != JInt or manifest[field].getInt != expected:
      raise newException(ValueError, "Aminmax timing protocol mismatch")
  let suppliedProfiles = manifest["profiles"]
  if suppliedProfiles.kind != JArray or suppliedProfiles.len != profiles.len:
    raise newException(ValueError, "Aminmax profile manifest mismatch")
  for index, profile in profiles:
    if suppliedProfiles[index].kind != JString or suppliedProfiles[index].getStr != profile:
      raise newException(ValueError, "Aminmax ordered profile manifest mismatch")
  let suppliedActions = manifest["actions"]
  if suppliedActions.kind != JArray or suppliedActions.len != actions.len:
    raise newException(ValueError, "Aminmax manifest must contain exactly 18 actions")
  for index, action in actions:
    if suppliedActions[index].kind != JString or suppliedActions[index].getStr != action:
      raise newException(ValueError, "Aminmax ordered action manifest mismatch")

  let base = loadMatrix[T](manifest["values"], "values", 0)
  let rowsHead = loadMatrix[T](manifest["rows_head_values"], "rows_head_values", 1)
  let rowsTail = loadMatrix[T](manifest["rows_tail_values"], "rows_tail_values", 2)
  let cube = base.reshape(cubeShape[0], cubeShape[1], cubeShape[2])
  if not rangesDisjoint(base, rowsHead) or not rangesDisjoint(base, rowsTail) or
      not rangesDisjoint(rowsHead, rowsTail) or
      cast[uint](cube.get_offset_ptr) != cast[uint](base.get_offset_ptr) or
      cube.size != base.size:
    raise newException(ValueError, "Aminmax inputs must be independent resident matrices with a base-sharing cube view")
  var boundActions = newSeq[BoundAction[T]](actions.len)
  for index, name in actions:
    boundActions[index] = bindAction(name, base, cube, rowsHead, rowsTail)

  var report = newJObject()
  for index, name in actions:
    let action = boundActions[index]
    let observation = observe(action, execute(action), dtypeName[T](),
                              base, rowsHead, rowsTail, cube)
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

proc runForDtype(manifest: JsonNode; preflightOnly: bool): JsonNode =
  if manifest.kind != JObject or not manifest.hasKey("input_dtype") or
      manifest["input_dtype"].kind != JString:
    raise newException(ValueError, "Aminmax manifest must declare a string input dtype")
  case manifest["input_dtype"].getStr
  of "float32": run[float32](manifest, preflightOnly)
  of "float64": run[float64](manifest, preflightOnly)
  of "int8": run[int8](manifest, preflightOnly)
  of "int16": run[int16](manifest, preflightOnly)
  of "int32": run[int32](manifest, preflightOnly)
  of "int64": run[int64](manifest, preflightOnly)
  of "uint8": run[uint8](manifest, preflightOnly)
  of "bool": run[bool](manifest, preflightOnly)
  else: raise newException(ValueError, "Unsupported aminmax benchmark dtype")

when isMainModule:
  if paramCount() < 1 or paramCount() > 2:
    echo "Usage: aminmax_runner INPUT.json [--preflight]"
    quit(if paramCount() == 0: 0 else: 2)
  let preflightOnly = paramCount() == 2 and paramStr(2) == "--preflight"
  if paramCount() == 2 and not preflightOnly:
    echo "Usage: aminmax_runner INPUT.json [--preflight]"
    quit(2)
  let manifest = parseFile(paramStr(1))
  echo runForDtype(manifest, preflightOnly)
