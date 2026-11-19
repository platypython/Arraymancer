# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating CPU public logsumexp calls for the pinned benchmark.
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
    "profiles", "actions", "warmup", "repeats", "iterations",
  ]

type
  BoundAction[T] = object
    input: Tensor[T]
    axis: int
    whole: bool
    keepdim: bool


template logsumexpOutput(T: typedesc): typedesc =
  when T is SomeInteger or T is bool: float32
  else: T

proc dtypeName[T](): string =
  when T is bool: "bool"
  elif T is int8: "int8"
  elif T is int16: "int16"
  elif T is int32: "int32"
  elif T is int64: "int64"
  elif T is uint8: "uint8"
  elif T is uint16: "uint16"
  elif T is uint32: "uint32"
  elif T is uint64: "uint64"
  elif T is float32: "float32"
  elif T is float64: "float64"
  elif T is Complex[float32]: "complex64"
  elif T is Complex[float64]: "complex128"
  else: {.error: "Unsupported logsumexp benchmark dtype".}

proc expectedValue[T](index, profile: int): T =
  let column = index mod columns
  when T is bool:
    case profile
    of 1: column != columns - 1
    of 2: column == columns - 1
    else: index mod 3 != 0
  elif T is Complex:
    type Component = typeof(default(T).re)
    if profile == 1:
      if column == 0: return T(re: Component(2.0), im: Component(0.125))
      if column == columns - 1: return T(re: Component(-2.0), im: Component(-0.125))
    elif profile == 2:
      if column == 0: return T(re: Component(-2.0), im: Component(-0.125))
      if column == columns - 1: return T(re: Component(2.0), im: Component(0.125))
    T(re: Component(float64(index mod 31 - 15) / 16.0),
      im: Component(float64(index mod 9 - 4) / 64.0))
  elif T is SomeFloat:
    if profile == 1:
      if column == 0: return T(4.0)
      if column == columns - 1: return T(-4.0)
    elif profile == 2:
      if column == 0: return T(-4.0)
      if column == columns - 1: return T(4.0)
    T(float64(index mod 63 - 31) / 8.0)
  elif T is SomeUnsignedInt:
    if profile == 1:
      if column == 0: return T(64)
      if column == columns - 1: return T(0)
    elif profile == 2:
      if column == 0: return T(0)
      if column == columns - 1: return T(64)
    T(index mod 63 + 1)
  else:
    if profile == 1:
      if column == 0: return T(64)
      if column == columns - 1: return T(-64)
    elif profile == 2:
      if column == 0: return T(-64)
      if column == columns - 1: return T(64)
    T(index mod 63 - 31)

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
      if value != expectedValue[T](index, profile):
        raise newException(ValueError, label & " values do not match the deterministic bool profile")
    elif T is SomeInteger:
      if values[index].kind != JInt:
        raise newException(ValueError, label & " integer values must be exact JSON integers")
      let exact = values[index].getInt
      if exact != int64(expectedValue[T](index, profile)):
        raise newException(ValueError, label & " values do not match the deterministic typed profile")
      value = T(exact)
    elif T is Complex:
      let pair = values[index]
      if pair.kind != JArray or pair.len != 2 or
          pair[0].kind notin {JInt, JFloat} or pair[1].kind notin {JInt, JFloat}:
        raise newException(ValueError, label & " complex values must be [real, imag] JSON number pairs")
      type Component = typeof(default(T).re)
      value = T(re: Component(pair[0].getFloat), im: Component(pair[1].getFloat))
      if value != expectedValue[T](index, profile):
        raise newException(ValueError, label & " values do not match the deterministic complex profile")
    else:
      if values[index].kind notin {JInt, JFloat}:
        raise newException(ValueError, label & " floating values must be JSON numbers")
      value = T(values[index].getFloat)
      if value != expectedValue[T](index, profile):
        raise newException(ValueError, label & " values do not match the deterministic typed profile")
    data[index] = value

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

proc bindAction[T](name: string; base, cube, rowsHead, rowsTail: Tensor[T]): BoundAction[T] =
  let pieces = name.split('_')
  if pieces.len < 2 or pieces[0] notin ["flat", "keepdim"]:
    raise newException(ValueError, "Malformed logsumexp action: " & name)
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
    raise newException(ValueError, "Unknown logsumexp profile: " & profile)

proc execute[T](action: BoundAction[T]): Tensor[logsumexpOutput(T)] =
  if action.whole:
    result = action.input.logsumexp(keepdim = action.keepdim)
  else:
    result = action.input.logsumexp(action.axis, keepdim = action.keepdim)

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
    elif T is SomeInteger:
      result.add jsonFloat(float64(item))
    elif T is bool:
      result.add jsonFloat(float64(ord(item)))
    else:
      result.add jsonFloat(float64(item))

proc observe[InputT, OutputT](action: BoundAction[InputT]; output, repeated: Tensor[OutputT];
    base, rowsHead, rowsTail, cube: Tensor[InputT]): JsonNode =
  let expected = expectedShape(action)
  if output.shape != expected or repeated.shape != expected or
      not output.is_C_contiguous or not repeated.is_C_contiguous:
    raise newException(ValueError, "logsumexp output dtype, shape, or contiguous layout mismatch")
  let outputDistinct = rangesDisjoint(output, repeated)
  let outputDisjoint = disjointFromInputs(output, base, rowsHead, rowsTail, cube)
  let repeatedDisjoint = disjointFromInputs(repeated, base, rowsHead, rowsTail, cube)
  if not outputDistinct or not outputDisjoint or not repeatedDisjoint:
    raise newException(ValueError, "logsumexp outputs must be independent allocations disjoint from every input")
  if valuesJson(output) != valuesJson(repeated):
    raise newException(ValueError, "Repeated logsumexp outputs are not deterministic")
  %*{
    "input_dtype": dtypeName[InputT](),
    "output_dtype": dtypeName[OutputT](),
    "output_shape": shapeJson(output),
    "output_values": valuesJson(output),
    "output_is_contiguous": output.is_C_contiguous,
    "output_disjoint_from_inputs": outputDisjoint,
    "output_distinct_from_repeat": outputDistinct,
  }

proc consume[T](value: Tensor[T]) =
  ## The memory clobber exposes every eager output write without another reduction.
  let address = cast[uint](value.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`address`)
    : "memory"
  """

proc run[T](manifest: JsonNode; preflightOnly: bool): JsonNode =
  if manifest.kind != JObject or manifest.len != manifestFields.len:
    raise newException(ValueError, "Logsumexp manifest has missing or unknown fields")
  for field in manifestFields:
    if not manifest.hasKey(field):
      raise newException(ValueError, "Logsumexp manifest has missing or unknown fields")
  if manifest["input_dtype"].kind != JString or manifest["input_dtype"].getStr != dtypeName[T]() or
      manifest["output_dtype"].kind != JString or
      manifest["output_dtype"].getStr != dtypeName[logsumexpOutput(T)]():
    raise newException(ValueError, "Logsumexp dtype fields do not match dispatched input and output types")
  if manifest["rows"].kind != JInt or manifest["columns"].kind != JInt or
      manifest["rows"].getInt != rows or manifest["columns"].getInt != columns or
      manifest["shape"] != %*[rows, columns] or manifest["cube_shape"] != %*[16, 32, 32]:
    raise newException(ValueError, "Logsumexp input shape metadata mismatch")
  if manifest["cube_values"] != manifest["values"]:
    raise newException(ValueError, "Logsumexp cube values must equal base matrix values")
  for (field, expected) in [("warmup", warmupCount), ("repeats", repeatCount),
                            ("iterations", callsPerSample)]:
    if manifest[field].kind != JInt or manifest[field].getInt != expected:
      raise newException(ValueError, "Logsumexp timing protocol mismatch")
  let suppliedProfiles = manifest["profiles"]
  if suppliedProfiles.kind != JArray or suppliedProfiles.len != profiles.len:
    raise newException(ValueError, "Logsumexp profile manifest mismatch")
  for index, profile in profiles:
    if suppliedProfiles[index].kind != JString or suppliedProfiles[index].getStr != profile:
      raise newException(ValueError, "Logsumexp ordered profile manifest mismatch")
  let suppliedActions = manifest["actions"]
  if suppliedActions.kind != JArray or suppliedActions.len != actions.len:
    raise newException(ValueError, "Logsumexp manifest must contain exactly 18 actions")
  for index, action in actions:
    if suppliedActions[index].kind != JString or suppliedActions[index].getStr != action:
      raise newException(ValueError, "Logsumexp ordered action manifest mismatch")

  let base = loadMatrix[T](manifest["values"], "values", 0)
  let rowsHead = loadMatrix[T](manifest["rows_head_values"], "rows_head_values", 1)
  let rowsTail = loadMatrix[T](manifest["rows_tail_values"], "rows_tail_values", 2)
  let cube = base.reshape(cubeShape[0], cubeShape[1], cubeShape[2])
  if not base.is_C_contiguous or not rowsHead.is_C_contiguous or not rowsTail.is_C_contiguous or
      not cube.is_C_contiguous or not rangesDisjoint(base, rowsHead) or
      not rangesDisjoint(base, rowsTail) or not rangesDisjoint(rowsHead, rowsTail) or
      cast[uint](cube.get_offset_ptr) != cast[uint](base.get_offset_ptr) or
      cube.size != base.size:
    raise newException(ValueError, "Logsumexp inputs must be independent resident matrices with a base-sharing cube view")
  var boundActions = newSeq[BoundAction[T]](actions.len)
  for index, name in actions:
    boundActions[index] = bindAction(name, base, cube, rowsHead, rowsTail)

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
    raise newException(ValueError, "Logsumexp manifest must declare a string input dtype")
  case manifest["input_dtype"].getStr
  of "bool": run[bool](manifest, preflightOnly)
  of "int8": run[int8](manifest, preflightOnly)
  of "int16": run[int16](manifest, preflightOnly)
  of "int32": run[int32](manifest, preflightOnly)
  of "int64": run[int64](manifest, preflightOnly)
  of "uint8": run[uint8](manifest, preflightOnly)
  of "uint16": run[uint16](manifest, preflightOnly)
  of "uint32": run[uint32](manifest, preflightOnly)
  of "uint64": run[uint64](manifest, preflightOnly)
  of "float32": run[float32](manifest, preflightOnly)
  of "float64": run[float64](manifest, preflightOnly)
  of "complex64": run[Complex[float32]](manifest, preflightOnly)
  of "complex128": run[Complex[float64]](manifest, preflightOnly)
  else: raise newException(ValueError, "Unsupported logsumexp benchmark dtype")

when isMainModule:
  if paramCount() < 1 or paramCount() > 2:
    echo "Usage: logsumexp_runner INPUT.json [--preflight]"
    quit(if paramCount() == 0: 0 else: 2)
  let preflightOnly = paramCount() == 2 and paramStr(2) == "--preflight"
  if paramCount() == 2 and not preflightOnly:
    echo "Usage: logsumexp_runner INPUT.json [--preflight]"
    quit(2)
  let manifest = parseFile(paramStr(1))
  echo runForDtype(manifest, preflightOnly)
