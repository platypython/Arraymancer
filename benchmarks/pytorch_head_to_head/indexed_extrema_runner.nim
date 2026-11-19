# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures public allocating CPU argmin/argmax calls for the pinned benchmark.
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
    "argmin_flat_whole_head", "argmin_flat_whole_tail", "argmin_flat_rows",
    "argmin_flat_columns", "argmin_flat_cube_axis0", "argmin_flat_cube_axis1",
    "argmin_flat_cube_axis2",
    "argmin_keepdim_whole_head", "argmin_keepdim_whole_tail", "argmin_keepdim_rows",
    "argmin_keepdim_columns", "argmin_keepdim_cube_axis0", "argmin_keepdim_cube_axis1",
    "argmin_keepdim_cube_axis2",
    "argmax_flat_whole_head", "argmax_flat_whole_tail", "argmax_flat_rows",
    "argmax_flat_columns", "argmax_flat_cube_axis0", "argmax_flat_cube_axis1",
    "argmax_flat_cube_axis2",
    "argmax_keepdim_whole_head", "argmax_keepdim_whole_tail", "argmax_keepdim_rows",
    "argmax_keepdim_columns", "argmax_keepdim_cube_axis0", "argmax_keepdim_cube_axis1",
    "argmax_keepdim_cube_axis2",
  ]

type
  Operation = enum
    opArgmin, opArgmax
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
  else: {.error: "Unsupported indexed-extrema benchmark dtype".}

proc lowFence[T](): T =
  when T is SomeFloat: T(-1024)
  else: low(T)

proc highFence[T](): T =
  when T is SomeFloat: T(1024)
  else: high(T)

proc expectedValue[T](index, fenceMode: int): T =
  if index == 0:
    case fenceMode
    of 1: return highFence[T]()
    of 2: return lowFence[T]()
    else: discard
  elif index == elementCount - 1:
    case fenceMode
    of 1: return lowFence[T]()
    of 2: return highFence[T]()
    else: discard
  T(index mod 63 + 1)

proc loadMatrix[T](values: JsonNode; label: string; fenceMode: int): Tensor[T] =
  if values.kind != JArray or values.len != elementCount:
    raise newException(ValueError, label & " must contain exactly 16,384 values")
  result = newTensorUninit[T](rows, columns)
  let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
  for index in 0 ..< elementCount:
    var value: T
    when T is SomeInteger:
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
    if value != expectedValue[T](index, fenceMode):
      raise newException(ValueError, label & " values do not match the deterministic profile")
    data[index] = value

proc bindAction[T](name: string; base, cube, wholeHead, wholeTail: Tensor[T]): BoundAction[T] =
  let pieces = name.split('_')
  if pieces.len < 3 or pieces[1] notin ["flat", "keepdim"]:
    raise newException(ValueError, "Malformed indexed-extrema action: " & name)
  case pieces[0]
  of "argmin": result.operation = opArgmin
  of "argmax": result.operation = opArgmax
  else: raise newException(ValueError, "Unknown indexed-extrema operation: " & pieces[0])
  result.keepdim = pieces[1] == "keepdim"
  let profile = pieces[2 .. ^1].join("_")
  case profile
  of "whole_head":
    result.input = wholeHead
    result.axis = -1
  of "whole_tail":
    result.input = wholeTail
    result.axis = -1
  of "rows":
    result.input = base
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
  else: raise newException(ValueError, "Unknown indexed-extrema profile: " & profile)

proc execute[T](action: BoundAction[T]): Tensor[int64] =
  case action.operation
  of opArgmin:
    if action.axis < 0:
      action.input.argmin(keepdim = action.keepdim)
    else:
      action.input.argmin(action.axis, keepdim = action.keepdim)
  of opArgmax:
    if action.axis < 0:
      action.input.argmax(keepdim = action.keepdim)
    else:
      action.input.argmax(action.axis, keepdim = action.keepdim)

proc observe(value: Tensor[int64]; inputDtype: string): JsonNode =
  var shape = newJArray()
  for dimension in value.shape:
    shape.add %dimension
  var values = newJArray()
  for item in value:
    values.add %($item)
  %*{"input_dtype": inputDtype, "dtype": "int64", "shape": shape, "values": values}

proc consume(value: Tensor[int64]) =
  ## Make each actual allocated result observable without reading its contents.
  let address = cast[uint](value.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`address`)
    : "memory"
  """

proc run[T](manifest: JsonNode): JsonNode =
  const manifestFields = [
    "dtype", "output_dtype", "rows", "columns", "shape", "values",
    "cube_shape", "cube_values", "whole_head_values", "whole_tail_values",
    "actions", "warmup", "repeats", "iterations",
  ]
  if manifest.kind != JObject or manifest.len != manifestFields.len:
    raise newException(ValueError, "Indexed-extrema manifest has missing or unknown fields")
  for field in manifestFields:
    if not manifest.hasKey(field):
      raise newException(ValueError, "Indexed-extrema manifest has missing or unknown fields")
  if manifest["dtype"].kind != JString or manifest["dtype"].getStr != dtypeName[T]() or
      manifest["output_dtype"].kind != JString or manifest["output_dtype"].getStr != "int64":
    raise newException(ValueError, "Indexed-extrema dtype fields do not match dispatched input and output types")
  if manifest["rows"].kind != JInt or manifest["columns"].kind != JInt or
      manifest["rows"].getInt != rows or manifest["columns"].getInt != columns or
      manifest["shape"] != %*[rows, columns] or manifest["cube_shape"] != %*[16, 32, 32]:
    raise newException(ValueError, "Indexed-extrema input shape metadata mismatch")
  if manifest["cube_values"] != manifest["values"]:
    raise newException(ValueError, "Indexed-extrema cube values must equal the base matrix values")
  for (field, expected) in [("warmup", warmupCount), ("repeats", repeatCount),
                            ("iterations", callsPerSample)]:
    if manifest[field].kind != JInt or manifest[field].getInt != expected:
      raise newException(ValueError, "Indexed-extrema timing protocol mismatch")
  let supplied = manifest["actions"]
  if supplied.kind != JArray or supplied.len != actions.len:
    raise newException(ValueError, "Indexed-extrema manifest must contain exactly 28 actions")
  for index, action in actions:
    if supplied[index].kind != JString or supplied[index].getStr != action:
      raise newException(ValueError, "Indexed-extrema ordered action manifest mismatch")

  let base = loadMatrix[T](manifest["values"], "values", 0)
  let wholeHead = loadMatrix[T](manifest["whole_head_values"], "whole_head_values", 1)
  let wholeTail = loadMatrix[T](manifest["whole_tail_values"], "whole_tail_values", 2)
  let cube = base.reshape(cubeShape[0], cubeShape[1], cubeShape[2])
  var boundActions = newSeq[BoundAction[T]](actions.len)
  for index, name in actions:
    boundActions[index] = bindAction(name, base, cube, wholeHead, wholeTail)

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
  if manifest.kind != JObject or not manifest.hasKey("dtype") or manifest["dtype"].kind != JString:
    raise newException(ValueError, "Indexed-extrema manifest must declare a string input dtype")
  case manifest["dtype"].getStr
  of "float32": run[float32](manifest)
  of "float64": run[float64](manifest)
  of "int8": run[int8](manifest)
  of "int16": run[int16](manifest)
  of "int32": run[int32](manifest)
  of "int64": run[int64](manifest)
  of "uint8": run[uint8](manifest)
  else: raise newException(ValueError, "Unsupported indexed-extrema benchmark dtype")

when isMainModule:
  if paramCount() != 1:
    echo "Usage: indexed_extrema_runner INPUT.json"
    quit(if paramCount() == 0: 0 else: 2)
  let manifest = parseFile(paramStr(1))
  echo runForDtype(manifest)
