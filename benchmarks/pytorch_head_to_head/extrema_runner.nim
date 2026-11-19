# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures public allocating CPU amin/amax overloads for the pinned benchmark.
import std/[algorithm, json, math, monotimes, os, times]
import arraymancer

const actions = [
  "amin_flat_whole", "amin_flat_rows", "amin_flat_columns",
  "amin_flat_axes_0_2", "amin_flat_axes_2_1",
  "amin_keepdim_whole", "amin_keepdim_rows", "amin_keepdim_columns",
  "amin_keepdim_axes_0_2", "amin_keepdim_axes_2_1",
  "amax_flat_whole", "amax_flat_rows", "amax_flat_columns",
  "amax_flat_axes_0_2", "amax_flat_axes_2_1",
  "amax_keepdim_whole", "amax_keepdim_rows", "amax_keepdim_columns",
  "amax_keepdim_axes_0_2", "amax_keepdim_axes_2_1",
]

proc dtypeName[T](): string =
  when T is bool: "bool"
  elif T is int8: "int8"
  elif T is int16: "int16"
  elif T is int32: "int32"
  elif T is int64: "int64"
  elif T is uint8: "uint8"
  elif T is float32: "float32"
  elif T is float64: "float64"
  else: {.error: "Unsupported extrema benchmark dtype".}

proc loadInput[T](manifest: JsonNode): Tensor[T] =
  let rows = manifest["rows"].getInt
  let columns = manifest["columns"].getInt
  if rows != 128 or columns != 128 or manifest["shape"] != %*[128, 128]:
    raise newException(ValueError, "Extrema benchmark input must be 128x128")
  let values = manifest["values"]
  if values.kind != JArray or values.len != rows * columns:
    raise newException(ValueError, "Extrema value count does not match input shape")
  result = newTensorUninit[T](rows, columns)
  let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
  for index in 0 ..< result.size:
    when T is bool:
      if values[index].kind != JBool:
        raise newException(ValueError, "Boolean extrema inputs must be JSON booleans")
      data[index] = values[index].getBool
    elif T is SomeInteger:
      if values[index].kind != JInt:
        raise newException(ValueError, "Integer extrema inputs must be JSON integers")
      data[index] = T(values[index].getInt)
    else:
      if values[index].kind notin {JInt, JFloat}:
        raise newException(ValueError, "Float extrema inputs must be JSON numbers")
      let value = values[index].getFloat
      if value.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, "Extrema inputs must be finite")
      data[index] = T(value)

proc execute[T](name: string; matrix: Tensor[T]; cube: Tensor[T]): Tensor[T] =
  case name
  of "amin_flat_whole": matrix.amin()
  of "amin_keepdim_whole": matrix.amin(keepdim = true)
  of "amax_flat_whole": matrix.amax()
  of "amax_keepdim_whole": matrix.amax(keepdim = true)
  of "amin_flat_rows": matrix.amin(1)
  of "amin_keepdim_rows": matrix.amin(1, keepdim = true)
  of "amax_flat_rows": matrix.amax(1)
  of "amax_keepdim_rows": matrix.amax(1, keepdim = true)
  of "amin_flat_columns": matrix.amin(0)
  of "amin_keepdim_columns": matrix.amin(0, keepdim = true)
  of "amax_flat_columns": matrix.amax(0)
  of "amax_keepdim_columns": matrix.amax(0, keepdim = true)
  of "amin_flat_axes_0_2": cube.amin([0, 2])
  of "amin_keepdim_axes_0_2": cube.amin([0, 2], keepdim = true)
  of "amax_flat_axes_0_2": cube.amax([0, 2])
  of "amax_keepdim_axes_0_2": cube.amax([0, 2], keepdim = true)
  of "amin_flat_axes_2_1": cube.amin([2, 1])
  of "amin_keepdim_axes_2_1": cube.amin([2, 1], keepdim = true)
  of "amax_flat_axes_2_1": cube.amax([2, 1])
  of "amax_keepdim_axes_2_1": cube.amax([2, 1], keepdim = true)
  else: raise newException(ValueError, "Unknown extrema action: " & name)

proc jsonFloat(value: float64): JsonNode =
  if value.isNaN: %"NaN"
  elif value == Inf: %"+Inf"
  elif value == NegInf: %"-Inf"
  else: %value

proc observe[T](input: Tensor[T]; sourceDtype: string): JsonNode =
  var shape = newJArray()
  for dimension in input.shape:
    shape.add %dimension
  var values = newJArray()
  for value in input:
    when T is bool:
      values.add %value
    elif T is SomeInteger:
      values.add %($value)
    else:
      values.add jsonFloat(float64(value))
  %*{"input_dtype": sourceDtype, "dtype": dtypeName[T](),
     "shape": shape, "values": values}

proc consume[T](input: Tensor[T]) =
  ## Make the allocated result observable while retaining its allocation cost.
  let address = cast[uint](input.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`address`)
    : "memory"
  """

proc run[T](manifest: JsonNode): JsonNode =
  const manifestKeys = [
    "dtype", "output_dtype", "rows", "columns", "shape", "values",
    "actions", "warmup", "repeats", "iterations",
  ]
  if manifest.len != manifestKeys.len:
    raise newException(ValueError, "Extrema manifest has missing or unknown fields")
  for key in manifestKeys:
    if not manifest.hasKey(key):
      raise newException(ValueError, "Extrema manifest has missing or unknown fields")
  if manifest["dtype"].getStr != dtypeName[T]() or
      manifest["output_dtype"].getStr != dtypeName[T]():
    raise newException(ValueError, "Extrema dtype fields do not match dispatched type")
  let supplied = manifest["actions"]
  if supplied.kind != JArray or supplied.len != actions.len:
    raise newException(ValueError, "Extrema action manifest must contain exactly 20 actions")
  for index, action in actions:
    if supplied[index].kind != JString or supplied[index].getStr != action:
      raise newException(ValueError, "Extrema action manifest differs from required ordered action list")
  for (key, expected) in [("warmup", 50), ("repeats", 100), ("iterations", 50)]:
    if manifest[key].kind != JInt or manifest[key].getInt != expected:
      raise newException(ValueError, "Extrema timing manifest does not match required protocol")
  let matrix = loadInput[T](manifest)
  let cube = matrix.reshape(16, 32, 32)
  var report = newJObject()
  for name in actions:
    let observation = observe(execute(name, matrix, cube), dtypeName[T]())
    for _ in 0 ..< 50:
      consume(execute(name, matrix, cube))
    var samples = newSeq[float64](100)
    for sample in 0 ..< 100:
      let start = getMonoTime()
      for _ in 0 ..< 50:
        consume(execute(name, matrix, cube))
      samples[sample] = float64((getMonoTime() - start).inNanoseconds) / 1e6 / 50.0
    var ordered = samples
    ordered.sort()
    let middle = samples.len div 2
    let median = (ordered[middle - 1] + ordered[middle]) / 2.0
    report[name] = %*{"observation": observation, "samples_ms": samples,
      "median_ms": median, "p95_ms": ordered[int(ceil(0.95 * float64(samples.len))) - 1]}
  report

proc runForDtype(manifest: JsonNode): JsonNode =
  case manifest["dtype"].getStr
  of "bool": run[bool](manifest)
  of "int8": run[int8](manifest)
  of "int16": run[int16](manifest)
  of "int32": run[int32](manifest)
  of "int64": run[int64](manifest)
  of "uint8": run[uint8](manifest)
  of "float32": run[float32](manifest)
  of "float64": run[float64](manifest)
  else: raise newException(ValueError, "Unsupported extrema benchmark dtype")

when isMainModule:
  if paramCount() != 1:
    echo "Usage: extrema_runner INPUT.json"
    quit(if paramCount() == 0: 0 else: 2)
  let manifest = parseFile(paramStr(1))
  echo runForDtype(manifest)
