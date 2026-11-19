# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating CPU public logcumsumexp calls for the pinned benchmark.
import std/[algorithm, complex, json, math, monotimes, os, times]
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
    "rows_head_values", "rows_tail_values", "profiles", "actions",
    "warmup", "repeats", "iterations",
  ]

type
  BoundAction[T] = object
    input: Tensor[T]
    axis: int

proc dtypeName[T](): string =
  when T is float32: "float32"
  elif T is float64: "float64"
  elif T is Complex[float32]: "complex64"
  elif T is Complex[float64]: "complex128"
  else: {.error: "Unsupported logcumsumexp benchmark dtype".}

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

proc expectedShape[T](action: BoundAction[T]): Metadata =
  action.input.shape

proc bindAction[T](name: string; base, cube, rowsHead, rowsTail, vector: Tensor[T]): BoundAction[T] =
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
    raise newException(ValueError, "Unknown logcumsumexp profile: " & name)

proc execute[T](action: BoundAction[T]): Tensor[T] =
  ## Input is resident; every invocation returns a fresh public API result.
  action.input.logcumsumexp(action.axis)

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

proc observe[T](action: BoundAction[T]; output, repeated: Tensor[T];
    base, rowsHead, rowsTail, cube, vector: Tensor[T]): JsonNode =
  let expected = expectedShape(action)
  if output.shape != expected or repeated.shape != expected or
      not output.is_C_contiguous or not repeated.is_C_contiguous:
    raise newException(ValueError, "logcumsumexp output dtype, shape, or contiguous layout mismatch")
  let outputDisjoint = disjointFromInputs(output, base, rowsHead, rowsTail, cube, vector)
  let repeatedDisjoint = disjointFromInputs(repeated, base, rowsHead, rowsTail, cube, vector)
  let outputDistinct = rangesDisjoint(output, repeated)
  if not outputDisjoint or not repeatedDisjoint or not outputDistinct:
    raise newException(ValueError, "logcumsumexp outputs must be independent allocations disjoint from every input")
  if not sameValues(output, repeated):
    raise newException(ValueError, "Repeated logcumsumexp outputs differ, including signed-zero or nonfinite categories")
  %*{
    "input_dtype": dtypeName[T](),
    "output_dtype": dtypeName[T](),
    "output_shape": shapeJson(output),
    "output_values": valuesJson(output),
    "output_data_ptr": $cast[uint](output.get_offset_ptr),
    "output_is_contiguous": output.is_C_contiguous,
    "output_disjoint_from_inputs": outputDisjoint,
    "output_distinct_from_repeat": outputDistinct,
  }

proc consume[T](value: Tensor[T]) =
  ## Expose the full eager allocation with a pointer input and compiler barrier only.
  let address = cast[uint](value.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`address`)
    : "memory"
  """

proc run[T](manifest: JsonNode; preflightOnly: bool): JsonNode =
  if manifest.kind != JObject or manifest.len != manifestFields.len:
    raise newException(ValueError, "Logcumsumexp manifest has missing or unknown fields")
  for field in manifestFields:
    if not manifest.hasKey(field):
      raise newException(ValueError, "Logcumsumexp manifest has missing or unknown fields")
  if manifest["input_dtype"].kind != JString or manifest["input_dtype"].getStr != dtypeName[T]() or
      manifest["output_dtype"].kind != JString or manifest["output_dtype"].getStr != dtypeName[T]():
    raise newException(ValueError, "Logcumsumexp dtype fields do not match dispatched types")
  if manifest["rows"].kind != JInt or manifest["columns"].kind != JInt or
      manifest["rows"].getInt != rows or manifest["columns"].getInt != columns or
      manifest["shape"] != %*[rows, columns] or manifest["cube_shape"] != %*[16, 32, 32] or
      manifest["vector_shape"] != %*[elementCount]:
    raise newException(ValueError, "Logcumsumexp input shape metadata mismatch")
  if manifest["cube_values"] != manifest["values"] or
      manifest["vector_values"] != manifest["values"]:
    raise newException(ValueError, "Logcumsumexp cube/vector values must equal base matrix values")
  for (field, expected) in [("warmup", warmupCount), ("repeats", repeatCount),
                            ("iterations", callsPerSample)]:
    if manifest[field].kind != JInt or manifest[field].getInt != expected:
      raise newException(ValueError, "Logcumsumexp timing protocol mismatch")
  let suppliedProfiles = manifest["profiles"]
  if suppliedProfiles.kind != JArray or suppliedProfiles.len != profiles.len:
    raise newException(ValueError, "Logcumsumexp profile manifest mismatch")
  for index, profile in profiles:
    if suppliedProfiles[index].kind != JString or suppliedProfiles[index].getStr != profile:
      raise newException(ValueError, "Logcumsumexp ordered profile manifest mismatch")
  let suppliedActions = manifest["actions"]
  if suppliedActions.kind != JArray or suppliedActions.len != actions.len:
    raise newException(ValueError, "Logcumsumexp manifest must contain exactly 8 actions")
  for index, action in actions:
    if suppliedActions[index].kind != JString or suppliedActions[index].getStr != action:
      raise newException(ValueError, "Logcumsumexp ordered action manifest mismatch")

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
    raise newException(ValueError, "Logcumsumexp requires three independent resident matrices and base-sharing cube/vector views")
  var boundActions = newSeq[BoundAction[T]](actions.len)
  for index, name in actions:
    boundActions[index] = bindAction(name, base, cube, rowsHead, rowsTail, vector)

  var report = newJObject()
  for index, name in actions:
    let action = boundActions[index]
    let output = execute(action)
    let repeated = execute(action)
    let observation = observe(action, output, repeated, base, rowsHead, rowsTail, cube, vector)
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
    raise newException(ValueError, "Logcumsumexp manifest must declare a string input dtype")
  case manifest["input_dtype"].getStr
  of "float32": run[float32](manifest, preflightOnly)
  of "float64": run[float64](manifest, preflightOnly)
  of "complex64": run[Complex[float32]](manifest, preflightOnly)
  of "complex128": run[Complex[float64]](manifest, preflightOnly)
  else: raise newException(ValueError, "Unsupported logcumsumexp benchmark dtype")

when isMainModule:
  if paramCount() < 1 or paramCount() > 2:
    echo "Usage: logcumsumexp_runner INPUT.json [--preflight]"
    quit(if paramCount() == 0: 0 else: 2)
  let preflightOnly = paramCount() == 2 and paramStr(2) == "--preflight"
  if paramCount() == 2 and not preflightOnly:
    echo "Usage: logcumsumexp_runner INPUT.json [--preflight]"
    quit(2)
  let manifest = parseFile(paramStr(1))
  echo runForDtype(manifest, preflightOnly)
