# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating public CPU cumulative min/max-with-indices calls.
import std/[algorithm, complex, json, math, monotimes, os, strutils, times]
import arraymancer

const
  rows = 128
  columns = 128
  elementCount = rows * columns
  cubeShape = [16, 32, 32]
  offsetShape = [rows, columns + 2]
  emptyShape = [2, 0, 3]
  warmupCount = 50
  repeatCount = 100
  callsPerSample = 500
  operations = ["cummin", "cummax"]
  nonemptyDtypes = ["bool", "uint8", "int8", "int16", "int32", "int64", "float32", "float64"]
  nonemptyProfiles = [
    "rows", "rows_head", "rows_tail", "columns", "cube_axis0", "cube_axis1",
    "cube_axis2", "vector", "transpose", "offset", "reverse", "broadcast",
  ]
  specialProfiles = ["scalar", "empty"]
  profiles = [
    "rows", "rows_head", "rows_tail", "columns", "cube_axis0", "cube_axis1",
    "cube_axis2", "vector", "transpose", "offset", "reverse", "broadcast", "scalar", "empty",
  ]
  manifestFields = [
    "input_dtype", "values_dtype", "indices_dtype", "rows", "columns", "shape",
    "cube_shape", "vector_shape", "offset_backing_shape", "scalar_shape", "scalar_value",
    "empty_shape", "values", "rows_head_values", "rows_tail_values",
    "offset_backing_values", "profiles", "actions", "profile_value_counts",
    "profile_layouts", "warmup", "repeats", "iterations",
  ]
  layoutFields = [
    "shape", "axis", "torch_strides", "torch_storage_offset", "torch_storage_owner",
    "nim_strides", "nim_storage_offset", "nim_storage_owner",
  ]

type
  Operation = enum
    opCummin, opCummax
  ResidentInputs[T] = object
    base, rowsHead, rowsTail: Tensor[T]
    cube, vector, transpose: Tensor[T]
    offsetBacking, offset, reverse, broadcast: Tensor[T]
    scalar, empty: Tensor[T]
  BoundAction[T] = object
    input: Tensor[T]
    operation: Operation
    axis: int
    profile: string
  ActionProgress = object
    preflightCallsAttempted, preflightCallsCompleted: int
    warmupCallsAttempted, warmupCallsCompleted: int
    sampleCallsAttempted, sampleCallsCompleted, samplesCompleted: int

proc dtypeName[T](): string =
  when T is bool: "bool"
  elif T is uint8: "uint8"
  elif T is uint16: "uint16"
  elif T is uint32: "uint32"
  elif T is uint64: "uint64"
  elif T is int8: "int8"
  elif T is int16: "int16"
  elif T is int32: "int32"
  elif T is int64: "int64"
  elif T is float32: "float32"
  elif T is float64: "float64"
  elif T is Complex[float32]: "complex64"
  elif T is Complex[float64]: "complex128"
  else: {.error: "Unsupported cumulative-extrema benchmark dtype".}

proc actionsFor(dtype: string): seq[string] =
  result = @[]
  for operation in operations:
    if dtype in nonemptyDtypes:
      for profile in nonemptyProfiles:
        result.add operation & "_" & profile
    for profile in specialProfiles:
      result.add operation & "_" & profile

proc intArray(node: JsonNode; label: string): seq[int] =
  if node.kind != JArray:
    raise newException(ValueError, label & " must be a JSON array")
  result = newSeq[int](node.len)
  for index in 0 ..< node.len:
    let item = node[index]
    if item.kind != JInt:
      raise newException(ValueError, label & " must contain only JSON integers")
    result[index] = int(item.getInt)

proc jsonArray(values: openArray[int]): JsonNode =
  result = newJArray()
  for value in values:
    result.add %value

proc expectedShape(profile: string): seq[int] =
  case profile
  of "rows", "rows_head", "rows_tail", "columns", "transpose", "offset", "reverse", "broadcast":
    @[rows, columns]
  of "cube_axis0", "cube_axis1", "cube_axis2":
    @[cubeShape[0], cubeShape[1], cubeShape[2]]
  of "vector": @[elementCount]
  of "scalar": @[]
  of "empty": @[emptyShape[0], emptyShape[1], emptyShape[2]]
  else: raise newException(ValueError, "Unknown cumulative-extrema profile: " & profile)

proc expectedAxis(profile: string): int =
  case profile
  of "rows", "rows_head", "rows_tail", "transpose", "offset", "reverse": 1
  of "columns", "cube_axis0", "broadcast", "scalar": 0
  of "cube_axis1", "empty": 1
  of "cube_axis2": 2
  of "vector": 0
  else: raise newException(ValueError, "Unknown cumulative-extrema profile: " & profile)

proc expectedNimStrides(profile: string): seq[int] =
  case profile
  of "rows", "rows_head", "rows_tail", "columns": @[columns, 1]
  of "cube_axis0", "cube_axis1", "cube_axis2": @[1024, 32, 1]
  of "vector": @[1]
  of "transpose": @[1, rows]
  of "offset": @[columns + 2, 1]
  of "reverse": @[-columns, -1]
  of "broadcast": @[0, 1]
  of "scalar": @[]
  of "empty": @[0, 3, 1]
  else: raise newException(ValueError, "Unknown cumulative-extrema profile: " & profile)

proc expectedNimOffset(profile: string): int =
  case profile
  of "offset": 1
  of "reverse": elementCount - 1
  else: 0

proc expectedTorchOwner(profile: string): string =
  case profile
  of "rows", "columns", "cube_axis0", "cube_axis1", "cube_axis2", "vector", "transpose", "broadcast": "base"
  of "rows_head": "rows_head"
  of "rows_tail": "rows_tail"
  of "offset": "offset_backing"
  of "reverse": "reverse"
  of "scalar": "scalar"
  of "empty": "empty"
  else: raise newException(ValueError, "Unknown cumulative-extrema profile: " & profile)

proc expectedNimOwner(profile: string): string =
  case profile
  of "rows", "columns", "cube_axis0", "cube_axis1", "cube_axis2", "vector", "transpose", "reverse", "broadcast": "base"
  of "rows_head": "rows_head"
  of "rows_tail": "rows_tail"
  of "offset": "offset_backing"
  of "scalar": "scalar"
  of "empty": "empty"
  else: raise newException(ValueError, "Unknown cumulative-extrema profile: " & profile)

proc expectedCount(profile: string): int =
  if profile == "scalar":
    return 1
  if profile == "empty":
    return 0
  result = 1
  for dimension in expectedShape(profile):
    result *= dimension

proc expectedContiguous(profile: string): bool =
  profile notin ["transpose", "offset", "reverse", "broadcast"]

proc scalarValue[T](): T =
  when T is bool: true
  elif T is Complex:
    type Component = typeof(default(T).re)
    T(re: Component(2.0), im: Component(0.25))
  else: T(3)

proc baseValue[T](index: int): T =
  when T is bool: (index mod 31) != 0
  elif T is Complex: {.error: "Complex values are benchmarked only for scalar and empty inputs".}
  else: T(index mod 31)

proc matrixValue[T](index: int; profile: string): T =
  let column = index mod columns
  when T is bool:
    if profile == "rows_head":
      if column == 0: return true
      if column == columns - 1: return false
    elif profile == "rows_tail":
      if column == 0: return false
      if column == columns - 1: return true
    (index mod 31) != 0
  else:
    if profile == "rows_head":
      if column == 0: return T(63)
      if column == columns - 1: return T(0)
    elif profile == "rows_tail":
      if column == 0: return T(0)
      if column == columns - 1: return T(63)
    T(index mod 31)

proc sourceValue[T](profile: string; index: int): T =
  when T is Complex:
    {.error: "Complex values are benchmarked only for scalar and empty inputs".}
  else:
    case profile
    of "rows", "columns", "cube_axis0", "cube_axis1", "cube_axis2", "vector":
      baseValue[T](index)
    of "rows_head": matrixValue[T](index, "rows_head")
    of "rows_tail": matrixValue[T](index, "rows_tail")
    of "transpose":
      let sourceIndex = (index mod columns) * rows + index div columns
      baseValue[T](sourceIndex)
    of "offset":
      let sourceIndex = (index div columns) * (columns + 2) + index mod columns + 1
      baseValue[T](sourceIndex)
    of "reverse":
      let sourceIndex = (rows - 1 - index div columns) * columns + columns - 1 - index mod columns
      baseValue[T](sourceIndex)
    of "broadcast":
      baseValue[T](index mod columns)
    of "scalar": scalarValue[T]()
    else: raise newException(ValueError, "Unexpected source-value profile: " & profile)

proc numberValue(node: JsonNode; label: string): float64 =
  case node.kind
  of JInt: float64(node.getInt)
  of JFloat: node.getFloat
  else: raise newException(ValueError, label & " must be a JSON number")

proc parseValue[T](node: JsonNode; label: string): T =
  when T is bool:
    if node.kind != JBool:
      raise newException(ValueError, label & " bool values must be exact JSON booleans")
    node.getBool
  elif T is Complex:
    if node.kind != JArray or node.len != 2:
      raise newException(ValueError, label & " complex scalar must be [real, imag]")
    type Component = typeof(default(T).re)
    T(re: Component(numberValue(node[0], label)), im: Component(numberValue(node[1], label)))
  elif T is SomeUnsignedInt:
    if node.kind != JInt:
      raise newException(ValueError, label & " unsigned values must be JSON integers")
    let exact = node.getInt
    if exact < 0:
      raise newException(ValueError, label & " unsigned value is negative")
    when T is uint64:
      T(exact)
    else:
      if uint64(exact) > uint64(high(T)):
        raise newException(ValueError, label & " unsigned value is outside the input dtype")
      T(exact)
  elif T is SomeInteger:
    if node.kind != JInt:
      raise newException(ValueError, label & " integer values must be JSON integers")
    let exact = node.getInt
    if exact < int64(low(T)) or exact > int64(high(T)):
      raise newException(ValueError, label & " integer value is outside the input dtype")
    T(exact)
  else:
    if node.kind notin {JInt, JFloat}:
      raise newException(ValueError, label & " float values must be JSON numbers")
    let exact = numberValue(node, label)
    T(exact)

proc componentBits[T: SomeFloat](value: T): string =
  when T is float32:
    toHex(cast[uint32](value), 8).toLowerAscii
  else:
    toHex(cast[uint64](value), 16).toLowerAscii

proc bitString[T](value: T): string =
  when T is bool:
    if value: "01" else: "00"
  elif T is Complex:
    componentBits(value.re) & componentBits(value.im)
  elif T is SomeFloat:
    componentBits(value)
  elif T is SomeUnsignedInt:
    when sizeof(T) == 1: toHex(cast[uint8](value), 2).toLowerAscii
    elif sizeof(T) == 2: toHex(cast[uint16](value), 4).toLowerAscii
    elif sizeof(T) == 4: toHex(cast[uint32](value), 8).toLowerAscii
    else: toHex(cast[uint64](value), 16).toLowerAscii
  else:
    when sizeof(T) == 1: toHex(cast[uint8](value), 2).toLowerAscii
    elif sizeof(T) == 2: toHex(cast[uint16](value), 4).toLowerAscii
    elif sizeof(T) == 4: toHex(cast[uint32](value), 8).toLowerAscii
    else: toHex(cast[uint64](value), 16).toLowerAscii

proc sameValue[T](left, right: T): bool =
  when T is SomeFloat:
    bitString(left) == bitString(right)
  elif T is Complex:
    bitString(left) == bitString(right)
  else:
    left == right

proc sameValues[T](left, right: Tensor[T]): bool =
  if left.shape != right.shape or left.size != right.size:
    return false
  if left.size == 0:
    return true
  let leftData = cast[ptr UncheckedArray[T]](left.get_data_ptr)
  let rightData = cast[ptr UncheckedArray[T]](right.get_data_ptr)
  for index in 0 ..< left.size:
    if not sameValue(leftData[index], rightData[index]):
      return false
  true

proc sameIndices(left, right: Tensor[int64]): bool =
  if left.shape != right.shape or left.size != right.size:
    return false
  if left.size == 0:
    return true
  let leftData = cast[ptr UncheckedArray[int64]](left.get_data_ptr)
  let rightData = cast[ptr UncheckedArray[int64]](right.get_data_ptr)
  for index in 0 ..< left.size:
    if leftData[index] != rightData[index]:
      return false
  true

proc shapeSeq[T](value: Tensor[T]): seq[int] =
  result = newSeq[int](value.shape.len)
  for index, dimension in value.shape:
    result[index] = dimension

proc stridesSeq[T](value: Tensor[T]): seq[int] =
  result = newSeq[int](value.strides.len)
  for index, stride in value.strides:
    result[index] = stride

proc tensorAt[T](sources: ResidentInputs[T]; profile: string): Tensor[T] =
  case profile
  of "rows", "columns": sources.base
  of "rows_head": sources.rowsHead
  of "rows_tail": sources.rowsTail
  of "cube_axis0", "cube_axis1", "cube_axis2": sources.cube
  of "vector": sources.vector
  of "transpose": sources.transpose
  of "offset": sources.offset
  of "reverse": sources.reverse
  of "broadcast": sources.broadcast
  of "scalar": sources.scalar
  of "empty": sources.empty
  else: raise newException(ValueError, "Unknown cumulative-extrema profile: " & profile)

proc storageOwnerPointer[T](sources: ResidentInputs[T]; owner: string): uint =
  case owner
  of "base": cast[uint](sources.base.get_data_ptr)
  of "rows_head": cast[uint](sources.rowsHead.get_data_ptr)
  of "rows_tail": cast[uint](sources.rowsTail.get_data_ptr)
  of "offset_backing": cast[uint](sources.offsetBacking.get_data_ptr)
  of "scalar": cast[uint](sources.scalar.get_data_ptr)
  of "empty": cast[uint](sources.empty.get_data_ptr)
  else: raise newException(ValueError, "Unknown cumulative-extrema storage owner: " & owner)

proc expectedValueCountNode(profile: string): JsonNode =
  %expectedCount(profile)


proc validateLayoutManifest(manifest: JsonNode) =
  let counts = manifest["profile_value_counts"]
  if counts.kind != JObject or counts.len != profiles.len:
    raise newException(ValueError, "Profile value-count map must contain all fourteen profiles")
  let layouts = manifest["profile_layouts"]
  if layouts.kind != JObject or layouts.len != profiles.len:
    raise newException(ValueError, "Profile layout map must contain all fourteen profiles")
  for profile in profiles:
    if not counts.hasKey(profile) or counts[profile] != expectedValueCountNode(profile):
      raise newException(ValueError, "Profile value count mismatch for " & profile)
    if not layouts.hasKey(profile):
      raise newException(ValueError, "Missing profile layout for " & profile)
    let item = layouts[profile]
    if item.kind != JObject or item.len != layoutFields.len:
      raise newException(ValueError, "Malformed profile layout for " & profile)
    for field in layoutFields:
      if not item.hasKey(field):
        raise newException(ValueError, "Missing layout field " & field & " for " & profile)
    if item["shape"] != jsonArray(expectedShape(profile)) or
        item["axis"].kind != JInt or item["axis"].getInt != expectedAxis(profile) or
        item["nim_strides"] != jsonArray(expectedNimStrides(profile)) or
        item["nim_storage_offset"].kind != JInt or
        item["nim_storage_offset"].getInt != expectedNimOffset(profile) or
        item["nim_storage_owner"].kind != JString or
        item["nim_storage_owner"].getStr != expectedNimOwner(profile) or
        item["torch_storage_owner"].kind != JString or
        item["torch_storage_owner"].getStr != expectedTorchOwner(profile) or
        item["torch_storage_offset"].kind != JInt:
      raise newException(ValueError, "Profile layout or axis mismatch for " & profile)
    let torchStrides = intArray(item["torch_strides"], profile & " torch_strides")
    if torchStrides.len != expectedShape(profile).len:
      raise newException(ValueError, "Torch stride rank mismatch for " & profile)

proc validateTensorLayout[T](sources: ResidentInputs[T]; profile: string) =
  let input = tensorAt(sources, profile)
  if shapeSeq(input) != expectedShape(profile) or
      stridesSeq(input) != expectedNimStrides(profile) or
      input.offset != expectedNimOffset(profile) or
      input.is_C_contiguous != expectedContiguous(profile):
    raise newException(ValueError, "Native input layout mismatch for " & profile)
  let expectedOwner = expectedNimOwner(profile)
  if input.size > 0 and cast[uint](input.get_data_ptr) != storageOwnerPointer(sources, expectedOwner):
    raise newException(ValueError, "Native input storage owner mismatch for " & profile)

proc validateProfileValues[T](sources: ResidentInputs[T]; profile: string) =
  when T is Complex:
    if profile == "scalar":
      let scalar = tensorAt(sources, profile)
      let data = cast[ptr UncheckedArray[T]](scalar.get_data_ptr)
      if not sameValue(data[0], scalarValue[T]()):
        raise newException(ValueError, "Complex scalar input differs from the exact deterministic source")
    elif profile == "empty":
      discard
    else:
      raise newException(ValueError, "Complex nonempty profiles are outside the benchmark contract")
  else:
    let input = tensorAt(sources, profile)
    var index = 0
    for item in input:
      if not sameValue(item, sourceValue[T](profile, index)):
        raise newException(ValueError, "Native logical input values mismatch profile " & profile)
      inc index
    if index != expectedCount(profile):
      raise newException(ValueError, "Native logical input value count mismatch for " & profile)

proc valueJson[T](value: T): JsonNode =
  when T is bool:
    %value
  elif T is Complex:
    %*{"real": valueJson(value.re), "imag": valueJson(value.im)}
  elif T is SomeFloat:
    let exact = float64(value)
    if exact.isNaN: newJString("NaN")
    elif exact == Inf: newJString("+Inf")
    elif exact == NegInf: newJString("-Inf")
    else: %exact
  elif T is SomeUnsignedInt:
    if uint64(value) > uint64(high(int64)):
      newJString($value)
    else:
      %int64(value)
  else:
    %int64(value)

proc valuesJson[T](value: Tensor[T]): JsonNode =
  result = newJArray()
  for item in value:
    result.add valueJson(item)

proc indicesJson(value: Tensor[int64]): JsonNode =
  result = newJArray()
  for item in value:
    result.add %item

proc valuesBitsJson[T](value: Tensor[T]): JsonNode =
  result = newJArray()
  for item in value:
    result.add %bitString(item)

proc byteRange[T](value: Tensor[T]): tuple[first, last: uint] =
  if value.size == 0:
    return (0'u, 0'u)
  result.first = cast[uint](value.get_data_ptr)
  result.last = result.first + uint(value.size) * uint(sizeof(T))

proc rangesDisjoint[A, B](left: Tensor[A]; right: Tensor[B]): bool =
  if left.size == 0 or right.size == 0:
    return true
  let leftRange = byteRange(left)
  let rightRange = byteRange(right)
  leftRange.last <= rightRange.first or rightRange.last <= leftRange.first

proc disjointFromInputs[T, U](output: Tensor[T]; sources: ResidentInputs[U]): bool =
  when U is Complex:
    rangesDisjoint(output, sources.scalar) and rangesDisjoint(output, sources.empty)
  else:
    rangesDisjoint(output, sources.base) and rangesDisjoint(output, sources.rowsHead) and
      rangesDisjoint(output, sources.rowsTail) and rangesDisjoint(output, sources.offsetBacking) and
      rangesDisjoint(output, sources.scalar) and rangesDisjoint(output, sources.empty)

proc outputsIndependent[T](left, right: tuple[values: Tensor[T], indices: Tensor[int64]]): bool =
  rangesDisjoint(left.values, right.values) and rangesDisjoint(left.values, right.indices) and
    rangesDisjoint(left.indices, right.values) and rangesDisjoint(left.indices, right.indices)

proc execute[T](action: BoundAction[T]): tuple[values: Tensor[T], indices: Tensor[int64]] =
  case action.operation
  of opCummin: action.input.cummin(action.axis)
  of opCummax: action.input.cummax(action.axis)

proc shapeJson[T](value: Tensor[T]): JsonNode =
  jsonArray(shapeSeq(value))

proc validateOutput[T](action: BoundAction[T]; pair: tuple[values: Tensor[T], indices: Tensor[int64]];
                       sources: ResidentInputs[T]) =
  let shape = expectedShape(action.profile)
  if shapeSeq(pair.values) != shape or shapeSeq(pair.indices) != shape or
      not pair.values.is_C_contiguous or not pair.indices.is_C_contiguous:
    raise newException(ValueError, "Cumulative-extrema output dtype, shape, or contiguous layout mismatch")
  if not disjointFromInputs(pair.values, sources) or not disjointFromInputs(pair.indices, sources):
    raise newException(ValueError, "Cumulative-extrema outputs must be independent of all resident inputs")
  if not rangesDisjoint(pair.values, pair.indices):
    raise newException(ValueError, "Cumulative-extrema values and indices must own distinct allocations")

proc observe[T](action: BoundAction[T]; first, peer, repeated: tuple[values: Tensor[T], indices: Tensor[int64]];
                sources: ResidentInputs[T]): JsonNode =
  validateOutput(action, first, sources)
  validateOutput(action, peer, sources)
  validateOutput(action, repeated, sources)
  if not outputsIndependent(first, peer) or not outputsIndependent(first, repeated) or
      not outputsIndependent(peer, repeated):
    raise newException(ValueError, "Cumulative-extrema peer/repeat outputs alias another allocation")
  if not sameValues(first.values, peer.values) or not sameValues(first.values, repeated.values) or
      not sameIndices(first.indices, peer.indices) or not sameIndices(first.indices, repeated.indices):
    raise newException(ValueError, "Cumulative-extrema peer/repeat outputs differ exactly")
  %*{
    "input_dtype": dtypeName[T](),
    "values_dtype": dtypeName[T](),
    "indices_dtype": "int64",
    "values_shape": shapeJson(first.values),
    "indices_shape": shapeJson(first.indices),
    "values": valuesJson(first.values),
    "indices": indicesJson(first.indices),
    "values_bits": valuesBitsJson(first.values),
    "values_data_ptr": $cast[uint](first.values.get_offset_ptr),
    "indices_data_ptr": $cast[uint](first.indices.get_offset_ptr),
    "values_contiguous": first.values.is_C_contiguous,
    "indices_contiguous": first.indices.is_C_contiguous,
    "values_disjoint_from_inputs": true,
    "indices_disjoint_from_inputs": true,
    "values_indices_independent": true,
    "peer_independent": true,
    "repeat_independent": true,
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

proc progressJson(progress: ActionProgress): JsonNode =
  %*{
    "preflight_calls_attempted": progress.preflightCallsAttempted,
    "preflight_calls_completed": progress.preflightCallsCompleted,
    "warmup_calls_attempted": progress.warmupCallsAttempted,
    "warmup_calls_completed": progress.warmupCallsCompleted,
    "sample_calls_attempted": progress.sampleCallsAttempted,
    "sample_calls_completed": progress.sampleCallsCompleted,
    "samples_completed": progress.samplesCompleted,
  }


proc bindAction[T](name: string; sources: ResidentInputs[T]): BoundAction[T] =
  let separator = name.find('_')
  if separator <= 0 or separator == name.high:
    raise newException(ValueError, "Malformed cumulative-extrema action: " & name)
  let operationName = name[0 ..< separator]
  let profile = name[separator + 1 .. ^1]
  if profile notin profiles:
    raise newException(ValueError, "Unknown cumulative-extrema profile: " & profile)
  case operationName
  of "cummin": result.operation = opCummin
  of "cummax": result.operation = opCummax
  else: raise newException(ValueError, "Unknown cumulative-extrema operation: " & operationName)
  result.profile = profile
  result.axis = expectedAxis(profile)
  result.input = tensorAt(sources, profile)

proc metadata(shape: openArray[int]): Metadata =
  result.len = shape.len
  for index, dimension in shape:
    result[index] = dimension

proc loadTensor[T](values: JsonNode; shape: openArray[int]; label, sourceProfile: string): Tensor[T] =
  var count = 1
  for dimension in shape:
    count *= dimension
  if values.kind != JArray or values.len != count:
    raise newException(ValueError, label & " must contain exactly " & $count & " values")
  result = newTensorUninit[T](metadata(shape))
  if count == 0:
    return
  let data = cast[ptr UncheckedArray[T]](result.get_data_ptr)
  for index in 0 ..< count:
    let value = parseValue[T](values[index], label)
    if not sameValue(value, sourceValue[T](sourceProfile, index)):
      raise newException(ValueError, label & " values do not match the exact deterministic source")
    data[index] = value


proc scalarTensor[T](node: JsonNode): Tensor[T] =
  let value = parseValue[T](node, "scalar_value")
  if not sameValue(value, scalarValue[T]()):
    raise newException(ValueError, "Scalar value does not match the exact deterministic source")
  newTensorWith[T](default(Metadata), value)

proc validateCommonManifest[T](manifest: JsonNode): seq[string] =
  if manifest.kind != JObject or manifest.len != manifestFields.len:
    raise newException(ValueError, "Cumulative-extrema manifest has missing or unknown fields")
  for field in manifestFields:
    if not manifest.hasKey(field):
      raise newException(ValueError, "Cumulative-extrema manifest has missing or unknown fields")
  let dtype = dtypeName[T]()
  if manifest["input_dtype"].kind != JString or manifest["input_dtype"].getStr != dtype or
      manifest["values_dtype"].kind != JString or manifest["values_dtype"].getStr != dtype or
      manifest["indices_dtype"].kind != JString or manifest["indices_dtype"].getStr != "int64":
    raise newException(ValueError, "Cumulative-extrema dtype fields do not match dispatched types")
  if manifest["rows"].kind != JInt or manifest["rows"].getInt != rows or
      manifest["columns"].kind != JInt or manifest["columns"].getInt != columns or
      manifest["shape"] != %*[rows, columns] or manifest["cube_shape"] != %*[16, 32, 32] or
      manifest["vector_shape"] != %*[elementCount] or
      manifest["offset_backing_shape"] != %*[rows, columns + 2] or
      manifest["scalar_shape"] != newJArray() or manifest["empty_shape"] != %*[2, 0, 3]:
    raise newException(ValueError, "Cumulative-extrema input shape metadata mismatch")
  for (field, expected) in [("warmup", warmupCount), ("repeats", repeatCount),
                            ("iterations", callsPerSample)]:
    if manifest[field].kind != JInt or manifest[field].getInt != expected:
      raise newException(ValueError, "Cumulative-extrema timing protocol mismatch")
  let suppliedProfiles = manifest["profiles"]
  if suppliedProfiles.kind != JArray or suppliedProfiles.len != profiles.len:
    raise newException(ValueError, "Cumulative-extrema profile manifest must contain fourteen profiles")
  for index, profile in profiles:
    if suppliedProfiles[index].kind != JString or suppliedProfiles[index].getStr != profile:
      raise newException(ValueError, "Cumulative-extrema ordered profile manifest mismatch")
  validateLayoutManifest(manifest)
  result = actionsFor(dtype)
  let suppliedActions = manifest["actions"]
  if suppliedActions.kind != JArray or suppliedActions.len != result.len:
    raise newException(ValueError, "Cumulative-extrema action count mismatch for input dtype")
  for index, action in result:
    if suppliedActions[index].kind != JString or suppliedActions[index].getStr != action:
      raise newException(ValueError, "Cumulative-extrema ordered action manifest mismatch")

proc validateOwners[T](sources: ResidentInputs[T]; dtype: string) =
  if not sources.base.is_C_contiguous or not sources.rowsHead.is_C_contiguous or
      not sources.rowsTail.is_C_contiguous or not sources.offsetBacking.is_C_contiguous:
    raise newException(ValueError, "Cumulative-extrema resident source matrices must be contiguous")
  let basePointer = cast[uint](sources.base.get_data_ptr)
  if cast[uint](sources.cube.get_data_ptr) != basePointer or
      cast[uint](sources.vector.get_data_ptr) != basePointer or
      cast[uint](sources.transpose.get_data_ptr) != basePointer or
      cast[uint](sources.reverse.get_data_ptr) != basePointer or
      cast[uint](sources.broadcast.get_data_ptr) != basePointer or
      cast[uint](sources.offset.get_data_ptr) != cast[uint](sources.offsetBacking.get_data_ptr):
    raise newException(ValueError, "Cumulative-extrema resident views do not share their specified backing storage")
  if sources.offset.offset != 1 or sources.reverse.offset != elementCount - 1 or
      sources.broadcast.strides[0] != 0 or sources.transpose.strides[0] != 1:
    raise newException(ValueError, "Cumulative-extrema offset, reverse, transpose, or broadcast layout mismatch")
  let rootPointers = [
    cast[uint](sources.base.get_data_ptr), cast[uint](sources.rowsHead.get_data_ptr),
    cast[uint](sources.rowsTail.get_data_ptr), cast[uint](sources.offsetBacking.get_data_ptr),
    cast[uint](sources.scalar.get_data_ptr),
  ]
  for left in 0 ..< rootPointers.len:
    for right in left + 1 ..< rootPointers.len:
      if rootPointers[left] == rootPointers[right]:
        raise newException(ValueError, "Cumulative-extrema resident input roots must own distinct storage")
  if dtype in nonemptyDtypes:
    for profile in nonemptyProfiles:
      validateTensorLayout(sources, profile)
      validateProfileValues(sources, profile)
  validateTensorLayout(sources, "scalar")
  validateTensorLayout(sources, "empty")
  validateProfileValues(sources, "scalar")
  validateProfileValues(sources, "empty")

proc prepareInputs[T](manifest: JsonNode; dtype: string): ResidentInputs[T] =
  when T is Complex:
    for field in ["values", "rows_head_values", "rows_tail_values", "offset_backing_values"]:
      if manifest[field].kind != JArray or manifest[field].len != 0:
        raise newException(ValueError, "Complex benchmark inputs must not contain nonempty profile payloads")
    result.scalar = scalarTensor[T](manifest["scalar_value"])
    result.empty = newTensorUninit[T](emptyShape[0], emptyShape[1], emptyShape[2])
    if result.empty.size != 0:
      raise newException(ValueError, "Complex empty input must have no values")
    validateTensorLayout(result, "scalar")
    validateTensorLayout(result, "empty")
    validateProfileValues(result, "scalar")
    validateProfileValues(result, "empty")
  else:
    result.base = loadTensor[T](manifest["values"], [rows, columns], "values", "rows")
    result.rowsHead = loadTensor[T](manifest["rows_head_values"], [rows, columns], "rows_head_values", "rows_head")
    result.rowsTail = loadTensor[T](manifest["rows_tail_values"], [rows, columns], "rows_tail_values", "rows_tail")
    result.offsetBacking = loadTensor[T](manifest["offset_backing_values"], offsetShape,
                                         "offset_backing_values", "vector")
    result.cube = result.base.reshape(cubeShape)
    result.vector = result.base.reshape(elementCount)
    result.transpose = result.base.transpose2d()
    result.offset = result.offsetBacking[_, 1..columns]
    result.reverse = result.base[^1..0|-1, ^1..0|-1]
    result.broadcast = result.base[0..0, _].broadcast(rows, columns)
    result.scalar = scalarTensor[T](manifest["scalar_value"])
    result.empty = newTensorUninit[T](emptyShape[0], emptyShape[1], emptyShape[2])
    if result.empty.size != 0:
      raise newException(ValueError, "Empty input must have no values")
    validateOwners(result, dtype)

proc reportForDtype[T](manifest: JsonNode; preflightOnly: bool): JsonNode =
  let dtype = dtypeName[T]()
  let actions = validateCommonManifest[T](manifest)
  let sources = prepareInputs[T](manifest, dtype)
  var actionReports = newJObject()
  var total = ActionProgress()
  var actionsCompleted = 0
  for name in actions:
    var progress = ActionProgress()
    try:
      let action = bindAction(name, sources)
      progress.preflightCallsAttempted += 1
      let first = execute(action)
      progress.preflightCallsCompleted += 1
      progress.preflightCallsAttempted += 1
      let peer = execute(action)
      progress.preflightCallsCompleted += 1
      progress.preflightCallsAttempted += 1
      let repeated = execute(action)
      progress.preflightCallsCompleted += 1
      let observation = observe(action, first, peer, repeated, sources)
      if preflightOnly:
        actionReports[name] = %*{
          "observation": observation,
          "progress": progressJson(progress),
        }
      else:
        var samples = newSeq[float64](repeatCount)
        for _ in 0 ..< warmupCount:
          inc progress.warmupCallsAttempted
          consume(execute(action))
          inc progress.warmupCallsCompleted
        for sample in 0 ..< repeatCount:
          let start = getMonoTime()
          for _ in 0 ..< callsPerSample:
            inc progress.sampleCallsAttempted
            consume(execute(action))
            inc progress.sampleCallsCompleted
          samples[sample] = float64((getMonoTime() - start).inNanoseconds) / 1e6 /
            float64(callsPerSample)
          inc progress.samplesCompleted
        let ordered = sorted(samples)
        let middle = samples.len div 2
        actionReports[name] = %*{
          "observation": observation,
          "samples_ms": samples,
          "median_ms": (ordered[middle - 1] + ordered[middle]) / 2.0,
          "p95_ms": ordered[int(ceil(0.95 * float64(samples.len))) - 1],
          "progress": progressJson(progress),
        }
      inc actionsCompleted
    except CatchableError as error:
      actionReports[name] = %*{
        "failure": error.msg,
        "progress": progressJson(progress),
      }
    total.preflightCallsAttempted += progress.preflightCallsAttempted
    total.preflightCallsCompleted += progress.preflightCallsCompleted
    total.warmupCallsAttempted += progress.warmupCallsAttempted
    total.warmupCallsCompleted += progress.warmupCallsCompleted
    total.sampleCallsAttempted += progress.sampleCallsAttempted
    total.sampleCallsCompleted += progress.sampleCallsCompleted
    total.samplesCompleted += progress.samplesCompleted
  # Recheck resident storage after the last measured call, outside every clock.
  when T is Complex:
    validateProfileValues(sources, "scalar")
  else:
    validateOwners(sources, dtype)
    let backing = cast[ptr UncheckedArray[T]](sources.offsetBacking.get_data_ptr)
    for index in 0 ..< sources.offsetBacking.size:
      if not sameValue(backing[index], baseValue[T](index)):
        raise newException(ValueError, "Offset backing changed during cumulative-extrema execution")
  %*{
    "actions": actionReports,
    "progress": {
      "actions_attempted": actions.len,
      "actions_completed": actionsCompleted,
      "preflight_calls_attempted": total.preflightCallsAttempted,
      "preflight_calls_completed": total.preflightCallsCompleted,
      "warmup_calls_attempted": total.warmupCallsAttempted,
      "warmup_calls_completed": total.warmupCallsCompleted,
      "sample_calls_attempted": total.sampleCallsAttempted,
      "sample_calls_completed": total.sampleCallsCompleted,
      "samples_completed": total.samplesCompleted,
    },
  }

proc runForDtype(manifest: JsonNode; preflightOnly: bool): JsonNode =
  if manifest.kind != JObject or not manifest.hasKey("input_dtype") or
      manifest["input_dtype"].kind != JString:
    raise newException(ValueError, "Cumulative-extrema manifest must declare a string input dtype")
  case manifest["input_dtype"].getStr
  of "bool": reportForDtype[bool](manifest, preflightOnly)
  of "uint8": reportForDtype[uint8](manifest, preflightOnly)
  of "int8": reportForDtype[int8](manifest, preflightOnly)
  of "int16": reportForDtype[int16](manifest, preflightOnly)
  of "int32": reportForDtype[int32](manifest, preflightOnly)
  of "int64": reportForDtype[int64](manifest, preflightOnly)
  of "float32": reportForDtype[float32](manifest, preflightOnly)
  of "float64": reportForDtype[float64](manifest, preflightOnly)
  of "uint16": reportForDtype[uint16](manifest, preflightOnly)
  of "uint32": reportForDtype[uint32](manifest, preflightOnly)
  of "uint64": reportForDtype[uint64](manifest, preflightOnly)
  of "complex64": reportForDtype[Complex[float32]](manifest, preflightOnly)
  of "complex128": reportForDtype[Complex[float64]](manifest, preflightOnly)
  else: raise newException(ValueError, "Unsupported cumulative-extrema benchmark dtype")

when isMainModule:
  if paramCount() < 1 or paramCount() > 2:
    echo "Usage: cumulative_extrema_runner INPUT.json [--preflight]"
    quit(if paramCount() == 0: 0 else: 2)
  let preflightOnly = paramCount() == 2 and paramStr(2) == "--preflight"
  if paramCount() == 2 and not preflightOnly:
    echo "Usage: cumulative_extrema_runner INPUT.json [--preflight]"
    quit(2)
  let manifest = parseFile(paramStr(1))
  echo runForDtype(manifest, preflightOnly)
