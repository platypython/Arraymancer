# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating CPU extrema forward and explicitly seeded backward actions.
import std/[algorithm, json, math, monotimes, os, strutils, times]
import arraymancer

const actions = [
  "amin_flat_whole", "amin_flat_rows", "amin_flat_columns", "amin_flat_axes_0_2", "amin_flat_axes_2_1",
  "amin_keepdim_whole", "amin_keepdim_rows", "amin_keepdim_columns", "amin_keepdim_axes_0_2", "amin_keepdim_axes_2_1",
  "amax_flat_whole", "amax_flat_rows", "amax_flat_columns", "amax_flat_axes_0_2", "amax_flat_axes_2_1",
  "amax_keepdim_whole", "amax_keepdim_rows", "amax_keepdim_columns", "amax_keepdim_axes_0_2", "amax_keepdim_axes_2_1",
]

proc shapeOf(node: JsonNode): Metadata =
  if node.kind != JArray:
    raise newException(ValueError, "Tensor shape must be an array")
  result.len = node.len
  for index in 0 ..< node.len:
    if node[index].kind != JInt or node[index].getInt <= 0:
      raise newException(ValueError, "Tensor dimensions must be positive integers")
    result[index] = node[index].getInt

proc loadTensor[T](shapeNode, values: JsonNode): Tensor[T] =
  let shape = shapeOf(shapeNode)
  result = newTensorUninit[T](shape)
  if values.kind != JArray or values.len != result.size:
    raise newException(ValueError, "Tensor values do not match the declared shape")
  let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
  for index in 0 ..< result.size:
    if values[index].kind notin {JInt, JFloat}:
      raise newException(ValueError, "Tensor values must be JSON numbers")
    data[index] = T(values[index].getFloat)

proc runAction[T: SomeFloat](op, caseName: string; keepdim: bool; input, seed: Tensor[T]): Tensor[T] =
  ## Each call owns a fresh graph while immutable inputs and seeds stay shared.
  let context = newContext(Tensor[T])
  let variable = context.variable(input, requires_grad = true)
  let output =
    case caseName
    of "whole":
      if op == "amin": variable.amin(keepdim = keepdim) else: variable.amax(keepdim = keepdim)
    of "rows":
      if op == "amin": variable.amin(1, keepdim = keepdim) else: variable.amax(1, keepdim = keepdim)
    of "columns":
      if op == "amin": variable.amin(0, keepdim = keepdim) else: variable.amax(0, keepdim = keepdim)
    of "axes_0_2":
      if op == "amin": variable.amin([0, 2], keepdim = keepdim) else: variable.amax([0, 2], keepdim = keepdim)
    of "axes_2_1":
      if op == "amin": variable.amin([2, 1], keepdim = keepdim) else: variable.amax([2, 1], keepdim = keepdim)
    else: raise newException(ValueError, "Unknown extrema gradient action: " & caseName)
  output.backprop(seed)
  variable.grad

proc negativeZero[T: SomeFloat](value: T): bool =
  value == T(0) and (when T is float32: (cast[uint32](value) and 0x80000000'u32) != 0
                     else: (cast[uint64](value) and 0x8000000000000000'u64) != 0)

proc observe[T: SomeFloat](value: Tensor[T]; dtype: string): JsonNode =
  var shape = newJArray()
  for dimension in value.shape: shape.add %dimension
  var values = newJArray()
  var negativeZeros = newJArray()
  for index in 0 ..< value.size:
    let item = value.atContiguousIndex(index)
    if item.classify in {fcNan, fcInf, fcNegInf}:
      values.add(if item.isNaN: %"nan" elif item > T(0): %"+inf" else: %"-inf")
    else:
      values.add %float64(item)
    if negativeZero(item): negativeZeros.add %index
  %*{"dtype": dtype, "shape": shape, "values": values, "negative_zero_indices": negativeZeros}

proc consume[T](value: Tensor[T]) =
  let address = cast[uint](value.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`address`)
    : "memory"
  """

proc run[T: SomeFloat](manifest: JsonNode; dtype: string): JsonNode =
  const required = ["dtype", "shape", "rows", "columns", "values", "cube_shape", "cube_values",
                    "actions", "seeds", "warmup", "repeats", "iterations"]
  if manifest.kind != JObject or manifest.len != required.len:
    raise newException(ValueError, "Extrema gradient manifest has missing or unknown fields")
  for key in required:
    if not manifest.hasKey(key): raise newException(ValueError, "Extrema gradient manifest has missing or unknown fields")
  if manifest["dtype"].getStr != dtype or manifest["rows"].getInt != 128 or manifest["columns"].getInt != 128 or
      manifest["shape"] != %*[128, 128] or manifest["cube_shape"] != %*[16, 32, 32]:
    raise newException(ValueError, "Extrema gradient input metadata mismatch")
  for (key, expected) in [("warmup", 50), ("repeats", 100), ("iterations", 50)]:
    if manifest[key].kind != JInt or manifest[key].getInt != expected:
      raise newException(ValueError, "Extrema gradient timing protocol mismatch")
  let supplied = manifest["actions"]
  if supplied.kind != JArray or supplied.len != actions.len:
    raise newException(ValueError, "Extrema gradient action manifest must contain exactly 20 actions")
  for index, action in actions:
    if supplied[index].kind != JString or supplied[index].getStr != action:
      raise newException(ValueError, "Extrema gradient ordered action manifest mismatch")
  let matrix = loadTensor[T](manifest["shape"], manifest["values"])
  if manifest["cube_values"] != manifest["values"]:
    raise newException(ValueError, "Extrema gradient cube must share the declared matrix values")
  let cube = matrix.reshape(16, 32, 32)
  let seeds = manifest["seeds"]
  if seeds.kind != JObject or seeds.len != actions.len:
    raise newException(ValueError, "Extrema gradient seed manifest must contain exactly 20 seeds")
  var seedTensors = newSeq[Tensor[T]](actions.len)
  for index, name in actions:
    if not seeds.hasKey(name) or seeds[name].kind != JObject or seeds[name].len != 2 or
        not seeds[name].hasKey("shape") or not seeds[name].hasKey("values"):
      raise newException(ValueError, "Malformed seed record for " & name)
    seedTensors[index] = loadTensor[T](seeds[name]["shape"], seeds[name]["values"])
  var report = newJObject()
  for index, name in actions:
    let pieces = name.split('_')
    let op = pieces[0]
    let keepdim = pieces[1] == "keepdim"
    let caseName = pieces[2 .. ^1].join("_")
    let input = if caseName.startsWith("axes_"): cube else: matrix
    let observation = observe(runAction(op, caseName, keepdim, input, seedTensors[index]), dtype)
    for _ in 0 ..< 50: consume(runAction(op, caseName, keepdim, input, seedTensors[index]))
    var samples = newSeq[float64](100)
    for sample in 0 ..< 100:
      let start = getMonoTime()
      for _ in 0 ..< 50: consume(runAction(op, caseName, keepdim, input, seedTensors[index]))
      samples[sample] = float64((getMonoTime() - start).inNanoseconds) / 1e6 / 50.0
    var ordered = samples
    ordered.sort()
    report[name] = %*{"observation": observation, "samples_ms": samples,
      "median_ms": (ordered[49] + ordered[50]) / 2.0,
      "p95_ms": ordered[int(ceil(0.95 * float64(samples.len))) - 1]}
  report

when isMainModule:
  if paramCount() != 1:
    echo "Usage: extrema_gradient_runner INPUT.json"
    quit(if paramCount() == 0: 0 else: 2)
  let manifest = parseFile(paramStr(1))
  case manifest["dtype"].getStr
  of "float32": echo run[float32](manifest, "float32")
  of "float64": echo run[float64](manifest, "float64")
  else: raise newException(ValueError, "Unsupported extrema gradient dtype")
