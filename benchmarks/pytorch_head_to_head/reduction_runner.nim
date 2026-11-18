# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating native Metal reductions using externally supplied float32 data.
import std/[algorithm, json, math, monotimes, os, times]
import arraymancer

const cases = ["prod", "sum_rows", "sum_columns", "mean_rows", "mean_columns",
               "prod_rows", "prod_columns", "mean_rows_backward", "mean_columns_backward"]

proc execute(name: string; input: MetalTensor[float32]): MetalTensor[float32] =
  if name in ["mean_rows_backward", "mean_columns_backward"]:
    let context = newContext(MetalTensor[float32])
    let variable = context.variable(input, requires_grad = true)
    let output = variable.mean(if name == "mean_rows_backward": -1 else: -2)
    output.backprop(output.value.ones_like)
    return variable.grad
  case name
  of "prod": input.prod
  of "sum_rows": input.sum(-1)
  of "sum_columns": input.sum(-2)
  of "mean_rows": input.mean(-1)
  of "mean_columns": input.mean(-2)
  of "prod_rows": input.prod(-1)
  of "prod_columns": input.prod(-2)
  else: raise newException(ValueError, "Unknown reduction: " & name)

proc observe(input: MetalTensor[float32]): JsonNode =
  let host = input.cpu
  var shape = newJArray()
  for dimension in host.shape: shape.add %dimension
  var values = newJArray()
  for value in host: values.add %value
  %*{"shape": shape, "values": values}

proc consume(input: MetalTensor[float32]) =
  let address = cast[uint](input.storage.owner.handle)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`address`)
    : "memory"
  """

when isMainModule:
  if paramCount() != 1:
    echo "Usage: reduction_runner INPUT.json"
    quit(if paramCount() == 0: 0 else: 2)
  let manifest = parseFile(paramStr(1))
  var source = newTensorUninit[float32](manifest["rows"].getInt, manifest["columns"].getInt)
  let data = cast[ptr UncheckedArray[float32]](source.get_offset_ptr)
  for index in 0 ..< source.size: data[index] = float32(manifest["values"][index].getFloat)
  let input = source.metal
  let iterations = manifest["iterations"].getInt
  let repeats = manifest["repeats"].getInt
  var report = newJObject()
  for name in cases:
    let observation = observe(execute(name, input))
    for index in 0 ..< manifest["warmup"].getInt: consume(execute(name, input))
    metalSynchronize()
    var samples = newSeq[float64](repeats)
    for sample in 0 ..< repeats:
      metalSynchronize()
      let start = getMonoTime()
      for index in 0 ..< iterations: consume(execute(name, input))
      metalSynchronize()
      samples[sample] = float64((getMonoTime() - start).inNanoseconds) / 1e6 / float64(iterations)
    var ordered = samples
    ordered.sort()
    report[name] = %*{"observation": observation, "samples_ms": samples,
      "median_ms": ordered[repeats div 2],
      "p95_ms": ordered[int(ceil(0.95 * float64(repeats))) - 1]}
  echo report
