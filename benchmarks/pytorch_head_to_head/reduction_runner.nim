# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating CPU/Metal reductions and scans on externally supplied floating-point data.
import std/[algorithm, json, math, monotimes, os, strutils, times]
import arraymancer

const cases = ["sum", "mean", "prod", "sum_rows", "sum_columns", "mean_rows", "mean_columns",
               "prod_rows", "prod_columns", "mean_rows_backward", "mean_columns_backward",
               "sum_backward", "mean_backward", "prod_backward", "prod_rows_backward", "prod_columns_backward",
               "cumsum_rows", "cumsum_columns", "cumsum_rows_backward", "cumsum_columns_backward",
               "cumprod_rows", "cumprod_columns", "cumprod_rows_backward", "cumprod_columns_backward"]

when defined(reductionFloat64) and defined(metal):
  {.error: "reductionFloat64 is supported only by the CPU reduction runner".}

when defined(reductionFloat64):
  type ReductionScalar = float64
else:
  type ReductionScalar = float32

when defined(metal):
  type ReductionTensor = MetalTensor[ReductionScalar]
else:
  type ReductionTensor = Tensor[ReductionScalar]

template synchronize() =
  when defined(metal): metalSynchronize()

proc execute(name: string; input: ReductionTensor): ReductionTensor =
  if name.endsWith("_backward"):
    when defined(metal):
      let context = newContext(MetalTensor[ReductionScalar])
    else:
      let context = newContext(Tensor[ReductionScalar])
    let variable = context.variable(input, requires_grad = true)
    let axis = if "_rows_" in name: -1 else: -2
    let output =
      if name == "prod_backward": variable.product
      elif name == "sum_backward": variable.sum
      elif name == "mean_backward": variable.mean
      elif name.startsWith("mean_"): variable.mean(axis)
      elif name.startsWith("cumsum_"): variable.cumsum(axis)
      elif name.startsWith("cumprod_"): variable.cumprod(axis)
      else: variable.product(axis)
    output.backprop(output.value.ones_like)
    return variable.grad
  case name
  of "prod":
    when defined(metal): input.prod
    else: newTensorWith[ReductionScalar]([], input.product)
  of "sum":
    when defined(metal): input.sum
    else: newTensorWith[ReductionScalar]([], input.sum)
  of "mean":
    when defined(metal): input.mean
    else: newTensorWith[ReductionScalar]([], input.mean)
  of "sum_rows": input.sum(-1)
  of "sum_columns": input.sum(-2)
  of "mean_rows": input.mean(-1)
  of "mean_columns": input.mean(-2)
  of "prod_rows":
    when defined(metal): input.prod(-1)
    else: input.product(-1)
  of "prod_columns":
    when defined(metal): input.prod(-2)
    else: input.product(-2)
  of "cumsum_rows": input.cumsum(-1)
  of "cumsum_columns": input.cumsum(-2)
  of "cumprod_rows": input.cumprod(-1)
  of "cumprod_columns": input.cumprod(-2)
  else: raise newException(ValueError, "Unknown reduction: " & name)

proc observe(input: ReductionTensor): JsonNode =
  when defined(metal):
    let host = input.cpu
  else:
    let host = input
  var shape = newJArray()
  for dimension in host.shape: shape.add %dimension
  var values = newJArray()
  for value in host: values.add %value
  when defined(reductionFloat64):
    %*{"shape": shape, "dtype": "float64", "values": values}
  else:
    %*{"shape": shape, "dtype": "float32", "values": values}

proc consume(input: ReductionTensor) =
  when defined(metal):
    let address = cast[uint](input.storage.owner.handle)
  else:
    let address = cast[uint](input.get_offset_ptr)
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
  var source = newTensorUninit[ReductionScalar](manifest["rows"].getInt, manifest["columns"].getInt)
  let data = cast[ptr UncheckedArray[ReductionScalar]](source.get_offset_ptr)
  for index in 0 ..< source.size: data[index] = ReductionScalar(manifest["values"][index].getFloat)
  when defined(metal):
    let input = source.metal
  else:
    let input = source
  let iterations = manifest["iterations"].getInt
  let repeats = manifest["repeats"].getInt
  var report = newJObject()
  for name in cases:
    let observation = observe(execute(name, input))
    for index in 0 ..< manifest["warmup"].getInt: consume(execute(name, input))
    synchronize()
    var samples = newSeq[float64](repeats)
    for sample in 0 ..< repeats:
      synchronize()
      let start = getMonoTime()
      for index in 0 ..< iterations: consume(execute(name, input))
      synchronize()
      samples[sample] = float64((getMonoTime() - start).inNanoseconds) / 1e6 / float64(iterations)
    var ordered = samples
    ordered.sort()
    report[name] = %*{"observation": observation, "samples_ms": samples,
      "median_ms": ordered[repeats div 2],
      "p95_ms": ordered[int(ceil(0.95 * float64(repeats))) - 1]}
  echo report
