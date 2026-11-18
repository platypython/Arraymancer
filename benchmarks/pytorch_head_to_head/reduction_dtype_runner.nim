# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating CPU reductions and cumulative scans with explicit float32/float64 output dtypes.
import std/[algorithm, json, math, monotimes, os, times]
import arraymancer

const cases = [
  "mean", "mean_rows", "mean_columns", "sum", "sum_rows", "sum_columns",
  "prod", "prod_rows", "prod_columns",
  "cumsum_rows", "cumsum_columns", "cumprod_rows", "cumprod_columns",
]

proc nativeValue[T](word: int64): T =
  ## Match Torch's signed-word cast, including unsigned wraparound.
  when T is bool:
    word != 0
  else:
    cast[T](word)

proc loadInput[T](manifest: JsonNode): Tensor[T] =
  let rows = manifest["rows"].getInt
  let columns = manifest["columns"].getInt
  result = newTensorUninit[T](rows, columns)
  if manifest.hasKey("words"):
    let words = manifest["words"]
    if words.len != result.size:
      raise newException(ValueError, "Word count does not match input shape")
    let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
    for index in 0 ..< result.size:
      data[index] = nativeValue[T](words[index].getBiggestInt)
  else:
    let values = manifest["values"]
    if values.len != result.size:
      raise newException(ValueError, "Value count does not match input shape")
    let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
    for index in 0 ..< result.size:
      when T is bool:
        data[index] = values[index].getFloat != 0.0
      else:
        data[index] = T(values[index].getFloat)

proc execute[InputT, OutputT](name: string; input: Tensor[InputT]): Tensor[OutputT] =
  case name
  of "mean": newTensorWith[OutputT]([], input.mean(dtype = OutputT))
  of "mean_rows": input.mean(-1, dtype = OutputT)
  of "mean_columns": input.mean(-2, dtype = OutputT)
  of "sum": newTensorWith[OutputT]([], input.sum(dtype = OutputT))
  of "sum_rows": input.sum(-1, dtype = OutputT)
  of "sum_columns": input.sum(-2, dtype = OutputT)
  of "prod": newTensorWith[OutputT]([], input.product(dtype = OutputT))
  of "prod_rows": input.product(-1, dtype = OutputT)
  of "prod_columns": input.product(-2, dtype = OutputT)
  of "cumsum_rows": input.cumsum(-1, dtype = OutputT)
  of "cumsum_columns": input.cumsum(-2, dtype = OutputT)
  of "cumprod_rows": input.cumprod(-1, dtype = OutputT)
  of "cumprod_columns": input.cumprod(-2, dtype = OutputT)
  else: raise newException(ValueError, "Unknown reduction dtype action: " & name)

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
  else: {.error: "Unsupported reduction dtype".}
proc jsonFloat(value: float64): JsonNode =
  ## JSON has no non-finite numeric literals; use explicit tokens for exact values.
  if value.isNaN: newJString("NaN")
  elif value == Inf: newJString("+Inf")
  elif value == NegInf: newJString("-Inf")
  else: %value

proc observe[T](input: Tensor[T]; inputDtype: string): JsonNode =
  let dtype = dtypeName[T]()
  var shape = newJArray()
  for dimension in input.shape:
    shape.add %dimension
  var values = newJArray()
  for value in input:
    values.add jsonFloat(float64(value))
  %*{"input_dtype": inputDtype, "dtype": dtype, "shape": shape, "values": values}

proc consume[T](input: Tensor[T]) =
  ## Keep the allocating tensor result observable to the optimizer.
  let address = cast[uint](input.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`address`)
    : "memory"
  """

proc run[InputT, OutputT](manifest: JsonNode): JsonNode =
  let input = loadInput[InputT](manifest)
  let iterations = manifest["iterations"].getInt
  let repeats = manifest["repeats"].getInt
  let warmup = manifest["warmup"].getInt
  var report = newJObject()
  for name in cases:
    let observation = observe(execute[InputT, OutputT](name, input), dtypeName[InputT]())
    for _ in 0 ..< warmup:
      consume(execute[InputT, OutputT](name, input))
    var samples = newSeq[float64](repeats)
    for sample in 0 ..< repeats:
      let start = getMonoTime()
      for _ in 0 ..< iterations:
        consume(execute[InputT, OutputT](name, input))
      samples[sample] = float64((getMonoTime() - start).inNanoseconds) /
                        1e6 / float64(iterations)
    var ordered = samples
    ordered.sort()
    let middle = repeats div 2
    let median = if repeats mod 2 == 0:
                   (ordered[middle - 1] + ordered[middle]) / 2.0
                 else:
                   ordered[middle]
    report[name] = %*{"observation": observation, "samples_ms": samples,
      "median_ms": median,
      "p95_ms": ordered[int(ceil(0.95 * float64(repeats))) - 1]}
  report

proc runForInput[InputT](manifest: JsonNode): JsonNode =
  case manifest["output_dtype"].getStr
  of "float32": run[InputT, float32](manifest)
  of "float64": run[InputT, float64](manifest)
  else: raise newException(ValueError, "Unsupported reduction output dtype")

proc runForDtype(manifest: JsonNode): JsonNode =
  case manifest["dtype"].getStr
  of "bool": runForInput[bool](manifest)
  of "int8": runForInput[int8](manifest)
  of "int16": runForInput[int16](manifest)
  of "int32": runForInput[int32](manifest)
  of "int64": runForInput[int64](manifest)
  of "uint8": runForInput[uint8](manifest)
  of "uint16": runForInput[uint16](manifest)
  of "uint32": runForInput[uint32](manifest)
  of "uint64": runForInput[uint64](manifest)
  of "float32": runForInput[float32](manifest)
  of "float64": runForInput[float64](manifest)
  else: raise newException(ValueError, "Unsupported reduction input dtype")

when isMainModule:
  if paramCount() != 1:
    echo "Usage: reduction_dtype_runner INPUT.json"
    quit(if paramCount() == 0: 0 else: 2)
  let manifest = parseFile(paramStr(1))
  if manifest["rows"].getInt != 128 or manifest["columns"].getInt != 128:
    raise newException(ValueError, "Reduction dtype benchmark input must be 128x128")
  echo runForDtype(manifest)
