# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures allocating CPU integer reductions and scans on signed int64 words.
import std/[algorithm, json, math, monotimes, os, times]
import arraymancer

const cases = ["sum", "prod", "sum_rows", "sum_columns", "prod_rows", "prod_columns",
               "cumsum_rows", "cumsum_columns", "cumprod_rows", "cumprod_columns"]

proc nativeValue[T](word: int64): T =
  ## Match Torch's signed-word cast, including unsigned wraparound.
  when T is bool:
    word != 0
  else:
    cast[T](word)

proc loadInput[T](manifest: JsonNode): Tensor[T] =
  let rows = manifest["rows"].getInt
  let columns = manifest["columns"].getInt
  let words = manifest["words"]
  result = newTensorUninit[T](rows, columns)
  if words.len != result.size:
    raise newException(ValueError, "Word count does not match input shape")
  let data = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
  for index in 0 ..< result.size:
    data[index] = nativeValue[T](words[index].getBiggestInt)

proc execute[T](name: string; input: Tensor[T]): Tensor[int64] =
  case name
  of "sum": newTensorWith[int64]([], input.sum())
  of "prod": newTensorWith[int64]([], input.product())
  of "sum_rows": input.sum(-1)
  of "sum_columns": input.sum(-2)
  of "prod_rows": input.product(-1)
  of "prod_columns": input.product(-2)
  of "cumsum_rows": input.cumsum(-1)
  of "cumsum_columns": input.cumsum(-2)
  of "cumprod_rows": input.cumprod(-1)
  of "cumprod_columns": input.cumprod(-2)
  else: raise newException(ValueError, "Unknown integer action: " & name)

proc observe(input: Tensor[int64]): JsonNode =
  var shape = newJArray()
  for dimension in input.shape:
    shape.add %dimension
  var values = newJArray()
  for value in input:
    values.add %value
  %*{"dtype": "int64", "shape": shape, "values": values}

proc consume(input: Tensor[int64]) =
  ## Keep allocating tensor work observable to the optimizer.
  let address = cast[uint](input.get_offset_ptr)
  asm {.asmSyntax: "gcc".}"""
    ""
    :
    : "r"(`address`)
    : "memory"
  """

proc run[T](manifest: JsonNode): JsonNode =
  let input = loadInput[T](manifest)
  let iterations = manifest["iterations"].getInt
  let repeats = manifest["repeats"].getInt
  let warmup = manifest["warmup"].getInt
  var report = newJObject()
  for name in cases:
    let observation = observe(execute(name, input))
    for _ in 0 ..< warmup:
      consume(execute(name, input))
    var samples = newSeq[float64](repeats)
    for sample in 0 ..< repeats:
      let start = getMonoTime()
      for _ in 0 ..< iterations:
        consume(execute(name, input))
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

proc runForDtype(manifest: JsonNode): JsonNode =
  case manifest["dtype"].getStr
  of "bool": run[bool](manifest)
  of "int8": run[int8](manifest)
  of "int16": run[int16](manifest)
  of "int32": run[int32](manifest)
  of "int64": run[int64](manifest)
  of "uint8": run[uint8](manifest)
  of "uint16": run[uint16](manifest)
  of "uint32": run[uint32](manifest)
  of "uint64": run[uint64](manifest)
  else: raise newException(ValueError, "Unsupported integer benchmark dtype")

when isMainModule:
  if paramCount() != 1:
    echo "Usage: integer_runner INPUT.json"
    quit(if paramCount() == 0: 0 else: 2)
  let manifest = parseFile(paramStr(1))
  if manifest["rows"].getInt != 128 or manifest["columns"].getInt != 128:
    raise newException(ValueError, "Integer benchmark input must be 128x128")
  echo runForDtype(manifest)
