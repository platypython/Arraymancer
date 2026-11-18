# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Checks native inclusive scans and VJPs on the pinned oracle's shared exact inputs.
import std/[json, math, unittest]
import ../../src/arraymancer

proc inputTensor(desc: JsonNode): Tensor[float32] =
  var shape = newSeq[int](desc["shape"].len)
  for index in 0 ..< shape.len: shape[index] = desc["shape"][index].getInt
  result = newTensorUninit[float32](toMetadata(shape))
  var raw = result.unsafe_raw_offset
  for index in 0 ..< result.size: raw[index] = float32(desc["values"][index].getFloat)

proc checkObservation(actual: MetalTensor[float32], expected: JsonNode) =
  let host = actual.cpu
  var shape = newSeq[int](expected["shape"].len)
  for index in 0 ..< shape.len: shape[index] = expected["shape"][index].getInt
  check host.shape == toMetadata(shape)
  check host.size == expected["numel"].getInt
  for index, value in enumerate(host):
    let target = expected["values"][index]
    if target.kind == JString:
      case target.getStr
      of "nan": check classify(value) == fcNan
      of "+inf": check classify(value) == fcInf
      of "-inf": check classify(value) == fcNegInf
      else: raise newException(ValueError, "Unknown nonfinite observation")
    else:
      let reference = target.getFloat
      check abs(float64(value) - reference) <= 5e-5 * max(1.0, abs(reference))

proc checkOracle(fixtureId: string) =
  let fixtures = parseFile("tests/parity/fixtures.json")
  var found = false
  for fixture in fixtures["cases"]:
    if fixture["id"].getStr != fixtureId: continue
    found = true
    for name, spec in fixture["inputs"]:
      var host = inputTensor(spec["x"])
      for special in spec["nonfinite"]:
        let value = case special["value"].getStr
                    of "nan": NaN.float32
                    of "-inf": NegInf.float32
                    else: Inf.float32
        host.unsafe_raw_offset[special["index"].getInt] = value
      var source = host.metal
      case spec["layout"].getStr
      of "transpose": source = source.transpose
      of "offset": source = source.transpose[_, 1..^1]
      of "reverse": source = source[^1..0|-1]
      of "broadcast": source = source.broadcast(4, 3)
      else: discard
      for index in 0 ..< spec["operations"].len:
        checkpoint name & " operation " & $index
        let operation = spec["operations"][index]
        let context = newContext(MetalTensor[float32])
        let x = context.variable(source, requires_grad = true)
        let output = if fixtureId == "autograd.cumprod_gradient":
          x.cumprod(operation["axis"].getInt)
        else:
          x.cumsum(operation["axis"].getInt)
        var seed = inputTensor(operation["seed"])
        for special in operation["seed_nonfinite"]:
          seed.unsafe_raw_offset[special["index"].getInt] =
            (case special["value"].getStr
             of "nan": NaN.float32
             of "-inf": NegInf.float32
             else: Inf.float32)
        output.backprop(seed.metal)
        checkObservation(output.value, fixture["expected"][name][index]["output"])
        checkObservation(x.grad, fixture["expected"][name][index]["grad"])
  check found

suite "Metal cumulative-sum derivative oracle":
  test "Inclusive scans and reverse VJPs preserve axis, view and nonfinite semantics":
    checkOracle("autograd.cumsum_gradient")

suite "Metal cumulative-product derivative oracle":
  test "Inclusive scans and reverse VJPs preserve axis, view and nonfinite semantics":
    checkOracle("autograd.cumprod_gradient")
