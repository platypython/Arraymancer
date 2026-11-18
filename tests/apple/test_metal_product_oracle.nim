# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Checks native product VJPs against the pinned CPU oracle's edge observations.
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

suite "Metal product derivative oracle":
  test "Full and axis seeded products preserve zero/nonfinite and view semantics":
    let fixtures = parseFile("tests/parity/fixtures.json")
    var found = false
    for fixture in fixtures["cases"]:
      if fixture["id"].getStr != "autograd.product_gradient": continue
      found = true
      for name, spec in fixture["inputs"]:
        var host = inputTensor(spec["x"])
        for special in spec["nonfinite"]:
          host.unsafe_raw_offset[special["index"].getInt] =
            (if special["value"].getStr == "nan": NaN.float32 else: Inf.float32)
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
          let output =
            if operation["axis"].kind == JNull: x.product
            else: x.product(operation["axis"].getInt, keepdim = operation["keepdim"].getBool)
          output.backprop(inputTensor(operation["seed"]).metal)
          checkObservation(output.value, fixture["expected"][name][index]["output"])
          checkObservation(x.grad, fixture["expected"][name][index]["grad"])
    check found

  test "Full product gradients merge zero counts across native partial blocks":
    const count = 65537
    for zeros in 0 .. 2:
      var host = ones[float32](count)
      if zeros >= 1: host[count - 1] = 0'f32
      if zeros == 2: host[0] = 0'f32
      let input = host.metal
      let gradient = productGradient(input, input.prod,
        newTensorWith[float32]([], 2'f32).metal, -1).cpu
      check gradient.shape == [count]
      for index, value in enumerate(gradient):
        let expected =
          if zeros == 0 or (zeros == 1 and index == count - 1): 2'f32
          else: 0'f32
        check value == expected
