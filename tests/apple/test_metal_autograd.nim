# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises GPU-resident view, matrix-product and loss cotangents on the existing tape.
import std/[unittest, math, json]
import ../../src/arraymancer

proc checkValues(t: MetalTensor[float32]; shape: openArray[int]; expected: openArray[float32];
                 relativeTolerance = 0'f32) =
  check t.storage != nil
  check t.storage.owner.handle != nil
  check t.shape == shape.toMetadata
  check t.size == expected.len
  let host = t.cpu
  let values = cast[ptr UncheckedArray[float32]](host.get_offset_ptr)
  for i, value in expected:
    case value.classify
    of fcNan: check values[i].classify == fcNan
    of fcInf, fcNegInf: check values[i] == value
    else: check abs(values[i] - value) < 1e-6'f32 + relativeTolerance * abs(value)

proc fixtureData(spec: JsonNode): tuple[shape: seq[int], values: seq[float32]] =
  ## Cast pinned CPU observations to the dtype supported by native Metal arithmetic.
  for dim in spec["shape"]: result.shape.add(dim.getInt)
  for value in spec["values"]:
    if value.kind == JString:
      case value.getStr
      of "nan": result.values.add(NaN.float32)
      of "+inf": result.values.add(Inf.float32)
      of "-inf": result.values.add(-Inf.float32)
      else: raise newException(ValueError, "unsupported fixture float observation")
    else:
      result.values.add(float32(value.getFloat))

proc fixtureTensor(spec: JsonNode): MetalTensor[float32] =
  let data = fixtureData(spec)
  data.values.toTensor.reshape(data.shape).metal

proc checkFixture(t: MetalTensor[float32], spec: JsonNode; relativeTolerance = 0'f32) =
  let data = fixtureData(spec)
  t.checkValues(data.shape, data.values, relativeTolerance)

suite "Metal autograd on the existing tape":
  test "Native exp, ln and sqrt VJPs match pinned scalar, strided, empty and domain cases":
    const fixtures = staticRead("../parity/fixtures.json")
    var contract: JsonNode
    for item in parseJson(fixtures)["cases"]:
      if item["id"].getStr == "autograd.unary_math": contract = item
    doAssert not contract.isNil
    for name, spec in contract["inputs"]:
      for operation in ["exp", "ln", "sqrt"]:
        checkpoint name & ":" & operation
        let ctx = newContext(MetalTensor[float32])
        var value = fixtureTensor(spec["a"])
        if name == "strided": value = value.transpose[_, 1..^1]
        elif name == "empty": value = value.transpose(0, 2)
        let x = ctx.variable(value, requires_grad = true)
        let output = case operation
                     of "exp": x.exp
                     of "ln": x.ln
                     else: x.sqrt
        output.backprop(fixtureTensor(spec["seed"]))
        output.value.checkFixture(contract["expected"][name][operation]["output"], 2e-6'f32)
        x.grad.checkFixture(contract["expected"][name][operation]["grad"], 2e-6'f32)

  test "Native unary chains honor no-grad and scalar cotangents":
    let ctx = newContext(MetalTensor[float32])
    let x = ctx.variable(newTensorWith[float32]([], 4'f32).metal, requires_grad = true)
    ctx.no_grad_mode:
      check not x.exp.requires_grad
      check not x.ln.requires_grad
      check not x.sqrt.requires_grad
    let output = x.exp.ln.sqrt
    output.backprop(newTensorWith[float32]([], 3'f32).metal)
    output.value.checkValues([], [2'f32])
    x.grad.checkValues([], [0.75'f32])

  test "Native singleton views match pinned PyTorch values, aliases and VJPs":
    const fixtures = staticRead("../parity/fixtures.json")
    var contract: JsonNode
    for item in parseJson(fixtures)["cases"]:
      if item["id"].getStr == "autograd.singleton_views": contract = item
    doAssert not contract.isNil
    let inputs = contract["inputs"]
    let expected = contract["expected"]
    let ctx = newContext(MetalTensor[float32])
    let x = ctx.variable(fixtureTensor(inputs["x"]).transpose(0, 2), requires_grad = true)
    let output = x.squeeze(-2).unsqueeze(-1).squeeze(-1)
    output.backprop(fixtureTensor(inputs["seed"]))
    let noop = x.squeeze(-1)
    noop.backprop(noop.value.ones_like)
    output.value.checkFixture(expected["output"])
    noop.value.checkFixture(expected["noop"])
    x.grad.checkFixture(expected["grad"])
    check (output.value.storage == x.value.storage) == expected["shares_storage"].getBool
    let scalar = ctx.variable(fixtureTensor(inputs["scalar"]), requires_grad = true)
    let scalarOutput = scalar.squeeze.unsqueeze(-1).squeeze(0).squeeze(-1)
    scalarOutput.backprop(newTensorWith[float32]([], 3'f32).metal)
    scalarOutput.value.checkFixture(expected["scalar"])
    scalar.grad.checkFixture(expected["scalar_grad"])
    ctx.no_grad_mode:
      check not scalar.squeeze.requires_grad
      check not scalar.unsqueeze(-1).requires_grad

  test "Native singleton views preserve empty shapes and bound inserted ranks":
    let ctx = newContext(MetalTensor[float32])
    let x = ctx.variable(newMetalTensorUninit[float32](2, 0, 1), requires_grad = true)
    let output = x.squeeze(-1).unsqueeze(-1)
    output.backprop()
    x.grad.checkValues([2, 0, 1], [])
    expect MetalError: discard x.value.squeeze(-4)
    expect MetalError: discard x.value.unsqueeze(-5)
    var shape: Metadata
    shape.len = min(LASER_MAXRANK, 8)
    for i in 0 ..< shape.len: shape[i] = 1
    let fullRank = newMetalTensorUninit[float32](shape)
    expect MetalError: discard fullRank.unsqueeze(0)

  test "Native matrix product ranks and VJPs match all eight pinned PyTorch shapes":
    const fixtures = staticRead("../parity/fixtures.json")
    var contract: JsonNode
    for item in parseJson(fixtures)["cases"]:
      if item["id"].getStr == "autograd.matmul_ranks": contract = item
    doAssert not contract.isNil
    for name, inputs in contract["inputs"]:
      checkpoint name
      let ctx = newContext(MetalTensor[float32])
      let a = ctx.variable(fixtureTensor(inputs["a"]), requires_grad = true)
      let b = ctx.variable(fixtureTensor(inputs["b"]), requires_grad = true)
      let output = a * b
      output.backprop(fixtureTensor(inputs["seed"]))
      let expected = contract["expected"][name]
      output.value.checkFixture(expected["output"])
      a.grad.checkFixture(expected["grad_a"])
      b.grad.checkFixture(expected["grad_b"])

  test "Native vector products preserve reversed views and reject invalid shapes":
    let storage = [99'f32, 1, 2, 3, 99].toTensor.metal
    let reversed = storage.metalView([3].toMetadata, [-1].toMetadata, 3)
    (reversed * [1'f32, 2, 3].toTensor.metal).checkValues([], [10'f32])
    expect MetalError:
      discard newTensorWith[float32]([], 1'f32).metal * storage
    expect MetalError:
      discard ones[float32](2, 3).metal * ones[float32](2, 2).metal
    expect MetalError:
      discard ones[float32](2, 2, 3).metal * ones[float32](3, 3, 2).metal

  test "MSE reductions broadcast prediction and target VJPs on the GPU":
    for reduction in [mseMean, mseSum, mseNone]:
      let ctx = newContext(MetalTensor[float32])
      let prediction = ctx.variable([[2'f32], [4'f32]].toTensor.metal, requires_grad = true)
      let target = ctx.variable([1'f32, 2, 3].toTensor.metal, requires_grad = true)
      let loss = prediction.mse_loss(target, reduction)
      if reduction == mseNone:
        loss.value.checkValues([2, 3], [1'f32, 0, 1, 9, 4, 1])
        loss.backprop([[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal)
        prediction.grad.checkValues([2, 1], [-4'f32, 56])
        target.grad.checkValues([3], [-26'f32, -20, -6])
      else:
        let divisor = if reduction == mseMean: 6'f32 else: 1'f32
        loss.value.checkValues([], [16'f32 / divisor])
        loss.backprop()
        prediction.grad.checkValues([2, 1], [0'f32, 12'f32 / divisor])
        target.grad.checkValues([3], [-8'f32 / divisor, -4'f32 / divisor, 0])

  test "Empty broadcast MSE reduces to zero sums and NaN means on the GPU":
    for reduction in [mseMean, mseSum, mseNone]:
      let ctx = newContext(MetalTensor[float32])
      let prediction = ctx.variable(ones[float32](1, 3).metal, requires_grad = true)
      let target = ctx.variable(newMetalTensorUninit[float32](0, 3), requires_grad = true)
      let loss = prediction.mse_loss(target, reduction)
      if reduction == mseNone:
        loss.value.checkValues([0, 3], [])
      elif reduction == mseSum:
        loss.value.checkValues([], [0'f32])
      else:
        check loss.value.rank == 0
        check cast[ptr float32](loss.value.cpu.get_offset_ptr)[].classify == fcNan
      loss.backprop()
      prediction.grad.checkValues([1, 3], [0'f32, 0, 0])
      target.grad.checkValues([0, 3], [])

  test "Broadcast division reduces both GPU gradients to their operand shapes":
    let ctx = newContext(MetalTensor[float32])
    let a = ctx.variable([[2'f32], [4'f32]].toTensor.metal, requires_grad = true)
    let b = ctx.variable([1'f32, 2, 4].toTensor.metal, requires_grad = true)
    let output = a /. b
    output.backprop([[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal)
    a.grad.checkValues([2, 1], [2.75'f32, 8])
    b.grad.checkValues([3], [-18'f32, -6, -1.875])

  test "Negative-axis mean and sum restore squeezed GPU gradient dimensions":
    let ctx = newContext(MetalTensor[float32])
    let x = ctx.variable([[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal,
                         requires_grad = true)
    let total = x.sum(axis = -1, keepdim = false)
    let average = x.mean(axis = -1, keepdim = false)
    total.value.checkValues([2], [6'f32, 15])
    average.value.checkValues([2], [2'f32, 5])
    total.backprop([2'f32, 4].toTensor.metal)
    average.backprop([3'f32, 6].toTensor.metal)
    x.grad.checkValues([2, 3], [3'f32, 3, 3, 6, 6, 6])

  test "Retained seeded VJPs recompute GPU intermediates and accumulate leaves":
    let ctx = newContext(MetalTensor[float32])
    let x = ctx.variable([2'f32, 3].toTensor.metal, requires_grad = true)
    let squared = x *. x
    let output = squared *. squared
    let seed = [1'f32, 2].toTensor.metal
    output.backprop(seed, retain_graph = true)
    x.grad.checkValues([2], [32'f32, 216])
    output.backprop(seed)
    x.grad.checkValues([2], [64'f32, 432])
    seed.checkValues([2], [1'f32, 2])
    expect ValueError:
      output.backprop(seed)
  test "Mean and MSE preserve native scalar values and gradients":
    let ctx = newContext(MetalTensor[float32])
    let x = ctx.variable([[1'f32, 2], [3'f32, 4]].toTensor.metal, requires_grad = true)
    let loss = x.mean
    check loss.value.storage.device.registryId == x.value.storage.device.registryId
    loss.value.checkValues([], [2.5'f32])
    loss.backprop
    x.grad.checkValues([2, 2], [0.25'f32, 0.25, 0.25, 0.25])
    let p = ctx.variable([[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal, requires_grad = true)
    let target = zeros[float32](2, 3).metal
    let mse = p.mse_loss(target)
    mse.value.checkValues([], [91'f32 / 6])
    mse.backprop
    p.grad.checkValues([2, 3], [1'f32/3, 2'f32/3, 1, 4'f32/3, 5'f32/3, 2])
    check p.grad.storage.device.registryId == p.value.storage.device.registryId
  test "Scalar and empty means keep their metadata through backward":
    let ctx = newContext(MetalTensor[float32])
    let scalar = ctx.variable(newTensorWith[float32]([], 4'f32).metal, requires_grad = true)
    let loss = scalar.mean
    loss.backprop
    scalar.grad.checkValues([], [1'f32])
    let empty = ctx.variable(newMetalTensorUninit[float32](2, 0, 3), requires_grad = true)
    let emptyMean = empty.mean
    check emptyMean.value.rank == 0
    check cast[ptr float32](emptyMean.value.cpu.get_offset_ptr)[].classify == fcNan
    emptyMean.backprop
    empty.grad.checkValues([2, 0, 3], [])
    let emptyMse = empty.mse_loss(newMetalTensorUninit[float32](2, 0, 3))
    check cast[ptr float32](emptyMse.value.cpu.get_offset_ptr)[].classify == fcNan
    emptyMse.backprop
    empty.grad.checkValues([2, 0, 3], [])
  test "Broadcast gradients unbroadcast both operands on the GPU":
    let ctx = newContext(MetalTensor[float32])
    let a = ctx.variable([[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal, requires_grad = true)
    let b = ctx.variable([2'f32, 3, 4].toTensor.metal, requires_grad = true)
    let loss = (a *. b).sum
    loss.backprop
    a.grad.checkValues([2, 3], [2'f32, 3, 4, 2, 3, 4])
    b.grad.checkValues([3], [5'f32, 7, 9])
    let scalar = ctx.variable(newTensorWith[float32]([], 2'f32).metal, requires_grad = true)
    let total = (a +. scalar).sum
    total.backprop
    scalar.grad.checkValues([], [6'f32])
  test "No-grad restores tracing after an exception":
    let ctx = newContext(MetalTensor[float32])
    let x = ctx.variable([1'f32, 3].toTensor.metal, requires_grad = true)
    expect ValueError:
      ctx.no_grad_mode:
        check not x.is_grad_needed
        check not x.mean.requires_grad
        raise newException(ValueError, "intentional no-grad exit")
    check x.is_grad_needed
    let loss = x.mean
    loss.backprop
    x.grad.checkValues([2], [0.5'f32, 0.5])
  test "Reshape owns GPU materialization and rejects invalid metadata":
    let x = [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal
    let alias = x.reshape(3, 2)
    check alias.storage == x.storage
    let packed = x.transpose.reshape(6)
    check packed.storage != x.storage
    packed.checkValues([6], [1'f32, 4, 2, 5, 3, 6])
    check x.transpose(0, 1).shape == [3, 2].toMetadata
    check x.transpose2d.shape == [3, 2].toMetadata
    expect MetalError: discard x.reshape(5)
    check x.squeeze(0).shape == x.shape
    expect MetalError: discard x.squeeze(2)
    expect MetalError: discard x.broadcast(2, 4)
    expect MetalError: discard mse_forward_metal(x, zeros[float32](2, 2).metal)
    expect MetalError: discard mse_backward_metal(x, x, x)
    let upstream = [99'f32, 0.5].toTensor.metal.metalView(Metadata(), Metadata(), 1)
    let gradient = mse_backward_metal(upstream, x, zeros[float32](2, 3).metal)
    gradient.checkValues([2, 3], [1'f32/6, 2'f32/6, 0.5, 4'f32/6, 5'f32/6, 1])
