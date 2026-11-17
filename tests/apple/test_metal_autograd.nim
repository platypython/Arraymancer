# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises the existing tape with GPU-resident reductions and fused MSE gradients.
import std/[unittest, math]
import ../../src/arraymancer

proc checkValues(t: MetalTensor[float32]; shape: openArray[int]; expected: openArray[float32]) =
  check t.storage != nil
  check t.storage.owner.handle != nil
  check t.shape == shape.toMetadata
  check t.size == expected.len
  let host = t.cpu
  let values = cast[ptr UncheckedArray[float32]](host.get_offset_ptr)
  for i, value in expected: check abs(values[i] - value) < 1e-6'f32

suite "Metal autograd on the existing tape":
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
