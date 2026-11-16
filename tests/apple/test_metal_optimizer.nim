# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Verifies separate native SGD update/reset semantics and preserved storage identity.
import std/unittest
import ../../src/arraymancer

proc checkValues(t: MetalTensor[float32]; expected: openArray[float32]) =
  check t.storage != nil
  let host = t.cpu
  check host.size == expected.len
  let values = cast[ptr UncheckedArray[float32]](host.get_offset_ptr)
  for i, value in expected: check abs(values[i] - value) < 1e-6'f32

suite "Metal SGD lifecycle":
  test "Training step leaves gradients intact until explicit reset":
    let ctx = newContext(MetalTensor[float32])
    let weights = ctx.variable([1'f32, 2].toTensor.metal, requires_grad = true)
    let frozen = ctx.variable([9'f32].toTensor.metal)
    let input = ctx.variable([3'f32, 4].toTensor.metal)
    let prediction = (weights *. input).sum
    let loss = prediction.mse_loss(newTensorWith[float32]([], 0'f32).metal)
    check prediction.value.rank == 0
    check loss.value.rank == 0
    prediction.value.checkValues([11'f32])
    loss.value.checkValues([121'f32])
    loss.backprop
    weights.grad.checkValues([66'f32, 88])
    let valueOwner = weights.value.storage
    let gradOwner = weights.grad.storage
    let optim = optimizer((trainable: @[weights], frozen: frozen), SGD, 0.01'f32, MetalTensor[float32])
    optim.step
    check weights.value.storage == valueOwner
    check weights.grad.storage == gradOwner
    check weights.grad.storage.device.registryId == weights.value.storage.device.registryId
    weights.value.checkValues([0.34'f32, 1.12])
    weights.grad.checkValues([66'f32, 88])
    var evidence: AMEvidence
    checkMetal metalApi().evidence(weights.value.storage.owner.handle, addr evidence)
    check evidence.completed == 1
    check evidence.kind == 2
    check evidence.kernelBits == 107
    optim.zeroGrads
    check weights.grad.storage == gradOwner
    weights.grad.checkValues([0'f32, 0])
    frozen.value.checkValues([9'f32])
    weights.grad.fill(2'f32)
    optim.update
    weights.value.checkValues([0.32'f32, 1.10])
    weights.grad.checkValues([0'f32, 0])
    check weights.grad.storage == gradOwner
    check weights.value.storage == valueOwner
  test "Mixed backends and unsafe gradient storage fail without mutation":
    let ctx = newContext(MetalTensor[float32])
    let weights = ctx.variable([1'f32, 2].toTensor.metal, requires_grad = true)
    let cpuContext = newContext(Tensor[float32])
    let cpuWeight = cpuContext.variable([3'f32].toTensor, requires_grad = true)
    expect ValueError:
      discard optimizer((native: weights, host: cpuWeight), SGD, 0.01'f32, MetalTensor[float32])
    let optim = optimizer(weights, SGD, 0.01'f32, MetalTensor[float32])
    weights.grad = [1'f32].toTensor.metal
    expect MetalError: optim.step
    weights.grad = weights.value
    expect MetalError: optim.step
    expect MetalError: optim.zeroGrads
    weights.value.checkValues([1'f32, 2])
    weights.grad = weights.value.broadcast(2, 2)
    expect MetalError: optim.step
