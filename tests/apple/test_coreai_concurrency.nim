# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises independent calls on one Sendable function and deterministic mutable conflicts.
import std/[options, unittest]
import ../../src/arraymancer
import ./coreai_test_support

suite "Core AI concurrent call isolation":
  test "Shared identity function retains independent inputs and writable outputs":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity").get
    var firstJob, secondJob: CoreAIJob
    var firstOutput, secondOutput: CoreAIArray
    var firstIdentity, secondIdentity: uint64
    block:
      let first = [0'f32, 0].toTensor.metal
      let second = [0'f32, 0].toTensor.metal
      firstIdentity = originalMetalBufferIdentity(first)
      secondIdentity = originalMetalBufferIdentity(second)
      firstOutput = coreAIView(first)
      secondOutput = coreAIView(second)
      firstJob = function.submitInference([CoreAIBinding(name: "x", array: toCoreAI([1'f32, 2].toTensor))],
        states = [], outputViews = [CoreAIBinding(name: "y", array: firstOutput)])
      secondJob = function.submitInference([CoreAIBinding(name: "x", array: toCoreAI([3'f32, 4].toTensor))],
        states = [], outputViews = [CoreAIBinding(name: "y", array: secondOutput)])
    GC_fullCollect()
    check firstJob.wait().len == 0
    check secondJob.wait().len == 0
    check firstIdentity != secondIdentity
    check firstOutput.descriptor.bufferIdentity == firstIdentity
    check secondOutput.descriptor.bufferIdentity == secondIdentity
    check firstOutput.cpu(float32) == [1'f32, 2].toTensor
    check secondOutput.cpu(float32) == [3'f32, 4].toTensor
    check firstJob.coreAIJobEvidence.invocationId != secondJob.coreAIJobEvidence.invocationId
    discard function.infer([CoreAIBinding(name: "x", array: toCoreAI([9'f32, 10].toTensor))],
      states = [], outputViews = [CoreAIBinding(name: "y", array: firstOutput)])
    check firstOutput.cpu(float32) == [9'f32, 10].toTensor
    check secondOutput.cpu(float32) == [3'f32, 4].toTensor

  test "Concurrent stateful calls use separate zero states and allocated outputs":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("increment").get
    let inputs = [CoreAIBinding(name: "delta", array: toCoreAI(newTensorWith[float32]([], 1'f32)))]
    for iteration in 0 ..< 20:
      let firstState = toCoreAI(newTensorWith[float32]([], 0'f32))
      let secondState = toCoreAI(newTensorWith[float32]([], 0'f32))
      let first = function.submitInference(inputs,
        states = [CoreAIBinding(name: "counter", array: firstState)], outputViews = [])
      let second = function.submitInference(inputs,
        states = [CoreAIBinding(name: "counter", array: secondState)], outputViews = [])
      let firstOutputs = first.wait()
      let secondOutputs = second.wait()
      check firstOutputs[0].array.cpu(float32) == newTensorWith[float32]([], 1'f32)
      check secondOutputs[0].array.cpu(float32) == newTensorWith[float32]([], 1'f32)
      check firstState.cpu(float32) == newTensorWith[float32]([], 1'f32)
      check secondState.cpu(float32) == newTensorWith[float32]([], 1'f32)
      check first.coreAIJobEvidence.invocationId != second.coreAIJobEvidence.invocationId
      let next = function.infer(inputs,
        states = [CoreAIBinding(name: "counter", array: firstState)], outputViews = [])
      check next[0].array.cpu(float32) == newTensorWith[float32]([], 2'f32)
      check firstOutputs[0].array.cpu(float32) == newTensorWith[float32]([], 1'f32)
      check secondOutputs[0].array.cpu(float32) == newTensorWith[float32]([], 1'f32)

  test "Gated state rejects overlap while independent ordinary inference completes":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("increment").get
    let queue = newMetalStream()
    let stream = newCoreAIStream(queue)
    let gate = newMetalEvent()
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      metalSynchronize(queue)
    let state = toCoreAI(newTensorWith[float32]([], 0'f32))
    let inputs = [CoreAIBinding(name: "delta", array: toCoreAI(newTensorWith[float32]([], 1'f32)))]
    let states = [CoreAIBinding(name: "counter", array: state)]
    wait(queue, gate, 1)
    let pending = function.encodeInference(inputs, states = states, outputViews = [], stream = stream)
    expect CoreAITimeoutError: discard pending.wait(timeoutMs = 0)
    expect CoreAIBufferBusyError:
      discard function.submitInference(inputs, states = states, outputViews = [])
    let independentState = toCoreAI(newTensorWith[float32]([], 0'f32))
    let independent = function.submitInference(inputs,
      states = [CoreAIBinding(name: "counter", array: independentState)], outputViews = [])
    check independent.wait()[0].array.cpu(float32) == newTensorWith[float32]([], 1'f32)
    expect CoreAITimeoutError: discard pending.wait(timeoutMs = 0)
    signalMetalEvent(gate, 1)
    released = true
    check pending.wait()[0].array.cpu(float32) == newTensorWith[float32]([], 1'f32)
    discard awaitCurrentWorkCompleted(stream)
    check function.infer(inputs, states = states, outputViews = [])[0].array.cpu(float32) == newTensorWith[float32]([], 2'f32)
    check independentState.cpu(float32) == newTensorWith[float32]([], 1'f32)
