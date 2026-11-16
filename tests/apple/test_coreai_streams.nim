# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises actual SDK encoding, preallocated shared outputs and queue-local completion.
import std/[options, unittest]
import ../../src/arraymancer
import ./coreai_test_support

suite "Core AI same-queue inference":
  test "Metal producer, encoded identity and Metal consumer share one queue without readback":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    let queue = newMetalStream()
    let stream = newCoreAIStream(queue)
    let unrelated = newMetalStream()
    let gate = newMetalEvent()
    defer:
      signalMetalEvent(gate, 1)
      metalSynchronize(unrelated)
    wait(unrelated, gate, 1)
    let source = [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal
    var produced: MetalTensor[float32]
    withMetalStream(queue): produced = source + 1'f32
    let output = newTensorWith[float32]([2, 3], 0'f32).metal
    let outputIdentity = originalMetalBufferIdentity(output)
    let inputView = coreAIView(produced)
    let outputView = coreAIView(output)
    let job = function.encodeInference([CoreAIBinding(name: "x", array: inputView)],
      states = [], outputViews = [CoreAIBinding(name: "y", array: outputView)], stream = stream)
    check job.wait().len == 0
    let inference = job.coreAIJobEvidence
    check inference.completed and inference.sdkStreamCompleted
    check not inference.metalQueueCompleted
    check inference.streamIdentity == queue.identity
    check inference.hostCopyBytes == 0 and inference.materializationBytes == 0
    check inference.borrowedInputCount == 1
    check inputView.descriptor.hostCopyBytes == 0
    check outputView.descriptor.hostCopyBytes == 0
    check outputView.descriptor.bufferIdentity == outputIdentity
    var consumer: MetalTensor[float32]
    withMetalStream(queue): consumer = output * 2'f32
    let boundary = awaitCurrentWorkCompleted(stream)
    check boundary.completed and boundary.sdkStreamCompleted and boundary.metalQueueCompleted
    check boundary.streamIdentity == queue.identity
    check metalStreamIsComplete(queue)
    check not metalStreamIsComplete(unrelated)
    check consumer.cpu == [[4'f32, 6, 8], [10'f32, 12, 14]].toTensor
    check inputView.descriptor.hostCopyBytes == 0
    check outputView.descriptor.hostCopyBytes == 0

  test "Queue marker waits preceding work and mutable leases reject access until completion":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    let queue = newMetalStream()
    let stream = newCoreAIStream(queue)
    let gate = newMetalEvent()
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      metalSynchronize(queue)
    var input = [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal
    var output = newTensorWith[float32]([2, 3], 0'f32).metal
    let inputView = coreAIView(input)
    let outputView = coreAIView(output)
    wait(queue, gate, 1)
    let job = function.encodeInference([CoreAIBinding(name: "x", array: inputView)],
      states = [], outputViews = [CoreAIBinding(name: "y", array: outputView)], stream = stream)
    expect CoreAITimeoutError: discard job.wait(timeoutMs = 0)
    expect CoreAITimeoutError: discard awaitCurrentWorkCompleted(stream, timeoutMs = 0)
    expect MetalBufferBusyError: input[0, 0] = 99'f32
    expect MetalBufferBusyError: output[0, 0] = 99'f32
    expect MetalBufferBusyError: discard output.cpu
    expect CoreAIBufferBusyError:
      discard function.encodeInference([CoreAIBinding(name: "x", array: inputView)],
        states = [], outputViews = [CoreAIBinding(name: "y", array: outputView)], stream = stream)
    signalMetalEvent(gate, 1)
    released = true
    check job.wait().len == 0
    let boundary = awaitCurrentWorkCompleted(stream)
    check boundary.sdkStreamCompleted and boundary.metalQueueCompleted
    check output.cpu == [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor
    output[0, 0] = 7'f32
    check output.cpu[0, 0] == 7'f32

  test "Native jobs retain creating queue, function and buffer owners after collection":
    let gate = newMetalEvent()
    var released = false
    var job: CoreAIJob
    var outputView: CoreAIArray
    var identity: uint64
    defer:
      if not released: signalMetalEvent(gate, 1)
      if job != nil: discard job.wait()
    block:
      let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
      let queue = newMetalStream()
      let stream = newCoreAIStream(queue)
      let input = [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal
      let output = newTensorWith[float32]([2, 3], 0'f32).metal
      identity = originalMetalBufferIdentity(output)
      outputView = coreAIView(output)
      wait(queue, gate, 1)
      job = function.encodeInference([CoreAIBinding(name: "x", array: coreAIView(input))],
        states = [], outputViews = [CoreAIBinding(name: "y", array: outputView)], stream = stream)
    GC_fullCollect()
    metalEmptyCache()
    let pressure = newTensorWith[float32]([2, 3], -1'f32).metal
    expect CoreAITimeoutError: discard job.wait(timeoutMs = 0)
    signalMetalEvent(gate, 1)
    released = true
    check job.wait().len == 0
    check outputView.descriptor.bufferIdentity == identity
    check outputView.descriptor.hostCopyBytes == 0
    check outputView.cpu(float32) == [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor
    check pressure.cpu == newTensorWith[float32]([2, 3], -1'f32)

  test "Caller output names, dtype and static dimensions are checked before encoding":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    let stream = newCoreAIStream(newMetalStream())
    let input = CoreAIBinding(name: "x", array: coreAIView([[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal))
    let valid = coreAIView(newTensorWith[float32]([2, 3], 0'f32).metal)
    expect CoreAIDescriptorError:
      discard function.encodeInference([input], states = [], outputViews = [CoreAIBinding(name: "unknown", array: valid)], stream = stream)
    expect CoreAIDescriptorError:
      discard function.encodeInference([input], states = [], outputViews = [CoreAIBinding(name: "y", array: valid),
        CoreAIBinding(name: "y", array: valid)], stream = stream)
    expect CoreAIDescriptorError:
      discard function.encodeInference([input], states = [], outputViews = [CoreAIBinding(name: "y",
        array: coreAIView(newTensorWith[int32]([2, 3], 0'i32).metal))], stream = stream)
    expect CoreAIDescriptorError:
      discard function.encodeInference([input], states = [], outputViews = [CoreAIBinding(name: "y",
        array: coreAIView(newTensorWith[float32]([2, 4], 0'f32).metal))], stream = stream)
    check function.encodeInference([input], states = [], outputViews = [CoreAIBinding(name: "y", array: valid)],
      stream = stream).wait().len == 0
    check valid.cpu(float32) == [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor
