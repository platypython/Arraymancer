# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises model-declared state, owned/shared mutable views and exclusive pending access.
import std/[options, unittest]
import ../../src/arraymancer
import ./coreai_test_support

proc scalar(value: float32): Tensor[float32] =
  ## A rank-zero tensor matches the exported counter, not a length-one vector.
  newTensorWith[float32]([], value)

suite "Core AI declared mutable state":
  test "Source and AOT declare scalar counter and preserve allocated clone outputs":
    for path in coreAIModelPaths():
      let function = openCoreAIModel(path).loadFunction("increment").get
      let descriptor = function.descriptor
      check descriptor.inputs.len == 1 and descriptor.outputs.len == 1 and descriptor.states.len == 1
      check descriptor.inputs[0].name == "delta"
      check descriptor.outputs[0].name == "y"
      check descriptor.states[0].name == "counter"
      for item in [descriptor.inputs[0], descriptor.outputs[0], descriptor.states[0]]:
        check item.dtype == "float32" and item.shape.len == 0
        check item.resolved and not item.dynamicShape
        check item.minimumBytes >= 4
      let state = toCoreAI(scalar(0))
      let delta = toCoreAI(scalar(1))
      let first = function.infer([CoreAIBinding(name: "delta", array: delta)],
        states = [CoreAIBinding(name: "counter", array: state)], outputViews = [])
      check first.len == 1 and first[0].name == "y"
      check first[0].array.cpu(float32) == scalar(1)
      check state.cpu(float32) == scalar(1)
      let job = function.submitInference([CoreAIBinding(name: "delta", array: delta)],
        states = [CoreAIBinding(name: "counter", array: state)], outputViews = [])
      let second = job.wait()
      check second.len == 1 and second[0].name == "y"
      check second[0].array.cpu(float32) == scalar(2)
      check state.cpu(float32) == scalar(2)
      check first[0].array.cpu(float32) == scalar(1)
      check job.coreAIJobEvidence.completed

  test "Owned caller outputs mutate through ordinary run and stream encode":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("increment").get
    for encoded in [false, true]:
      let state = toCoreAI(scalar(0))
      let output = toCoreAI(scalar(-1))
      let inputs = [CoreAIBinding(name: "delta", array: toCoreAI(scalar(1)))]
      let states = [CoreAIBinding(name: "counter", array: state)]
      let outputs = [CoreAIBinding(name: "y", array: output)]
      let stream = if encoded: newCoreAIStream(newMetalStream()) else: nil
      for expected in [1'f32, 2'f32]:
        let job = if encoded:
          function.encodeInference(inputs, states = states, outputViews = outputs, stream = stream)
        else:
          function.submitInference(inputs, states = states, outputViews = outputs)
        check job.wait().len == 0
        if encoded:
          let boundary = awaitCurrentWorkCompleted(stream)
          check boundary.sdkStreamCompleted and boundary.metalQueueCompleted
        check state.cpu(float32) == scalar(expected)
        check output.cpu(float32) == scalar(expected)

  test "Borrowed state and output keep native identities without bridge copies":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("increment").get
    let state = scalar(0).metal
    let output = scalar(-1).metal
    let stateIdentity = originalMetalBufferIdentity(state)
    let outputIdentity = originalMetalBufferIdentity(output)
    let stateView = coreAIView(state)
    let outputView = coreAIView(output)
    let inputs = [CoreAIBinding(name: "delta", array: toCoreAI(scalar(1)))]
    let states = [CoreAIBinding(name: "counter", array: stateView)]
    let outputs = [CoreAIBinding(name: "y", array: outputView)]
    check function.infer(inputs, states = states, outputViews = outputs).len == 0
    let stream = newCoreAIStream(newMetalStream())
    let job = function.encodeInference(inputs, states = states, outputViews = outputs, stream = stream)
    check job.wait().len == 0
    discard awaitCurrentWorkCompleted(stream)
    check stateView.descriptor.bufferIdentity == stateIdentity
    check outputView.descriptor.bufferIdentity == outputIdentity
    check stateView.descriptor.hostCopyBytes == 0 and outputView.descriptor.hostCopyBytes == 0
    check state.cpu == scalar(2)
    check output.cpu == scalar(2)

  test "State namespaces, dtype and rank are validated without mutating caller state":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("increment").get
    let state = toCoreAI(scalar(0))
    let inputs = [CoreAIBinding(name: "delta", array: toCoreAI(scalar(1)))]
    let counter = CoreAIBinding(name: "counter", array: state)
    expect CoreAIDescriptorError: discard function.infer(inputs, states = [], outputViews = [])
    expect CoreAIDescriptorError:
      discard function.infer(inputs, states = [CoreAIBinding(name: "unknown", array: state)], outputViews = [])
    expect CoreAIDescriptorError:
      discard function.infer(inputs, states = [counter, counter], outputViews = [])
    expect CoreAIDescriptorError:
      discard function.infer(inputs, states = [CoreAIBinding(name: "counter",
        array: toCoreAI(newTensorWith[int32]([], 0'i32)))], outputViews = [])
    expect CoreAIDescriptorError:
      discard function.infer(inputs, states = [CoreAIBinding(name: "counter",
        array: toCoreAI([0'f32].toTensor))], outputViews = [])
    expect CoreAIDescriptorError:
      discard function.infer(inputs, states = [counter],
        outputViews = [CoreAIBinding(name: "unknown", array: toCoreAI(scalar(0)))])
    let identity = openCoreAIModel(coreAISourceAsset).loadFunction("identity").get
    expect CoreAIDescriptorError:
      discard identity.infer([CoreAIBinding(name: "x", array: toCoreAI([1'f32, 2].toTensor))],
        states = [counter], outputViews = [])
    check state.cpu(float32) == scalar(0)
    check function.infer(inputs, states = [counter], outputViews = [])[0].array.cpu(float32) == scalar(1)

  test "Pending owned and borrowed mutable storage rejects reads and overlapping calls":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("increment").get
    for borrowed in [false, true]:
      let queue = newMetalStream()
      let stream = newCoreAIStream(queue)
      let gate = newMetalEvent()
      var released = false
      defer:
        if not released: signalMetalEvent(gate, 1)
        metalSynchronize(queue)
      var nativeState, nativeOutput: MetalTensor[float32]
      var state, output: CoreAIArray
      if borrowed:
        nativeState = scalar(0).metal
        nativeOutput = scalar(0).metal
        state = coreAIView(nativeState)
        output = coreAIView(nativeOutput)
      else:
        state = toCoreAI(scalar(0))
        output = toCoreAI(scalar(0))
      let inputs = [CoreAIBinding(name: "delta", array: toCoreAI(scalar(1)))]
      let states = [CoreAIBinding(name: "counter", array: state)]
      let outputs = [CoreAIBinding(name: "y", array: output)]
      wait(queue, gate, 1)
      let job = function.encodeInference(inputs, states = states, outputViews = outputs, stream = stream)
      expect CoreAITimeoutError: discard job.wait(timeoutMs = 0)
      expect CoreAIBufferBusyError: discard state.cpu(float32)
      expect CoreAIBufferBusyError: discard output.cpu(float32)
      expect CoreAIBufferBusyError:
        discard function.submitInference(inputs, states = states, outputViews = [])
      expect CoreAIBufferBusyError:
        discard function.submitInference([CoreAIBinding(name: "delta", array: state)],
          states = [CoreAIBinding(name: "counter", array: toCoreAI(scalar(0)))], outputViews = [])
      if borrowed:
        expect MetalBufferBusyError: nativeState.setElement([], 99'f32)
        expect MetalBufferBusyError: discard nativeOutput.cpu
      signalMetalEvent(gate, 1)
      released = true
      check job.wait().len == 0
      discard awaitCurrentWorkCompleted(stream)
      check state.cpu(float32) == scalar(1)
      check output.cpu(float32) == scalar(1)
      check function.infer(inputs, states = states, outputViews = outputs).len == 0
      check state.cpu(float32) == scalar(2)
      check output.cpu(float32) == scalar(2)
