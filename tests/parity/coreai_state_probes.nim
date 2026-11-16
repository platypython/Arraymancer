# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes declared SDK state, persistent mutation and caller-owned output views.
import std/[json, options]
import ../../src/arraymancer
import ../apple/coreai_test_support
import ./[parity_support, coreai_model_probes]

proc scalar(value: float32): Tensor[float32] =
  ## Rank zero matches the exported counter's actual scalar descriptor.
  newTensorWith[float32]([], value)

proc stateDescriptor(descriptor: CoreAIArrayDescriptor): JsonNode =
  var shape = newJArray()
  for dimension in descriptor.shape: shape.add %dimension
  %*{"name": descriptor.name, "dtype": descriptor.dtype, "shape": shape,
    "dynamic_shape": descriptor.dynamicShape}

proc coreAIStateObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.coreai.stateful_execution", profile: "apple-coreai",
    execution: aeError, actual: newJObject())
  try:
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("increment").get
    let descriptor = function.descriptor
    var actualDescriptor = %*{"inputs": [], "outputs": [], "states": []}
    for item in descriptor.inputs: actualDescriptor["inputs"].add stateDescriptor(item)
    for item in descriptor.outputs: actualDescriptor["outputs"].add stateDescriptor(item)
    for item in descriptor.states: actualDescriptor["states"].add stateDescriptor(item)
    result.actual["descriptor"] = actualDescriptor
    result.differences.add compareObservations(%*{
      "inputs": [{"name": "delta", "dtype": "float32", "shape": [], "dynamic_shape": false}],
      "outputs": [{"name": "y", "dtype": "float32", "shape": [], "dynamic_shape": false}],
      "states": [{"name": "counter", "dtype": "float32", "shape": [], "dynamic_shape": false}]},
      actualDescriptor, 0, 0)
    let inputs = [CoreAIBinding(name: "delta", array: toCoreAI(scalar(1)))]
    let ownedState = toCoreAI(scalar(0))
    let states = [CoreAIBinding(name: "counter", array: ownedState)]
    let first = function.infer(inputs, states = states, outputViews = [])
    let second = function.infer(inputs, states = states, outputViews = [])
    var observed = newJArray()
    for call in [first, second]:
      var named = newJArray()
      for binding in call: named.add %*{"name": binding.name, "array": observeCoreAI(binding.array)}
      observed.add named
    result.actual["allocated_calls"] = observed
    result.differences.add compareObservations(%*[
      [{"name": "y", "array": {"shape": [], "dtype": "float32", "backend": "coreai", "values": [1.0]}}],
      [{"name": "y", "array": {"shape": [], "dtype": "float32", "backend": "coreai", "values": [2.0]}}]],
      observed, 0, 0)
    let ownedObserved = observeCoreAI(ownedState)
    result.actual["owned_state"] = ownedObserved
    result.differences.add compareObservations(%*{"shape": [], "dtype": "float32",
      "backend": "coreai", "values": [2.0]}, ownedObserved, 0, 0)
    var missingRejected = false
    try: discard function.infer(inputs, states = [], outputViews = [])
    except CoreAIDescriptorError as error:
      missingRejected = true
      result.actual["missing_state_diagnostic"] = %*{"category": "descriptor",
        "native_code": error.nativeCode, "message": error.msg}
    let nativeState = scalar(0).metal
    let nativeOutput = scalar(-1).metal
    let stateIdentity = originalMetalBufferIdentity(nativeState)
    let outputIdentity = originalMetalBufferIdentity(nativeOutput)
    let stateView = coreAIView(nativeState)
    let outputView = coreAIView(nativeOutput)
    let mutableStates = [CoreAIBinding(name: "counter", array: stateView)]
    let outputViews = [CoreAIBinding(name: "y", array: outputView)]
    let firstProvided = function.infer(inputs, states = mutableStates, outputViews = outputViews)
    let stream = newCoreAIStream(newMetalStream())
    let job = function.encodeInference(inputs, states = mutableStates, outputViews = outputViews, stream = stream)
    let secondProvided = job.wait()
    let boundary = awaitCurrentWorkCompleted(stream)
    var allocatedNames: seq[string]
    for binding in firstProvided: allocatedNames.add binding.name
    for binding in secondProvided: allocatedNames.add binding.name
    let facts = %*{"missing_state_rejected": missingRejected, "allocated_output_names": allocatedNames,
      "same_state_buffer": stateView.descriptor.bufferIdentity == stateIdentity,
      "same_output_buffer": outputView.descriptor.bufferIdentity == outputIdentity,
      "state_bridge_copy_bytes": stateView.descriptor.hostCopyBytes,
      "output_bridge_copy_bytes": outputView.descriptor.hostCopyBytes,
      "state": nativeState.cpu.atContiguousIndex(0), "output": nativeOutput.cpu.atContiguousIndex(0),
      "sdk_stream_completed": boundary.sdkStreamCompleted, "metal_queue_completed": boundary.metalQueueCompleted}
    result.actual["provided_views"] = facts
    result.differences.add compareObservations(%*{"missing_state_rejected": true,
      "allocated_output_names": [], "same_state_buffer": true, "same_output_buffer": true,
      "state_bridge_copy_bytes": 0, "output_bridge_copy_bytes": 0, "state": 2.0, "output": 2.0,
      "sdk_stream_completed": true, "metal_queue_completed": true}, facts, 0, 0)
    result.evidence = coreAIEvidence(job)
    result.evidence["sdk_internal_copy_bytes"] = newJNull()
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CoreAIError as error:
    let category = case error.nativeCode
      of 102: "descriptor"
      of 104: "timeout"
      of 105: "buffer_busy"
      else: "native_error"
    result.error = %*{"category": category, "native_code": error.nativeCode, "message": error.msg}
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
