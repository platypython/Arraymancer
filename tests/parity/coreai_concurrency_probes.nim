# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes independent native jobs and deterministic same-state exclusion.
import std/[json, options]
import ../../src/arraymancer
import ../apple/coreai_test_support
import ./[parity_support, coreai_model_probes]

proc coreAIConcurrencyObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.coreai.concurrency", profile: "apple-coreai", execution: aeError, actual: newJObject())
  try:
    let model = openCoreAIModel(coreAISourceAsset)
    let identity = model.loadFunction("identity").get
    let firstBuffer = [0'f32, 0].toTensor.metal
    let secondBuffer = [0'f32, 0].toTensor.metal
    let firstView = coreAIView(firstBuffer)
    let secondView = coreAIView(secondBuffer)
    let first = identity.submitInference([CoreAIBinding(name: "x", array: toCoreAI([1'f32, 2].toTensor))],
      states = [], outputViews = [CoreAIBinding(name: "y", array: firstView)])
    let second = identity.submitInference([CoreAIBinding(name: "x", array: toCoreAI([3'f32, 4].toTensor))],
      states = [], outputViews = [CoreAIBinding(name: "y", array: secondView)])
    GC_fullCollect()
    discard first.wait()
    discard second.wait()
    let arrays = %*[observeCoreAI(firstView), observeCoreAI(secondView)]
    result.actual["identity_outputs"] = arrays
    result.differences.add compareObservations(%*[
      {"shape": [2], "dtype": "float32", "backend": "coreai", "values": [1.0, 2.0]},
      {"shape": [2], "dtype": "float32", "backend": "coreai", "values": [3.0, 4.0]}], arrays, 0, 0)
    let function = model.loadFunction("increment").get
    let inputs = [CoreAIBinding(name: "delta", array: toCoreAI(newTensorWith[float32]([], 1'f32)))]
    let state = toCoreAI(newTensorWith[float32]([], 0'f32))
    let separate = toCoreAI(newTensorWith[float32]([], 0'f32))
    let queue = newMetalStream()
    let stream = newCoreAIStream(queue)
    let gate = newMetalEvent()
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      metalSynchronize(queue)
    wait(queue, gate, 1)
    let states = [CoreAIBinding(name: "counter", array: state)]
    let pending = function.encodeInference(inputs, states = states, outputViews = [], stream = stream)
    var rejected, stillPending: bool
    try: discard function.submitInference(inputs, states = states, outputViews = [])
    except CoreAIBufferBusyError as error:
      rejected = true
      result.actual["busy_diagnostic"] = %*{"category": "buffer_busy", "native_code": error.nativeCode, "message": error.msg}
    let independent = function.submitInference(inputs, states = [CoreAIBinding(name: "counter", array: separate)], outputViews = [])
    let otherOutputs = independent.wait()
    try: discard pending.wait(timeoutMs = 0)
    except CoreAITimeoutError: stillPending = true
    signalMetalEvent(gate, 1)
    released = true
    let pendingOutputs = pending.wait()
    discard awaitCurrentWorkCompleted(stream)
    let reused = function.infer(inputs, states = states, outputViews = [])
    let values = %*[otherOutputs[0].array.cpu(float32).atContiguousIndex(0),
      pendingOutputs[0].array.cpu(float32).atContiguousIndex(0), reused[0].array.cpu(float32).atContiguousIndex(0)]
    result.actual["state_outputs"] = values
    result.differences.add compareObservations(%*[1.0, 1.0, 2.0], values, 0, 0)
    let facts = %*{"distinct_output_buffers": firstView.descriptor.bufferIdentity != secondView.descriptor.bufferIdentity,
      "distinct_identity_jobs": first.coreAIJobEvidence.invocationId != second.coreAIJobEvidence.invocationId,
      "same_state_rejected": rejected, "independent_call_completed_while_first_pending": stillPending}
    result.actual["invariants"] = facts
    result.differences.add compareObservations(%*{"distinct_output_buffers": true, "distinct_identity_jobs": true,
      "same_state_rejected": true, "independent_call_completed_while_first_pending": true}, facts, 0, 0)
    result.evidence = coreAIEvidence(pending)
    result.evidence["invocations"] = %*[coreAIEvidence(first), coreAIEvidence(second), coreAIEvidence(independent)]
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
