# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes producer/inference/consumer ordering and separate SDK/Metal completion boundaries.
import std/[json, options]
import ../../src/arraymancer
import ../apple/coreai_test_support
import ./[parity_support, coreai_model_probes, metal_math_probes]

proc coreAIStreamObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.coreai.stream_lifetime", profile: "apple-coreai",
    execution: aeError, actual: newJObject())
  try:
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    let queue = newMetalStream()
    let stream = newCoreAIStream(queue)
    let unrelated = newMetalStream()
    let gate = newMetalEvent()
    defer:
      signalMetalEvent(gate, 1)
      metalSynchronize(unrelated)
    wait(unrelated, gate, 1)
    var inputView, outputView: CoreAIArray
    var output: MetalTensor[float32]
    var identity: uint64
    var job: CoreAIJob
    block:
      let source = [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal
      var produced: MetalTensor[float32]
      withMetalStream(queue): produced = source + 1'f32
      output = newTensorWith[float32]([2, 3], 0'f32).metal
      identity = originalMetalBufferIdentity(output)
      inputView = coreAIView(produced)
      outputView = coreAIView(output)
      job = function.encodeInference([CoreAIBinding(name: "x", array: inputView)],
        outputViews = [CoreAIBinding(name: "y", array: outputView)], stream = stream)
    GC_fullCollect()
    let outputs = job.wait()
    var allocatedNames: seq[string]
    for binding in outputs: allocatedNames.add binding.name
    let inference = job.coreAIJobEvidence
    var consumer: MetalTensor[float32]
    withMetalStream(queue): consumer = output * 2'f32
    let completion = awaitCurrentWorkCompleted(stream)
    let facts = %*{"allocated_output_names": allocatedNames,
      "same_output_buffer": outputView.descriptor.bufferIdentity == identity,
      "same_inference_queue": inference.streamIdentity == queue.identity,
      "same_completion_queue": completion.streamIdentity == queue.identity,
      "inference_completed": inference.completed,
      "inference_sdk_stream_completed": inference.sdkStreamCompleted,
      "inference_metal_queue_completed": inference.metalQueueCompleted,
      "boundary_completed": completion.completed,
      "boundary_sdk_stream_completed": completion.sdkStreamCompleted,
      "boundary_metal_queue_completed": completion.metalQueueCompleted,
      "metal_queue_complete": metalStreamIsComplete(queue),
      "unrelated_queue_pending": not metalStreamIsComplete(unrelated),
      "input_bridge_copy_bytes": inputView.descriptor.hostCopyBytes,
      "output_bridge_copy_bytes": outputView.descriptor.hostCopyBytes,
      "inference_bridge_copy_bytes": inference.hostCopyBytes,
      "inference_materialization_bytes": inference.materializationBytes}
    result.actual["invariants"] = facts
    result.actual["output_buffer_identity"] = %($outputView.descriptor.bufferIdentity)
    result.actual["original_output_buffer_identity"] = %($identity)
    result.actual["metal_queue_identity"] = %queue.identity
    result.differences.add compareObservations(%*{"allocated_output_names": [],
      "same_output_buffer": true, "same_inference_queue": true, "same_completion_queue": true,
      "inference_completed": true, "inference_sdk_stream_completed": true,
      "inference_metal_queue_completed": false, "boundary_completed": true,
      "boundary_sdk_stream_completed": true, "boundary_metal_queue_completed": true,
      "metal_queue_complete": true, "unrelated_queue_pending": true,
      "input_bridge_copy_bytes": 0, "output_bridge_copy_bytes": 0,
      "inference_bridge_copy_bytes": 0, "inference_materialization_bytes": 0}, facts, 0, 0)
    let observed = observeMetal(consumer)
    result.actual["consumer"] = observed
    result.differences.add compareObservations(%*{"shape": [2, 3], "numel": 6,
      "dtype": "float32", "device": "metal", "values": [4.0, 6.0, 8.0, 10.0, 12.0, 14.0]},
      observed, 0, 0)
    result.evidence = coreAIEvidence(job)
    result.evidence["stream_boundary"] = %*{"invocation_id": completion.invocationId,
      "completed": completion.completed, "stream_identity": completion.streamIdentity,
      "sdk_stream_completed": completion.sdkStreamCompleted,
      "metal_queue_completed": completion.metalQueueCompleted}
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
