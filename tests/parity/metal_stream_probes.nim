# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures actual event-blocked queues before releasing a native producer/consumer chain.
import std/json
import ../../src/arraymancer
import ./[parity_support, metal_math_probes]

proc metalStreamObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.metal.queue_lifetime", profile: "apple-metal", execution: aeError,
    actual: newJObject())
  try:
    let previous = currentMetalStream()
    defer: setMetalStream(previous)
    let producer = newMetalStream()
    let consumer = newMetalStream()
    let defaultStream = defaultMetalStream()
    let blocker = newMetalEvent()
    let ready = newMetalEvent()
    var released = false
    defer:
      if not released: signalMetalEvent(blocker, 1)
      metalSynchronize(producer)
      metalSynchronize(consumer)
    setMetalStream(producer)
    let setMatches = currentMetalStream().identity == producer.identity
    try:
      withMetalStream(consumer):
        raise newException(ValueError, "verify exception-safe stream restoration")
    except ValueError: discard
    let restored = currentMetalStream().identity == producer.identity
    setMetalStream(previous)
    wait(producer, blocker, 1)
    var alias: MetalTensor[float32]
    var ownerShared = false
    block:
      withMetalStream(producer):
        let original = [1'f32, 2].toTensor.metal
        let generated = original + 1'f32
        alias = generated.metalView([2].toMetadata, [1].toMetadata, 0)
        ownerShared = alias.storage == generated.storage
        record(ready, producer)
    var output: MetalTensor[float32]
    withMetalStream(consumer):
      wait(consumer, ready)
      output = alias * 2'f32
    metalSynchronize(defaultStream)
    var producerEvidence, consumerEvidence: AMEvidence
    checkMetal metalApi().evidence(alias.storage.owner.handle, addr producerEvidence)
    checkMetal metalApi().evidence(output.storage.owner.handle, addr consumerEvidence)
    let pending = %*{"producer_incomplete": not metalStreamIsComplete(producer),
      "consumer_incomplete": not metalStreamIsComplete(consumer),
      "producer_kernel_pending": producerEvidence.completed == 0,
      "consumer_kernel_pending": consumerEvidence.completed == 0}
    result.actual["pending"] = pending
    signalMetalEvent(blocker, 1)
    released = true
    metalSynchronize(consumer)
    metalSynchronize(producer)
    let outputObservation = observeMetal(output)
    result.actual["result"] = outputObservation
    let invariants = %*{"set_matches": setMatches, "scope_restored": restored,
      "default_stable": defaultStream.identity == defaultMetalStream().identity,
      "distinct_queues": producer.identity != consumer.identity and producer.identity != defaultStream.identity,
      "alias_shared_owner": ownerShared, "producer_complete": metalStreamIsComplete(producer),
      "consumer_complete": metalStreamIsComplete(consumer)}
    result.actual["invariants"] = invariants
    result.differences = compareObservations(%*{"producer_incomplete": true,
      "consumer_incomplete": true, "producer_kernel_pending": true, "consumer_kernel_pending": true}, pending, 0, 0)
    result.differences.add compareObservations(%*{"shape": [2], "numel": 2,
      "dtype": "float32", "device": "metal", "values": [4.0, 6.0]}, outputObservation, 0, 0)
    result.differences.add compareObservations(%*{"set_matches": true, "scope_restored": true,
      "default_stable": true, "distinct_queues": true, "alias_shared_owner": true,
      "producer_complete": true, "consumer_complete": true}, invariants, 0, 0)
    result.evidence = metalEvidence(output)
    result.evidence["producer_queue_id"] = %($producer.identity)
    result.evidence["consumer_queue_id"] = %($consumer.identity)
    result.evidence["default_queue_id"] = %($defaultStream.identity)
    result.evidence["producer_invocation"] = metalEvidence(alias)
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
