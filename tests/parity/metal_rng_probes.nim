# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes Philox GPU draws and queue-independent reservation/replay semantics.
import std/json
import ../../src/arraymancer
import ./[parity_support, metal_math_probes]

proc rngStateJson(state: MetalRngState): JsonNode =
  %*{"version": state.version, "seed": $state.seed, "counter": state.counter}

proc metalRngObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.metal.rng", profile: "apple-metal", execution: aeError,
    actual: newJObject())
  try:
    let generator = newMetalGenerator(0)
    let saved = getRngState(generator)
    let first = randMetal([4].toMetadata, generator)
    let advanced = getRngState(generator)
    setRngState(generator, saved)
    let second = randMetal([4].toMetadata, generator)
    let firstObserved = observeMetal(first)
    let secondObserved = observeMetal(second)
    var knownValues = newJArray()
    # Random123 9545ff6413f258be2f04c1d319d99aaef7521150 tests/kat_vectors, Philox4x32-10 zero key/counter.
    for word in [0x6627e8d5'u32, 0xe169c58d'u32, 0xbc57ac4c'u32, 0x9b00dbd8'u32]:
      knownValues.add %(float32(word shr 8) * (1'f32 / 16777216'f32))
    result.differences = compareObservations(%*{"shape": [4], "numel": 4,
      "dtype": "float32", "device": "metal", "values": knownValues}, firstObserved, 0, 0)
    result.differences.add compareObservations(firstObserved, secondObserved, 0, 0)
    let replay = %*{"distinct_buffers": first.storage != second.storage,
      "reservation_advanced": advanced.counter == [1'u32, 0, 0, 0],
      "native_device": first.storage.device.registryId == generator.device.registryId}
    manualSeed(generator, 37)
    let beforePending = getRngState(generator)
    let stream = newMetalStream(generator.device)
    let gate = newMetalEvent(generator.device)
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      metalSynchronize(stream)
    wait(stream, gate, 1)
    var pending: MetalTensor[float32]
    withMetalStream(stream):
      pending = randMetal([5].toMetadata, generator)
      discard randMetal([4].toMetadata, generator)
    let reserved = getRngState(generator)
    var pendingEvidence: AMEvidence
    checkMetal metalApi().evidence(pending.storage.owner.handle, addr pendingEvidence)
    let wasPending = not metalStreamIsComplete(stream) and pendingEvidence.completed == 0
    setRngState(generator, beforePending)
    let repeated = randMetal([5].toMetadata, generator)
    let repeatedValues = repeated.cpu
    let independent = not metalStreamIsComplete(stream)
    signalMetalEvent(gate, 1)
    released = true
    metalSynchronize(stream)
    let pendingFacts = %*{"reserved_before_completion": wasPending and reserved.counter == [3'u32, 0, 0, 0],
      "independent_replay_completes": independent, "pending_seed_snapshot_preserved": pending.cpu == repeatedValues}
    var scalarShape: Metadata
    manualSeed(generator, 11)
    let scalar = randMetal(scalarShape, generator)
    let afterScalar = getRngState(generator)
    let empty = randMetal([2, 0, 3].toMetadata, generator)
    let emptyState = getRngState(generator)
    let metadata = %*{"scalar_rank_zero": scalar.rank == 0 and scalar.size == 1,
      "scalar_reserves_one": afterScalar.counter == [1'u32, 0, 0, 0],
      "empty_shape_preserved": empty.shape == [2, 0, 3].toMetadata and empty.size == 0,
      "empty_reserves_zero": emptyState == afterScalar, "empty_host_size_zero": empty.cpu.size == 0}
    var invalid = afterScalar
    invalid.version = 2
    var invalidRejected = false
    try: setRngState(generator, invalid)
    except MetalRngStateError: invalidRejected = true
    let invalidUnchanged = getRngState(generator) == afterScalar
    var exhausted = afterScalar
    exhausted.counter = [high(uint32) - 1, high(uint32), high(uint32), high(uint32)]
    setRngState(generator, exhausted)
    var overflowRejected = false
    try: discard randMetal([8].toMetadata, generator)
    except MetalRngStateError: overflowRejected = true
    let errors = %*{"invalid_version_rejected": invalidRejected,
      "invalid_state_unchanged": invalidUnchanged, "overflow_rejected": overflowRejected,
      "overflow_state_unchanged": getRngState(generator) == exhausted}
    let defaultGenerator = defaultMetalGenerator(generator.device)
    let defaultState = getRngState(defaultGenerator)
    defer: setRngState(defaultGenerator, defaultState)
    let defaultAgain = defaultMetalGenerator(generator.device)
    let entropySeed = seed(defaultGenerator)
    let entropyState = getRngState(defaultAgain)
    let entropy = %*{"default_shared": defaultGenerator.owner.handle == defaultAgain.owner.handle,
      "entropy_seed_returned": entropyState.seed == entropySeed,
      "entropy_resets_counter": entropyState.counter == [0'u32, 0, 0, 0]}
    result.actual = %*{"first": firstObserved, "redraw": secondObserved, "replay": replay,
      "pending": pendingFacts, "metadata": metadata, "errors": errors, "entropy": entropy}
    result.differences.add compareObservations(%*{"distinct_buffers": true,
      "reservation_advanced": true, "native_device": true}, replay, 0, 0)
    result.differences.add compareObservations(%*{"reserved_before_completion": true,
      "independent_replay_completes": true, "pending_seed_snapshot_preserved": true}, pendingFacts, 0, 0)
    result.differences.add compareObservations(%*{"scalar_rank_zero": true, "scalar_reserves_one": true,
      "empty_shape_preserved": true, "empty_reserves_zero": true, "empty_host_size_zero": true}, metadata, 0, 0)
    result.differences.add compareObservations(%*{"invalid_version_rejected": true,
      "invalid_state_unchanged": true, "overflow_rejected": true, "overflow_state_unchanged": true}, errors, 0, 0)
    result.differences.add compareObservations(%*{"default_shared": true,
      "entropy_seed_returned": true, "entropy_resets_counter": true}, entropy, 0, 0)
    metalSynchronize(generator.device)
    result.evidence = metalEvidence(second)
    result.evidence["first_invocation"] = metalEvidence(first)
    result.evidence["pending_invocation"] = metalEvidence(pending)
    result.evidence["rng_states"] = %*{"saved": rngStateJson(saved),
      "after_first": rngStateJson(advanced), "pending_reserved": rngStateJson(reserved),
      "after_scalar": rngStateJson(afterScalar), "after_empty": rngStateJson(emptyState),
      "entropy": rngStateJson(entropyState)}
    result.evidence["cross_backend_bitwise_equivalence"] = %"not_claimed"
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": (if error of MetalRngStateError: "rng_state" else: "native_error"),
      "message": error.msg}
