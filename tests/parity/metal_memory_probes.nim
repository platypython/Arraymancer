# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Measures native shared-owner accounting, pending leases, limits and GPU capture.
import std/[json, os, monotimes]
import ../../src/arraymancer
import ./[parity_support, metal_math_probes]

proc memoryJson(stats: MetalMemoryStats): JsonNode =
  %*{"live_bytes": stats.liveBytes, "cached_bytes": stats.cachedBytes,
    "in_flight_bytes": stats.inFlightBytes, "reserved_bytes": stats.reservedBytes,
    "driver_bytes": stats.driverBytes, "recommended_bytes": stats.recommendedBytes,
    "host_copy_bytes": stats.hostCopyBytes, "materialization_bytes": stats.materializationBytes}

proc allocationScope(device: MetalDevice): tuple[allocated, aliased: MetalMemoryStats, shared, survived: bool] =
  let tensor = [1'f32, 2, 3, 4, 5, 6].toTensor.reshape(2, 3).metal(device)
  result.allocated = metalMemoryStats(device)
  let alias = tensor.transpose
  result.aliased = metalMemoryStats(device)
  result.shared = alias.storage == tensor.storage
  metalEmptyCache(device)
  result.survived = alias.cpu == [[1'f32, 4], [2'f32, 5], [3'f32, 6]].toTensor

proc metalMemoryObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.metal.memory_profiling", profile: "apple-metal",
    execution: aeError, actual: newJObject())
  try:
    let device = metalDevice()
    let previousFraction = metalMemoryFraction(device)
    defer: setMetalMemoryFraction(device, previousFraction)
    setMetalMemoryFraction(device, 0)
    metalSynchronize(device)
    GC_fullCollect()
    metalEmptyCache(device)
    let baseline = metalMemoryStats(device)
    let scoped = allocationScope(device)
    metalSynchronize(device)
    GC_fullCollect()
    let released = metalMemoryStats(device)
    var cacheReused = false
    block:
      let reused = newMetalTensorUninit[float32]([2, 3].toMetadata, device)
      let measured = metalMemoryStats(device)
      cacheReused = measured.liveBytes == baseline.liveBytes + 24 and
        measured.cachedBytes + 24 == released.cachedBytes and
        measured.reservedBytes == released.reservedBytes and
        reused.storage.device.registryId == device.registryId
    GC_fullCollect()
    metalEmptyCache(device)
    let emptied = metalMemoryStats(device)
    let lifecycle = %*{"allocation_counted": scoped.allocated.liveBytes == baseline.liveBytes + 24,
      "alias_counted_once": scoped.aliased.liveBytes == scoped.allocated.liveBytes and scoped.shared,
      "live_alias_survives_clear": scoped.survived,
      "release_restores_live": released.liveBytes == baseline.liveBytes,
      "release_enters_cache": released.cachedBytes >= 24,
      "cache_reused": cacheReused, "cache_cleared": emptied.cachedBytes == 0,
      "recommended_positive": baseline.recommendedBytes > 0}
    let stream = newMetalStream(device)
    let gate = newMetalEvent(device)
    var signaled = false
    defer:
      if not signaled: signalMetalEvent(gate, 1)
      metalSynchronize(stream)
    wait(stream, gate, 1)
    var survivor: MetalTensor[float32]
    block:
      withMetalStream(stream):
        let input = [1'f32, 2].toTensor.metal(device)
        let output = input + 1'f32
        survivor = output.metalView([2].toMetadata, [1].toMetadata, 0)
    GC_fullCollect()
    let pending = metalMemoryStats(device)
    metalEmptyCache(device)
    let pendingCleared = metalMemoryStats(device)
    let remainedPending = not metalStreamIsComplete(stream)
    signalMetalEvent(gate, 1)
    signaled = true
    metalSynchronize(stream)
    let completed = metalMemoryStats(device)
    let survivedPending = survivor.cpu == [2'f32, 3].toTensor
    survivor = default(MetalTensor[float32])
    GC_fullCollect()
    let pendingFacts = %*{"unowned_input_not_live": pending.liveBytes == baseline.liveBytes + 8,
      "unique_pending_leases": pending.inFlightBytes == baseline.inFlightBytes + 16,
      "clear_preserves_pending": pendingCleared.inFlightBytes == pending.inFlightBytes and remainedPending,
      "leases_complete": completed.inFlightBytes == baseline.inFlightBytes,
      "pending_values_preserved": survivedPending,
      "last_alias_release_restores_live": metalCurrentAllocatedMemory(device) == baseline.liveBytes}
    var invalidRejected = 0
    for invalid in [-0.1, 2.1, NaN, Inf, NegInf]:
      try: setMetalMemoryFraction(device, invalid)
      except ValueError: inc invalidRejected
    setMetalMemoryFraction(device, 2)
    let maximumAccepted = metalMemoryFraction(device) == 2
    setMetalMemoryFraction(device, 1.0 / float64(metalRecommendedMaxMemory(device)))
    var tinyRejected = false
    try: discard newMetalTensorUninit[float32]([1024].toMetadata, device)
    except MetalOutOfMemoryError: tinyRejected = true
    setMetalMemoryFraction(device, 0)
    let unlimitedAccepted = metalMemoryFraction(device) == 0
    let limits = %*{"invalid_rejected": invalidRejected, "tiny_limit_rejected": tinyRejected,
      "maximum_accepted": maximumAccepted, "unlimited_accepted": unlimitedAccepted}
    createDir("build/apple")
    let path = absolutePath("build/apple/parity-memory-" & $getCurrentProcessId() & "-" &
      $getMonoTime().ticks & ".gputrace")
    let capture = startMetalCapture(device, path)
    var stopped = false
    defer:
      if not stopped: stopMetalCapture(capture)
    let input = [1'f32, 2, 3, 4].toTensor.metal(device)
    let doubled = input * 2'f32
    let output = doubled.clone
    metalSynchronize(device)
    let observed = observeMetal(output)
    stopMetalCapture(capture)
    stopped = true
    let captured = metalMemoryStats(device)
    let profiling = %*{"trace_created": fileExists(path) or dirExists(path),
      "host_copies_measured": captured.hostCopyBytes >= baseline.hostCopyBytes + 32,
      "materialization_measured": captured.materializationBytes >= baseline.materializationBytes + 16}
    result.actual = %*{"lifecycle": lifecycle, "pending": pendingFacts, "limits": limits,
      "profiling": profiling, "result": observed}
    result.differences = compareObservations(%*{"allocation_counted": true,
      "alias_counted_once": true, "live_alias_survives_clear": true,
      "release_restores_live": true, "release_enters_cache": true,
      "cache_reused": true, "cache_cleared": true, "recommended_positive": true}, lifecycle, 0, 0)
    result.differences.add compareObservations(%*{"unowned_input_not_live": true,
      "unique_pending_leases": true, "clear_preserves_pending": true, "leases_complete": true,
      "pending_values_preserved": true, "last_alias_release_restores_live": true}, pendingFacts, 0, 0)
    result.differences.add compareObservations(%*{"invalid_rejected": 5, "tiny_limit_rejected": true,
      "maximum_accepted": true, "unlimited_accepted": true}, limits, 0, 0)
    result.differences.add compareObservations(%*{"trace_created": true,
      "host_copies_measured": true, "materialization_measured": true}, profiling, 0, 0)
    result.differences.add compareObservations(%*{"shape": [4], "numel": 4,
      "dtype": "float32", "device": "metal", "values": [2.0, 4.0, 6.0, 8.0]}, observed, 0, 0)
    result.evidence = metalEvidence(output)
    result.evidence["memory_measurements"] = %*{"baseline": memoryJson(baseline),
      "allocated": memoryJson(scoped.allocated), "aliased": memoryJson(scoped.aliased),
      "released": memoryJson(released), "cache_emptied": memoryJson(emptied),
      "pending": memoryJson(pending), "pending_after_cache_clear": memoryJson(pendingCleared),
      "completed": memoryJson(completed), "captured": memoryJson(captured)}
    result.evidence["capture_path"] = %path
    result.evidence["arithmetic_invocation"] = metalEvidence(doubled)
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
