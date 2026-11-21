# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises shared-owner accounting, pending leases, limits and real GPU capture.
import std/[unittest, os, monotimes]
import ../../src/arraymancer

proc collectScopedOwners() {.noinline.} =
  ## Refc conservatively scans dead stack slots left by optimized fixture calls.
  when defined(gcDestructors):
    GC_fullCollect()
  else:
    var retired {.noinit, volatile.}: array[1024, pointer]
    for slot in 0..<retired.len:
      retired[slot] = nil
    GC_fullCollect()
    discard retired[0] # Keep the cleared frame alive through collection.

proc scopedAllocation(device: MetalDevice): tuple[live, aliased: MetalMemoryStats, shares: bool] {.noinline.} =
  let tensor = [1'f32, 2, 3, 4, 5, 6].toTensor.reshape(2, 3).metal(device)
  result.live = metalMemoryStats(device)
  let alias = tensor.transpose
  result.aliased = metalMemoryStats(device)
  result.shares = alias.storage == tensor.storage
  metalEmptyCache(device)
  doAssert alias.cpu == [[1'f32, 4], [2'f32, 5], [3'f32, 6]].toTensor

proc scopedArithmeticEvidence(input: MetalTensor[float32]): AMEvidence {.noinline.} =
  ## Retire temporary owners before the caller checks cached storage reuse.
  let produced = input + 2'f32
  doAssert produced.cpu == [3'f32, 4, 5, 6].toTensor
  checkMetal metalApi().evidence(produced.storage.owner.handle, addr result)
  doAssert result.kind == 2 and result.completed == 1

proc scopedReuse(device: MetalDevice):
    tuple[current: MetalMemoryStats, sameDevice: bool] {.noinline.} =
  ## Return accounting rather than an owner retained by refc's module roots.
  let reused = newMetalTensorUninit[float32]([2, 3].toMetadata, device)
  result.current = metalMemoryStats(device)
  result.sameDevice = reused.storage.device.registryId == device.registryId

proc scopedArithmeticReuse(device: MetalDevice):
    tuple[produced, fresh: AMEvidence, cachedBefore, cachedAfter: uint64] {.noinline.} =
  ## Keep both allocation owners inside this lifetime boundary.
  let input = [1'f32, 2, 3, 4].toTensor.metal(device)
  result.produced = scopedArithmeticEvidence(input)
  collectScopedOwners()
  result.cachedBefore = metalMemoryStats(device).cachedBytes
  let reused = newMetalTensorUninit[float32]([4].toMetadata, device)
  result.cachedAfter = metalMemoryStats(device).cachedBytes
  checkMetal metalApi().evidence(reused.storage.owner.handle, addr result.fresh)
  doAssert input.cpu == [1'f32, 2, 3, 4].toTensor

proc scopedPendingAlias(device: MetalDevice, stream: MetalStream,
                        alias: var MetalTensor[float32]) {.noinline.} =
  ## Keep only the surviving view while queued work retains its dropped input.
  withMetalStream(stream):
    let input = [1'f32, 2].toTensor.metal(device)
    let output = input + 1'f32
    alias = output.metalView([2].toMetadata, [1].toMetadata, 0)

suite "Metal allocation accounting and capture":
  test "Shared storage is counted once and only unused cache is released":
    let device = metalDevice()
    metalSynchronize(device)
    collectScopedOwners()
    metalEmptyCache(device)
    let baseline = metalMemoryStats(device)
    let allocated = scopedAllocation(device)
    check allocated.live.liveBytes == baseline.liveBytes + 24
    check allocated.aliased.liveBytes == allocated.live.liveBytes
    check allocated.shares
    metalSynchronize(device)
    collectScopedOwners()
    let released = metalMemoryStats(device)
    check released.liveBytes == baseline.liveBytes
    check released.cachedBytes >= 24
    let reused = scopedReuse(device)
    check reused.current.liveBytes == baseline.liveBytes + 24
    check reused.current.cachedBytes == released.cachedBytes - 24
    check reused.current.reservedBytes == released.reservedBytes
    check reused.sameDevice
    collectScopedOwners()
    metalEmptyCache(device)
    check metalMemoryStats(device).cachedBytes == 0
    check metalCurrentAllocatedMemory(device) == baseline.liveBytes
    check metalRecommendedMaxMemory(device) > 0

  test "Reused arithmetic storage starts with fresh allocation evidence":
    let device = metalDevice()
    metalSynchronize(device)
    collectScopedOwners()
    metalEmptyCache(device)
    let baseline = metalMemoryStats(device)
    let evidence = scopedArithmeticReuse(device)
    check evidence.cachedAfter == evidence.cachedBefore - 16
    check evidence.fresh.invocationId != evidence.produced.invocationId
    check evidence.fresh.kind == 1 and evidence.fresh.completed == 1
    check evidence.fresh.cpuFallbackCount == 0
    collectScopedOwners()
    check metalCurrentAllocatedMemory(device) == baseline.liveBytes

  test "Cache clearing cannot recycle a dropped input held by pending work":
    let device = metalDevice()
    metalSynchronize(device)
    collectScopedOwners()
    metalEmptyCache(device)
    let baseline = metalMemoryStats(device)
    let stream = newMetalStream(device)
    let gate = newMetalEvent(device)
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      metalSynchronize(stream)
    wait(stream, gate, 1)
    var alias: MetalTensor[float32]
    scopedPendingAlias(device, stream, alias)
    collectScopedOwners()
    let pending = metalMemoryStats(device)
    check pending.liveBytes == baseline.liveBytes + 8
    check pending.inFlightBytes == baseline.inFlightBytes + 16
    check not metalStreamIsComplete(stream)
    metalEmptyCache(device)
    let cleared = metalMemoryStats(device)
    check cleared.cachedBytes == 0
    check cleared.inFlightBytes == pending.inFlightBytes
    check not metalStreamIsComplete(stream)
    signalMetalEvent(gate, 1)
    released = true
    metalSynchronize(stream)
    check metalMemoryStats(device).inFlightBytes == baseline.inFlightBytes
    check alias.cpu == [2'f32, 3].toTensor
    alias = default(MetalTensor[float32])
    collectScopedOwners()
    check metalCurrentAllocatedMemory(device) == baseline.liveBytes

  test "Fractions reject invalid values and tiny limits without replacing live data":
    let device = metalDevice()
    let previous = metalMemoryFraction(device)
    defer: setMetalMemoryFraction(device, previous)
    for invalid in [-0.1, 2.1, NaN, Inf, NegInf]:
      expect ValueError: setMetalMemoryFraction(device, invalid)
    setMetalMemoryFraction(device, 0)
    check metalMemoryFraction(device) == 0
    let live = [3'f32, 4].toTensor.metal(device)
    setMetalMemoryFraction(device, 1.0 / float64(metalRecommendedMaxMemory(device)))
    expect MetalOutOfMemoryError:
      discard newMetalTensorUninit[float32]([1024].toMetadata, device)
    check live.cpu == [3'f32, 4].toTensor
    setMetalMemoryFraction(device, 2)
    check metalMemoryFraction(device) == 2

  test "Capture contains completed arithmetic and measured copies":
    let device = metalDevice()
    createDir("build/apple")
    let path = absolutePath("build/apple/memory-" & $getCurrentProcessId() & "-" &
      $getMonoTime().ticks & ".gputrace")
    let before = metalMemoryStats(device)
    let capture = startMetalCapture(device, path)
    var stopped = false
    defer:
      if not stopped: stopMetalCapture(capture)
    let input = [1'f32, 2, 3, 4].toTensor.metal(device)
    let output = (input * 2'f32).clone
    metalSynchronize(device)
    check output.cpu == [2'f32, 4, 6, 8].toTensor
    stopMetalCapture(capture)
    stopped = true
    check fileExists(path) or dirExists(path)
    let after = metalMemoryStats(device)
    check after.hostCopyBytes >= before.hostCopyBytes + 32
    check after.materializationBytes >= before.materializationBytes + 16
    var evidence: AMEvidence
    checkMetal metalApi().evidence(output.storage.owner.handle, addr evidence)
    check evidence.kind == 2 and evidence.completed == 1
    check evidence.cpuFallbackCount == 0
    checkpoint "GPU capture: " & path
