# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises shared-owner accounting, pending leases, limits and real GPU capture.
import std/[unittest, os, monotimes]
import ../../src/arraymancer

proc scopedAllocation(device: MetalDevice): tuple[live, aliased: MetalMemoryStats, shares: bool] =
  let tensor = [1'f32, 2, 3, 4, 5, 6].toTensor.reshape(2, 3).metal(device)
  result.live = metalMemoryStats(device)
  let alias = tensor.transpose
  result.aliased = metalMemoryStats(device)
  result.shares = alias.storage == tensor.storage
  metalEmptyCache(device)
  doAssert alias.cpu == [[1'f32, 4], [2'f32, 5], [3'f32, 6]].toTensor

suite "Metal allocation accounting and capture":
  test "Shared storage is counted once and only unused cache is released":
    let device = metalDevice()
    metalSynchronize(device)
    GC_fullCollect()
    metalEmptyCache(device)
    let baseline = metalMemoryStats(device)
    let allocated = scopedAllocation(device)
    check allocated.live.liveBytes == baseline.liveBytes + 24
    check allocated.aliased.liveBytes == allocated.live.liveBytes
    check allocated.shares
    metalSynchronize(device)
    GC_fullCollect()
    let released = metalMemoryStats(device)
    check released.liveBytes == baseline.liveBytes
    check released.cachedBytes >= 24
    block:
      let reused = newMetalTensorUninit[float32]([2, 3].toMetadata, device)
      let current = metalMemoryStats(device)
      check current.liveBytes == baseline.liveBytes + 24
      check current.cachedBytes == released.cachedBytes - 24
      check current.reservedBytes == released.reservedBytes
      check reused.storage.device.registryId == device.registryId
    GC_fullCollect()
    metalEmptyCache(device)
    check metalMemoryStats(device).cachedBytes == 0
    check metalCurrentAllocatedMemory(device) == baseline.liveBytes
    check metalRecommendedMaxMemory(device) > 0

  test "Cache clearing cannot recycle a dropped input held by pending work":
    let device = metalDevice()
    metalSynchronize(device)
    GC_fullCollect()
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
    block:
      withMetalStream(stream):
        let input = [1'f32, 2].toTensor.metal(device)
        let output = input + 1'f32
        alias = output.metalView([2].toMetadata, [1].toMetadata, 0)
    GC_fullCollect()
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
    GC_fullCollect()
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
