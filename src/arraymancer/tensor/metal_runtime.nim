## Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides thread-local Metal stream selection, memory accounting and native event ordering.
## Built-in Metal commands enqueue on the current stream. Independent streams have
## no implicit ordering; use events for cross-stream dependencies. Tensor readback
## waits for the tensor's producing work, not for unrelated streams. Stream sync
## waits only that queue; device sync waits previous work on every registered queue.
## Default-stream built-in math shares command buffers up to 32 dispatches or 4 MiB
## of queued output capacity, with a 100-microsecond submission deadline. An
## independent timer submits idle batches; host scheduling can delay that timer.
## Explicit streams commit eagerly. Sync, events, readback and foreign access flush
## pending math before exposing or waiting for its work.
import std/os
import ./[metal_types, init_metal]
export metal_types, init_metal

var selectedMetalStream {.threadvar.}: MetalStream

proc requireDevice(device: MetalDevice) {.gcsafe.} =
  if device == nil or device.owner == nil or device.owner.handle == nil:
    raise newException(MetalError, "Metal device is unavailable")

proc resolveDevice(device: MetalDevice): MetalDevice {.gcsafe.} =
  if device != nil:
    requireDevice(device)
    return device
  metalDevice()

type
  MetalMemoryStats* = object
    ## inFlightBytes overlaps liveBytes; it measures unique allocations with pending work.
    ## driverBytes/recommendedBytes are native MTLDevice measurements, not payload totals.
    liveBytes*, cachedBytes*, inFlightBytes*, driverBytes*: uint64
    recommendedBytes*, reservedBytes*, hostCopyBytes*, materializationBytes*: uint64

proc metalMemoryStats*(device: MetalDevice = nil): MetalMemoryStats {.gcsafe.} =
  ## Return payload, cache, in-flight, driver and cumulative transfer measurements.
  let dev = resolveDevice(device)
  var native: AMMemoryStats
  checkMetal metalApi().memoryStats(dev.owner.handle, addr native)
  result = MetalMemoryStats(
    liveBytes: native.liveBytes,
    cachedBytes: native.cachedBytes,
    inFlightBytes: native.inFlightBytes,
    driverBytes: native.driverBytes,
    recommendedBytes: native.recommendedBytes,
    reservedBytes: native.reservedBytes,
    hostCopyBytes: native.hostCopyBytes,
    materializationBytes: native.materializationBytes)

proc metalCurrentAllocatedMemory*(device: MetalDevice = nil): uint64 {.gcsafe.} =
  ## Return bytes held by live tensor storage, counted once per shared owner.
  metalMemoryStats(device).liveBytes

proc metalDriverAllocatedMemory*(device: MetalDevice = nil): uint64 {.gcsafe.} =
  ## Return MTLDevice.currentAllocatedSize, distinct from tensor payload accounting.
  metalMemoryStats(device).driverBytes

proc metalRecommendedMaxMemory*(device: MetalDevice = nil): uint64 {.gcsafe.} =
  ## Return MTLDevice.recommendedMaxWorkingSetSize.
  metalMemoryStats(device).recommendedBytes

proc metalEmptyCache*(device: MetalDevice = nil) {.gcsafe.} =
  ## Release completed cached allocations without invalidating live or pending buffers.
  let dev = resolveDevice(device)
  checkMetal metalApi().memoryEmptyCache(dev.owner.handle)

proc metalMemoryFraction*(device: MetalDevice = nil): float64 {.gcsafe.} =
  ## Return the native allocation limit fraction; zero means unlimited.
  let dev = resolveDevice(device)
  checkMetal metalApi().memoryFraction(dev.owner.handle, addr result, 0)

proc setMetalMemoryFraction*(device: MetalDevice; fraction: float64) {.gcsafe.} =
  ## Require a finite fraction in [0, 2]; zero disables the limit.
  ## Allocations are limited to fraction × recommended working set.
  if fraction != fraction or fraction < 0.0 or fraction > 2.0:
    raise newException(ValueError, "Metal memory fraction must be finite and in [0, 2]")
  let dev = resolveDevice(device)
  var value = fraction
  checkMetal metalApi().memoryFraction(dev.owner.handle, addr value, 1)

proc startMetalCapture*(device: MetalDevice; outputPath: string): MetalCapture {.gcsafe.} =
  ## Start an MTLCaptureManager capture at an absolute output path.
  if outputPath.len == 0:
    raise newException(MetalError, "Metal capture output path must not be empty")
  let dev = resolveDevice(device)
  let path = absolutePath(outputPath)
  var handle: pointer
  checkMetal metalApi().captureStart(dev.owner.handle, path.cstring, addr handle)
  result = MetalCapture(owner: ownMetal(handle), device: dev, active: true)

proc stopMetalCapture*(capture: MetalCapture) {.gcsafe.} =
  ## Stop a capture explicitly; its retained native handle releases with its owner.
  if capture == nil or capture.owner == nil or capture.owner.handle == nil or
      capture.device == nil or capture.device.owner == nil or capture.device.owner.handle == nil:
    raise newException(MetalError, "Metal capture is unavailable")
  if not capture.active:
    raise newException(MetalError, "Metal capture has already been stopped")
  checkMetal metalApi().captureStop(capture.owner.handle)
  capture.active = false

proc requireStream(stream: MetalStream) {.gcsafe.} =
  if stream == nil or stream.owner == nil or stream.owner.handle == nil:
    raise newException(MetalError, "Metal stream is unavailable")
  requireDevice(stream.device)

proc requireEvent(event: MetalEvent) {.gcsafe.} =
  if event == nil or event.owner == nil or event.owner.handle == nil:
    raise newException(MetalError, "Metal event is unavailable")
  requireDevice(event.device)

proc requireSameDevice(stream: MetalStream; event: MetalEvent) {.gcsafe.} =
  requireStream(stream)
  requireEvent(event)
  if stream.device.registryId != event.device.registryId:
    raise newException(MetalError, "Metal event and stream belong to different devices")

proc newMetalStream*(device: MetalDevice = nil): MetalStream {.gcsafe.} =
  ## Create a retained native command queue on this device.
  let dev = resolveDevice(device)
  var handle: pointer
  var identity: uint64
  checkMetal metalApi().stream(dev.owner.handle, 0, addr handle, addr identity)
  result = MetalStream(owner: ownMetal(handle), device: dev, identity: identity)

proc defaultMetalStream*(device: MetalDevice = nil): MetalStream {.gcsafe.} =
  ## Return the native default queue without changing this thread's selection.
  let dev = resolveDevice(device)
  var handle: pointer
  var identity: uint64
  checkMetal metalApi().stream(dev.owner.handle, 1, addr handle, addr identity)
  result = MetalStream(owner: ownMetal(handle), device: dev, identity: identity)

proc currentMetalStream*(device: MetalDevice = nil): MetalStream {.gcsafe.} =
  ## Return this thread's selected queue, or the device's default queue.
  if device != nil: requireDevice(device)
  if selectedMetalStream != nil:
    if device != nil and selectedMetalStream.device.registryId != device.registryId:
      raise newException(MetalError, "Selected Metal stream belongs to a different device")
    return selectedMetalStream
  defaultMetalStream(device)

proc setMetalStream*(stream: MetalStream) {.gcsafe.} =
  ## Select and retain this queue on the calling thread; native TLS mirrors it.
  requireStream(stream)
  checkMetal metalApi().streamSet(stream.owner.handle)
  selectedMetalStream = stream

proc newMetalEvent*(device: MetalDevice = nil): MetalEvent {.gcsafe.} =
  ## Create a retained shared-timeline event for explicit cross-stream ordering.
  let dev = resolveDevice(device)
  var handle: pointer
  checkMetal metalApi().event(dev.owner.handle, addr handle)
  result = MetalEvent(owner: ownMetal(handle), device: dev)

proc record*(event: MetalEvent; stream: MetalStream): uint64 {.gcsafe, discardable.} =
  ## Enqueue a signal on stream and return its native monotonic timeline value.
  requireSameDevice(stream, event)
  checkMetal metalApi().eventRecord(event.owner.handle, stream.owner.handle, addr result)

proc wait*(stream: MetalStream; event: MetalEvent) {.gcsafe.} =
  ## Enqueue a wait for the event's latest recorded value; unrecorded events error.
  requireSameDevice(stream, event)
  checkMetal metalApi().eventWait(stream.owner.handle, event.owner.handle, 0'u64, 0)

proc wait*(stream: MetalStream; event: MetalEvent; value: uint64) {.gcsafe.} =
  ## Enqueue a wait for an exact timeline value, including a future value.
  requireSameDevice(stream, event)
  checkMetal metalApi().eventWait(stream.owner.handle, event.owner.handle, value, 1)

proc signalMetalEvent*(event: MetalEvent; value: uint64) {.gcsafe.} =
  ## Host-signal a timeline value; native code rejects nonmonotonic signals.
  requireEvent(event)
  checkMetal metalApi().eventSignal(event.owner.handle, value)

proc metalStreamIsComplete*(stream: MetalStream): bool {.gcsafe.} =
  ## Query native completion without sleeps or host synchronization.
  requireStream(stream)
  var complete: int32
  checkMetal metalApi().streamComplete(stream.owner.handle, addr complete)
  result = complete != 0

proc metalSynchronize*(stream: MetalStream) {.gcsafe.} =
  ## Wait for work previously submitted to this stream only.
  requireStream(stream)
  checkMetal metalApi().streamSync(stream.owner.handle)

proc metalSynchronize*(device: MetalDevice = nil) {.gcsafe.} =
  ## Wait for previous work on every registered stream of this device.
  let dev = resolveDevice(device)
  checkMetal metalApi().deviceSync(dev.owner.handle)

template withMetalStream*(stream: MetalStream; body: untyped): untyped =
  ## Select a stream for body and restore this thread's prior current stream on exit.
  block:
    let previousMetalStream = currentMetalStream()
    setMetalStream(stream)
    try:
      body
    finally:
      setMetalStream(previousMetalStream)
