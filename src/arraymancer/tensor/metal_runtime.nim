# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides thread-local Metal stream selection and native event ordering.
## Built-in Metal commands enqueue on the current stream. Independent streams have
## no implicit ordering; use events for cross-stream dependencies. Tensor readback
## waits for the tensor's producing work, not for unrelated streams. Stream sync
## waits only that queue; device sync waits previous work on every registered queue.
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
