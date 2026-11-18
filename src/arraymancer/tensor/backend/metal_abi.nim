# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides the lazy, typed Metal ABI loader without link-time SDK dependencies.
import std/[dynlib, os, locks, atomics]
when not (defined(macosx) and defined(arm64)):
  {.error: "Metal and Core AI require macOS arm64".}

type
  MetalError* = object of CatchableError
  MetalUnavailableError* = object of MetalError
  MetalDtypeError* = object of MetalError
  MetalRngStateError* = object of MetalError
  MetalShaderCompileError* = object of MetalError
  MetalMissingFunctionError* = object of MetalError
  MetalLibraryLoadError* = object of MetalError
  MetalBufferBusyError* = object of MetalError

  MetalOutOfMemoryError* = object of MetalError
  AMRngState* {.bycopy.} = object
    version*, reserved*: uint32
    seed*: uint64
    counter*: array[4, uint32]
  AMStatus* {.bycopy.} = object
    code*: int32
    message*: cstring
  AMMemoryStats* {.bycopy.} = object
    liveBytes*, cachedBytes*, inFlightBytes*, driverBytes*: uint64
    recommendedBytes*, reservedBytes*, hostCopyBytes*, materializationBytes*: uint64
  AMDescriptor* {.bycopy.} = object
    rank*, dtype*: int32
    shape*, strides*: array[8, int64]
    offset*: int64
    capacity*: uint64
  AMTensorDescriptor* = AMDescriptor
  AMEvidence* {.bycopy.} = object
    invocationId*, registryId*, transferBytes*, materializationBytes*: uint64
    cpuFallbackCount*: uint64
    completed*, kind*, kernelBits*: int32
  AMMathParameters* {.bycopy.} = object
    operation*, scalarInput*, axis*, reserved*: int32
    scalar*: float32
  AMKernelBinding* {.bycopy.} = object
    buffer*: pointer
    descriptor*: AMTensorDescriptor
    access*, reserved*: int32
  AMGrid* {.bycopy.} = object
    x*, y*, z*: uint64


  MetalApi* = object
    library*: proc(device: pointer; text: cstring; mode: int32;
      outLibrary: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    kernel*: proc(library: pointer; name: cstring;
      outKernel: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    dispatch*: proc(kernel: pointer; bindings: ptr AMKernelBinding; count: uint32;
      grid, group: AMGrid; stream: pointer): AMStatus {.cdecl, gcsafe, raises: [].}

    messageFree*: proc(message: cstring) {.cdecl, gcsafe, raises: [].}
    retain*, release*: proc(handle: pointer) {.cdecl, gcsafe, raises: [].}
    deviceCount*: proc(count: ptr int32): AMStatus {.cdecl, gcsafe, raises: [].}
    device*: proc(index: int32; handle: ptr pointer; identity: ptr uint64): AMStatus {.cdecl, gcsafe, raises: [].}
    buffer*: proc(device: pointer; bytes: uint64; handle: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    bufferInfo*: proc(buffer: pointer; descriptor: AMTensorDescriptor;
      metalBuffer: ptr pointer; identity: ptr uint64): AMStatus {.cdecl, gcsafe, raises: [].}
    transfer*: proc(buffer: pointer; descriptor: AMDescriptor; host: pointer; upload: int32): AMStatus {.cdecl, gcsafe, raises: [].}
    evidence*: proc(buffer: pointer; evidence: ptr AMEvidence): AMStatus {.cdecl, gcsafe, raises: [].}
    copy*: proc(source: pointer; input: AMDescriptor; destination: pointer;
      output: AMDescriptor; libraryDir: cstring): AMStatus {.cdecl, gcsafe, raises: [].}
    setElement*: proc(destination: pointer; output: AMDescriptor; value: pointer;
      libraryDir: cstring): AMStatus {.cdecl, gcsafe, raises: [].}
    math*: proc(a: pointer; ad: AMDescriptor; b: pointer; bd: AMDescriptor;
      output: pointer; od: AMDescriptor; parameters: AMMathParameters;
      libraryDir: cstring): AMStatus {.cdecl, gcsafe, raises: [].}
    mseBackward*: proc(upstream: pointer; upstreamDescriptor: AMDescriptor;
      prediction: pointer; predictionDescriptor: AMDescriptor;
      target: pointer; targetDescriptor: AMDescriptor; output: pointer;
      outputDescriptor: AMDescriptor; libraryDir: cstring): AMStatus {.cdecl, gcsafe, raises: [].}
    stream*: proc(device: pointer; mode: int32; handle: ptr pointer;
      identity: ptr uint64): AMStatus {.cdecl, gcsafe, raises: [].}
    streamSet*: proc(stream: pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    streamSync*: proc(stream: pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    deviceSync*: proc(device: pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    streamComplete*: proc(stream: pointer; complete: ptr int32): AMStatus {.cdecl, gcsafe, raises: [].}
    event*: proc(device: pointer; handle: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    eventRecord*: proc(event, stream: pointer; recorded: ptr uint64): AMStatus {.cdecl, gcsafe, raises: [].}
    eventWait*: proc(stream, event: pointer; value: uint64;
      explicitValue: int32): AMStatus {.cdecl, gcsafe, raises: [].}
    eventSignal*: proc(event: pointer; value: uint64): AMStatus {.cdecl, gcsafe, raises: [].}
    memoryStats*: proc(device: pointer; stats: ptr AMMemoryStats): AMStatus {.cdecl, gcsafe, raises: [].}
    memoryEmptyCache*: proc(device: pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    memoryFraction*: proc(device: pointer; value: ptr cdouble; set: int32): AMStatus {.cdecl, gcsafe, raises: [].}
    captureStart*: proc(device: pointer; path: cstring; capture: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    captureStop*: proc(capture: pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    generator*: proc(device: pointer; seed: uint64; mode: int32;
      outHandle: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    rngState*: proc(generator: pointer; state: ptr AMRngState;
      mode: int32): AMStatus {.cdecl, gcsafe, raises: [].}
    rngSeed*: proc(generator: pointer; seed: ptr uint64;
      entropy: int32): AMStatus {.cdecl, gcsafe, raises: [].}
    rand*: proc(generator, output: pointer; descriptor: AMDescriptor;
      libraryDir: cstring): AMStatus {.cdecl, gcsafe, raises: [].}

const arraymancerAppleLibDir* {.strdefine.} = currentSourcePath().parentDir.parentDir.parentDir.parentDir.parentDir / "build/apple"
var api: MetalApi
var library: LibHandle
var loaderLock: Lock
var apiReady: Atomic[bool]
initLock(loaderLock)

proc metalApi*(): lent MetalApi {.gcsafe.} =
  ## Borrow the immutable loaded table instead of copying every function pointer.
  ## Load only on an explicit backend operation; a failed load can be retried.
  ## Publish only a complete table; subsequent calls need no loader mutex.
  if apiReady.load(moAcquire):
    return api
  withLock loaderLock:
    if library == nil:
      let handle = loadLib(arraymancerAppleLibDir / "libarraymancer_metal.dylib")
      if handle == nil:
        raise newException(MetalUnavailableError, "Cannot load Metal bridge from " & arraymancerAppleLibDir)
      template resolve(field, symbol: untyped) =
        api.field = cast[typeof(api.field)](symAddr(handle, symbol))
        if api.field == nil:
          unloadLib(handle)
          raise newException(MetalUnavailableError, "Missing Metal ABI symbol: " & symbol)
      resolve(messageFree, "am_message_free")
      resolve(retain, "am_retain")
      resolve(release, "am_release")
      resolve(deviceCount, "am_device_count")
      resolve(device, "am_device")
      resolve(buffer, "am_buffer")
      resolve(bufferInfo, "am_buffer_info")
      resolve(evidence, "am_buffer_evidence")
      resolve(transfer, "am_transfer")
      resolve(library, "am_library")
      resolve(kernel, "am_kernel")
      resolve(dispatch, "am_dispatch")

      resolve(copy, "am_copy")
      resolve(setElement, "am_set_element")
      resolve(math, "am_math")
      resolve(mseBackward, "am_mse_backward")
      resolve(stream, "am_stream")
      resolve(streamSet, "am_stream_set")
      resolve(streamSync, "am_stream_sync")
      resolve(deviceSync, "am_device_sync")
      resolve(streamComplete, "am_stream_complete")
      resolve(event, "am_event")
      resolve(eventRecord, "am_event_record")
      resolve(eventWait, "am_event_wait")
      resolve(eventSignal, "am_event_signal")
      resolve(memoryStats, "am_memory_stats")
      resolve(memoryEmptyCache, "am_memory_empty_cache")
      resolve(memoryFraction, "am_memory_fraction")
      resolve(captureStart, "am_capture_start")
      resolve(captureStop, "am_capture_stop")
      resolve(generator, "am_generator")
      resolve(rngState, "am_rng_state")
      resolve(rngSeed, "am_rng_seed")
      resolve(rand, "am_rand")
      library = handle
      apiReady.store(true, moRelease)
    result = api

proc checkMetal*(status: AMStatus) {.gcsafe.} =
  if status.code != 0:
    let message = if status.message == nil: "Native Metal error" else: $status.message
    if status.message != nil: metalApi().messageFree(status.message)
    if status.code == 1: raise newException(MetalUnavailableError, message)
    if status.code == 4: raise newException(MetalOutOfMemoryError, message)
    if status.code == 7: raise newException(MetalShaderCompileError, message)
    if status.code == 8: raise newException(MetalMissingFunctionError, message)
    if status.code == 9: raise newException(MetalLibraryLoadError, message)
    if status.code == 10: raise newException(MetalBufferBusyError, message)

    if status.code == 6: raise newException(MetalRngStateError, message)
    raise newException(MetalError, message)
