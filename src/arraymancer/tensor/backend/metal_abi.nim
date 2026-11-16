# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides the lazy, typed Metal ABI loader without link-time SDK dependencies.
import std/[dynlib, os, locks]
when not (defined(macosx) and defined(arm64)):
  {.error: "Metal and Core AI require macOS arm64".}

type
  MetalError* = object of CatchableError
  MetalUnavailableError* = object of MetalError
  MetalDtypeError* = object of MetalError
  AMStatus* {.bycopy.} = object
    code*: int32
    message*: cstring
  AMDescriptor* {.bycopy.} = object
    rank*, dtype*: int32
    shape*, strides*: array[8, int64]
    offset*: int64
    capacity*: uint64
  MetalApi* = object
    messageFree*: proc(message: cstring) {.cdecl, raises: [].}
    retain*, release*: proc(handle: pointer) {.cdecl, raises: [].}
    deviceCount*: proc(count: ptr int32): AMStatus {.cdecl.}
    device*: proc(index: int32; handle: ptr pointer; identity: ptr uint64): AMStatus {.cdecl.}
    buffer*: proc(device: pointer; bytes: uint64; handle: ptr pointer): AMStatus {.cdecl.}
    transfer*: proc(buffer: pointer; descriptor: AMDescriptor; host: pointer; upload: int32): AMStatus {.cdecl.}

const arraymancerAppleLibDir* {.strdefine.} = currentSourcePath().parentDir.parentDir.parentDir.parentDir.parentDir / "build/apple"
var api: MetalApi
var library: LibHandle
var loaderLock: Lock
initLock(loaderLock)

proc metalApi*(): MetalApi =
  ## Load only on an explicit backend operation; a failed load can be retried.
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
      resolve(transfer, "am_transfer")
      library = handle
    result = api

proc checkMetal*(status: AMStatus) =
  if status.code != 0:
    let message = if status.message == nil: "Native Metal error" else: $status.message
    if status.message != nil: metalApi().messageFree(status.message)
    if status.code == 1: raise newException(MetalUnavailableError, message)
    raise newException(MetalError, message)
