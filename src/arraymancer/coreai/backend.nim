# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides the lazy, typed Core AI C ABI loader without link-time SDK dependencies.
import std/[dynlib, locks, os]

when not (defined(macosx) and defined(arm64)):
  {.error: "Core AI requires macOS arm64".}

type
  CoreAIError* = object of CatchableError
    nativeCode*: int32
  CoreAIUnavailableError* = object of CoreAIError
  CoreAIDescriptorError* = object of CoreAIError
  CoreAITimeoutError* = object of CoreAIError
  CoreAIBufferBusyError* = object of CoreAIError

  AMStatus* {.bycopy.} = object
    code*: int32
    message*: cstring

  AMTensorDescriptor* {.bycopy.} = object
    rank*, dtype*: int32
    shape*, strides*: array[8, int64]
    offset*: int64
    capacity*: uint64

  ACArrayDescriptor* {.bycopy.} = object
    name*, dtype*: cstring
    rank*, dynamicShape*, resolved*, reserved*: int32
    shape*, strides*: array[8, int64]
    minimumBytes*, hostCopyBytes*: uint64
    descriptorHandle*: pointer
    bufferIdentity*, byteOffset*: uint64
    borrowed*, reserved2*: int32
  ACBinding* {.bycopy.} = object
    name*: cstring
    array*: pointer

  ACEvidence* {.bycopy.} = object
    invocationId*: uint64
    completed*, kind*, requestedPolicy*, actualPlacement*: int32
    availableMask*, allowedMask*: uint32
    preferredUnit*, reserved*: int32
    hostCopyBytes*, materializationBytes*, cpuFallbackCount*: uint64
    assetDigest*, functionName*, architecture*: cstring
    borrowedInputCount*, producerCount*: uint64
    streamIdentity*: uint64
    sdkStreamCompleted*, metalQueueCompleted*: int32

  AMStreamInfo* = proc(stream: pointer; metalQueue: ptr pointer;
    identity: ptr uint64): AMStatus {.cdecl, gcsafe, raises: [].}
  AMStreamFence* = proc(stream: pointer; fence: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
  AMStreamFenceWait* = proc(fence: pointer): AMStatus {.cdecl, gcsafe, raises: [].}
  ACStreamCreate* = proc(metalStream: pointer; stream: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
  ACEncode* = proc(function: pointer; inputs: ptr ACBinding; count: uint32;
    outputViews: ptr ACBinding; outputCount: uint32; stream: pointer;
    job: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
  ACStreamAwait* = proc(stream: pointer; job: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}

  CoreAIApi* = object
    messageFree*: proc(message: cstring) {.cdecl, gcsafe, raises: [].}
    retain*, release*: proc(handle: pointer) {.cdecl, gcsafe, raises: [].}
    architecture*: proc(outValue: ptr cstring): AMStatus {.cdecl, gcsafe, raises: [].}
    availableUnits*: proc(mask: ptr uint32): AMStatus {.cdecl, gcsafe, raises: [].}
    modelOpen*: proc(path: cstring; policy: int32; job: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    jobWait*: proc(job: pointer; timeoutMs: uint32): AMStatus {.cdecl, gcsafe, raises: [].}
    jobModel*: proc(job: pointer; model: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    functionCount*: proc(model: pointer; count: ptr uint32): AMStatus {.cdecl, gcsafe, raises: [].}
    functionName*: proc(model: pointer; index: uint32; name: ptr cstring): AMStatus {.cdecl, gcsafe, raises: [].}
    functionLoad*: proc(model: pointer; name: cstring; function: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    descriptorCount*: proc(function: pointer; kind: int32; count: ptr uint32): AMStatus {.cdecl, gcsafe, raises: [].}
    descriptor*: proc(function: pointer; kind: int32; index: uint32; value: ptr ACArrayDescriptor): AMStatus {.cdecl, gcsafe, raises: [].}
    arrayImport*: proc(host: pointer; descriptor: AMTensorDescriptor; array: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    arrayCopy*: proc(array: pointer; host: pointer; descriptor: AMTensorDescriptor): AMStatus {.cdecl, gcsafe, raises: [].}
    arrayDescriptor*: proc(array: pointer; descriptor: ptr ACArrayDescriptor): AMStatus {.cdecl, gcsafe, raises: [].}
    arrayView*: proc(buffer: pointer; descriptor: AMTensorDescriptor; array: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    descriptorResolve*: proc(descriptorHandle: pointer; shape: ptr int64;
      rank: uint32; outValue: ptr ACArrayDescriptor): AMStatus {.cdecl, gcsafe, raises: [].}
    submit*: proc(function: pointer; inputs: ptr ACBinding; count: uint32; job: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    jobOutputCount*: proc(job: pointer; count: ptr uint32): AMStatus {.cdecl, gcsafe, raises: [].}
    jobOutput*: proc(job: pointer; index: uint32; name: ptr cstring; array: ptr pointer): AMStatus {.cdecl, gcsafe, raises: [].}
    jobEvidence*: proc(job: pointer; evidence: ptr ACEvidence): AMStatus {.cdecl, gcsafe, raises: [].}
    streamInfo*: AMStreamInfo
    streamFence*: AMStreamFence
    streamFenceWait*: AMStreamFenceWait
    streamCreate*: ACStreamCreate
    encode*: ACEncode
    streamAwait*: ACStreamAwait

const arraymancerAppleLibDir {.strdefine.} = currentSourcePath().parentDir.parentDir.parentDir.parentDir / "build/apple"

var api: CoreAIApi
var library: LibHandle
var loaderLock: Lock
initLock(loaderLock)

proc coreAIApi*(): CoreAIApi {.gcsafe.} =
  ## Load only for explicit Core AI use; a failed load can be retried.
  withLock loaderLock:
    if library == nil:
      let handle = loadLib(arraymancerAppleLibDir / "libarraymancer_coreai.dylib")
      if handle == nil:
        var error = newException(CoreAIUnavailableError,
          "Cannot load Core AI bridge from " & arraymancerAppleLibDir)
        error.nativeCode = 100
        raise error
      template resolve(field, symbol: untyped) =
        api.field = cast[typeof(api.field)](symAddr(handle, symbol))
        if api.field == nil:
          unloadLib(handle)
          var error = newException(CoreAIUnavailableError, "Missing Core AI ABI symbol: " & symbol)
          error.nativeCode = 100
          raise error
      resolve(messageFree, "ac_message_free")
      resolve(retain, "ac_retain")
      resolve(release, "ac_release")
      resolve(architecture, "ac_architecture")
      resolve(availableUnits, "ac_available_units")
      resolve(modelOpen, "ac_model_open")
      resolve(jobWait, "ac_job_wait")
      resolve(jobModel, "ac_job_model")
      resolve(functionCount, "ac_function_count")
      resolve(functionName, "ac_function_name")
      resolve(functionLoad, "ac_function_load")
      resolve(descriptorCount, "ac_descriptor_count")
      resolve(descriptor, "ac_descriptor")
      resolve(arrayImport, "ac_array_import")
      resolve(arrayCopy, "ac_array_copy")
      resolve(arrayDescriptor, "ac_array_descriptor")
      resolve(arrayView, "ac_array_view")
      resolve(descriptorResolve, "ac_descriptor_resolve")
      resolve(submit, "ac_submit")
      resolve(jobOutputCount, "ac_job_output_count")
      resolve(jobOutput, "ac_job_output")
      resolve(jobEvidence, "ac_job_evidence")
      resolve(streamInfo, "am_stream_info")
      resolve(streamFence, "am_stream_fence")
      resolve(streamFenceWait, "am_stream_fence_wait")
      resolve(streamCreate, "ac_stream_create")
      resolve(encode, "ac_encode")
      resolve(streamAwait, "ac_stream_await")
      library = handle
    result = api
 

proc releaseCString*(value: cstring) {.gcsafe.} =
  if value != nil:
    coreAIApi().messageFree(value)

proc checkCoreAI*(status: AMStatus) {.gcsafe.} =
  if status.code == 0: return
  var message = "Native Core AI error"
  if status.message != nil:
    message = $status.message
    coreAIApi().messageFree(status.message)
  case status.code
  of 100:
    var error = newException(CoreAIUnavailableError, message)
    error.nativeCode = status.code
    raise error
  of 102:
    var error = newException(CoreAIDescriptorError, message)
    error.nativeCode = status.code
    raise error
  of 105:
    var error = newException(CoreAIBufferBusyError, message)
    error.nativeCode = status.code
    raise error
  of 104:
    var error = newException(CoreAITimeoutError, message)
    error.nativeCode = status.code
    raise error
  else:
    var error = newException(CoreAIError, message)
    error.nativeCode = status.code
    raise error
