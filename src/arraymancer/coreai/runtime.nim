# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides typed Core AI model, descriptor, array and inference-job APIs.
import std/options
from ../tensor/metal_types import MetalTensor, MetalStream
import ../tensor/[data_structure, init_cpu]
import ./backend except CoreAIError, CoreAIUnavailableError, CoreAIDescriptorError, CoreAITimeoutError, CoreAIBufferBusyError
type
  CoreAIError* = backend.CoreAIError
  CoreAIUnavailableError* = backend.CoreAIUnavailableError
  CoreAIDescriptorError* = backend.CoreAIDescriptorError
  CoreAITimeoutError* = backend.CoreAITimeoutError
  CoreAIBufferBusyError* = backend.CoreAIBufferBusyError

  CoreAIComputePolicy* = enum
    ## Preferences may fall back; they do not establish actual hardware placement.
    cpDefault = 0
    cpCpuOnly = 1
    cpGpuPreferred = 2
    cpNeuralEnginePreferred = 3

  CoreAIComputeUnit* = enum
    cacuCPU = 0
    cacuGPU = 1
    cacuNeuralEngine = 2

  CoreAIEvidence* = object
    ## Native invocation facts. Copy counters cover the bridge, not SDK internals.
    ## ``actualPlacement == -1`` means the SDK did not expose placement.
    invocationId*: uint64
    completed*: bool
    kind*: int32
    requestedPolicy*: CoreAIComputePolicy
    actualPlacement*, preferredUnit*: int32
    availableMask*, allowedMask*: uint32
    hostCopyBytes*, materializationBytes*, cpuFallbackCount*: uint64
    borrowedInputCount*, producerCount*: uint64
    streamIdentity*: uint64
    sdkStreamCompleted*, metalQueueCompleted*: bool
    assetDigest*, functionName*, architecture*: string

  CoreAIStream* = ref object
    ## Owns the Core AI wrapper for one retained Metal queue and its SDK stream.
    owner: CoreAIHandle
    metalStream: MetalStream
    streamIdentity*: uint64

  CoreAIHandleObj = object
    handle: pointer
    release: proc(handle: pointer) {.cdecl, gcsafe, raises: [].}
  CoreAIHandle = ref CoreAIHandleObj

  CoreAIArrayDescriptor* = object
    ## Dynamic dimensions are -1. Strides/minimumBytes are unavailable until resolved.
    name*, dtype*: string
    shape*, strides*: Metadata
    dynamicShape*, resolved*, borrowed*: bool
    minimumBytes*, hostCopyBytes*, bufferIdentity*, byteOffset*: uint64
    descriptorOwner: CoreAIHandle
  CoreAIFunctionDescriptor* = object
    name*: string
    inputs*, outputs*, states*: seq[CoreAIArrayDescriptor]

  CoreAIModel* = ref object
    ## Owns a loaded SDK model; compute options are selected when opening it.
    owner: CoreAIHandle
    policy*: CoreAIComputePolicy

  CoreAIFunction* = ref object
    owner: CoreAIHandle
    model: CoreAIModel
    name*: string

  CoreAIArray* = ref object
    owner: CoreAIHandle

  CoreAIBinding* = object
    name*: string
    array*: CoreAIArray

  CoreAIJob* = ref object
    ## Native work retains dependencies until completion, even after a wait times out.
    owner: CoreAIHandle
    function: CoreAIFunction
    stream: CoreAIStream
when defined(gcDestructors):
  proc `=destroy`(owner: var CoreAIHandleObj) =
    if owner.handle != nil:
      if owner.release != nil: owner.release(owner.handle)
      owner.handle = nil
else:
  proc finalize(owner: CoreAIHandle) =
    if owner.handle != nil:
      if owner.release != nil: owner.release(owner.handle)
      owner.handle = nil

proc ownCoreAI(handle: pointer): CoreAIHandle =
  if handle == nil:
    raise newException(CoreAIError, "Native Core AI returned a nil handle")
  let releaseProc = coreAIApi().release
  when defined(gcDestructors): new(result)
  else: new(result, finalize)
  result.handle = handle
  result.release = releaseProc

proc descriptorFailure(message: string) =
  var error = newException(CoreAIDescriptorError, message)
  error.nativeCode = 102
  raise error

proc requireOwner(owner: CoreAIHandle; what: string): pointer =
  if owner == nil or owner.handle == nil:
    raise newException(CoreAIError, "Invalid or released Core AI " & what)
  owner.handle

proc freeArrayDescriptor(value: var ACArrayDescriptor) =
  releaseCString(value.name)
  releaseCString(value.dtype)
  value.name = nil
  value.dtype = nil
  if value.descriptorHandle != nil:
    coreAIApi().release(value.descriptorHandle)
    value.descriptorHandle = nil

proc toPublicDescriptor(value: var ACArrayDescriptor): CoreAIArrayDescriptor =
  if value.rank < 0 or value.rank > 8 or value.rank > LASER_MAXRANK:
    descriptorFailure("Core AI descriptor rank is outside the supported range")
  if value.dynamicShape notin 0..1 or value.resolved notin 0..1 or value.borrowed notin 0..1:
    descriptorFailure("Core AI descriptor contains invalid shape flags")
  result.name = if value.name == nil: "" else: $value.name
  result.dtype = if value.dtype == nil: "" else: $value.dtype
  result.dynamicShape = value.dynamicShape != 0
  result.resolved = value.resolved != 0
  result.borrowed = value.borrowed != 0
  result.minimumBytes = value.minimumBytes
  result.hostCopyBytes = value.hostCopyBytes
  result.bufferIdentity = value.bufferIdentity
  result.byteOffset = value.byteOffset
  result.shape.len = int(value.rank)
  result.strides.len = int(value.rank)
  for axis in 0 ..< int(value.rank):
    let dimension = value.shape[axis]
    if dimension < 0 and not (dimension == -1 and result.dynamicShape and not result.resolved):
      descriptorFailure("Core AI descriptor contains an invalid dimension")
    result.shape[axis] = int(dimension)
    result.strides[axis] = int(value.strides[axis])
  if value.descriptorHandle != nil:
    result.descriptorOwner = ownCoreAI(value.descriptorHandle)
    value.descriptorHandle = nil
proc descriptor*(array: CoreAIArray): CoreAIArrayDescriptor =
  ## Snapshot the SDK array's concrete shape, strides, dtype and bridge copy count.
  let handle = requireOwner(if array == nil: nil else: array.owner, "array")
  var native: ACArrayDescriptor
  let status = coreAIApi().arrayDescriptor(handle, addr native)
  try:
    checkCoreAI(status)
    result = toPublicDescriptor(native)
  finally:
    freeArrayDescriptor(native)

proc getFunctionDescriptors(function: CoreAIFunction; kind: int32): seq[CoreAIArrayDescriptor] =
  let handle = requireOwner(function.owner, "function")
  var count: uint32
  checkCoreAI(coreAIApi().descriptorCount(handle, kind, addr count))
  result = newSeqOfCap[CoreAIArrayDescriptor](int(count))
  for index in 0'u32 ..< count:
    var native: ACArrayDescriptor
    let status = coreAIApi().descriptor(handle, kind, index, addr native)
    try:
      checkCoreAI(status)
      result.add(toPublicDescriptor(native))
    finally:
      freeArrayDescriptor(native)

proc descriptor*(function: CoreAIFunction): CoreAIFunctionDescriptor =
  ## Return named SDK input/output/state descriptors without resolving dynamic shapes.
  if function == nil: raise newException(CoreAIError, "Invalid Core AI function")
  result.name = function.name
  result.inputs = getFunctionDescriptors(function, 0)
  result.outputs = getFunctionDescriptors(function, 1)
  result.states = getFunctionDescriptors(function, 2)

proc coreAIArchitecture*(): string =
  ## Return the SDK architecture suffix used to select a matching compiled asset.
  var architecture: cstring
  let status = coreAIApi().architecture(addr architecture)
  try:
    checkCoreAI(status)
    if architecture == nil:
      raise newException(CoreAIError, "Core AI returned an empty architecture name")
    result = $architecture
  finally:
    releaseCString(architecture)

proc unitsFromMask(mask: uint32): set[CoreAIComputeUnit] =
  for unit in CoreAIComputeUnit:
    if (mask and (1'u32 shl ord(unit))) != 0:
      result.incl(unit)

proc coreAIAvailableUnits*(): set[CoreAIComputeUnit] =
  ## Enumerate SDK-available units; availability is not evidence of execution.
  var mask: uint32
  checkCoreAI(coreAIApi().availableUnits(addr mask))
  result = unitsFromMask(mask)

proc policyFromNative(value: int32): CoreAIComputePolicy =
  if value < int32(ord(low(CoreAIComputePolicy))) or value > int32(ord(high(CoreAIComputePolicy))):
    raise newException(CoreAIError, "Core AI returned an unknown compute policy")
  CoreAIComputePolicy(value)

proc freeEvidence(value: var ACEvidence) =
  releaseCString(value.assetDigest)
  releaseCString(value.functionName)
  releaseCString(value.architecture)
  value.assetDigest = nil
  value.functionName = nil
  value.architecture = nil

proc coreAIJobEvidence*(job: CoreAIJob): CoreAIEvidence =
  ## Snapshot native completion, asset identity, policy and bridge-only copy facts.
  let handle = requireOwner(if job == nil: nil else: job.owner, "job")
  var native: ACEvidence
  let status = coreAIApi().jobEvidence(handle, addr native)
  try:
    checkCoreAI(status)
    if native.completed notin 0..1:
      raise newException(CoreAIError, "Core AI returned an invalid completion flag")
    result.invocationId = native.invocationId
    result.completed = native.completed != 0
    result.kind = native.kind
    result.requestedPolicy = policyFromNative(native.requestedPolicy)
    result.actualPlacement = native.actualPlacement
    result.preferredUnit = native.preferredUnit
    result.availableMask = native.availableMask
    result.allowedMask = native.allowedMask
    result.hostCopyBytes = native.hostCopyBytes
    result.materializationBytes = native.materializationBytes
    result.cpuFallbackCount = native.cpuFallbackCount
    result.borrowedInputCount = native.borrowedInputCount
    result.producerCount = native.producerCount
    if native.sdkStreamCompleted notin 0..1 or native.metalQueueCompleted notin 0..1:
      raise newException(CoreAIError, "Core AI returned invalid stream completion flags")
    result.streamIdentity = native.streamIdentity
    result.sdkStreamCompleted = native.sdkStreamCompleted != 0
    result.metalQueueCompleted = native.metalQueueCompleted != 0
    result.assetDigest = if native.assetDigest == nil: "" else: $native.assetDigest
    result.functionName = if native.functionName == nil: "" else: $native.functionName
    result.architecture = if native.architecture == nil: "" else: $native.architecture
  finally:
    freeEvidence(native)

proc functionNames*(model: CoreAIModel): seq[string] =
  ## Copy the model's actual SDK entrypoint names.
  let handle = requireOwner(if model == nil: nil else: model.owner, "model")
  var count: uint32
  checkCoreAI(coreAIApi().functionCount(handle, addr count))
  result = newSeqOfCap[string](int(count))
  for index in 0'u32 ..< count:
    var name: cstring
    let status = coreAIApi().functionName(handle, index, addr name)
    try:
      checkCoreAI(status)
      if name == nil:
        raise newException(CoreAIError, "Core AI returned a nil function name")
      result.add($name)
    finally:
      releaseCString(name)

proc containsNul(value: string): bool =
  for character in value:
    if character == '\0': return true

proc loadFunction*(model: CoreAIModel; name: string): Option[CoreAIFunction] =
  ## Return none for an absent entrypoint; native load failures raise CoreAIError.
  if containsNul(name):
    descriptorFailure("Core AI function name contains a null byte")
  let handle = requireOwner(if model == nil: nil else: model.owner, "model")
  var nativeFunction: pointer
  let status = coreAIApi().functionLoad(handle, name.cstring, addr nativeFunction)
  try:
    checkCoreAI(status)
  except:
    if nativeFunction != nil: coreAIApi().release(nativeFunction)
    raise
  if nativeFunction == nil: return none(CoreAIFunction)
  result = some(CoreAIFunction(owner: ownCoreAI(nativeFunction), model: model, name: name))

proc openCoreAIModel*(path: string; policy: CoreAIComputePolicy = cpDefault): CoreAIModel =
  ## Load a source .aimodel or architecture-matched .aimodelc with a 30-second wait.
  ## Timeout raises CoreAITimeoutError; pending native loading retains its resources.
  if path.len == 0 or containsNul(path):
    descriptorFailure("Core AI model path is empty or contains a null byte")
  var nativeJob: pointer
  let status = coreAIApi().modelOpen(path.cstring, int32(ord(policy)), addr nativeJob)
  try:
    checkCoreAI(status)
  except:
    if nativeJob != nil: coreAIApi().release(nativeJob)
    raise
  let job = ownCoreAI(nativeJob)
  checkCoreAI(coreAIApi().jobWait(job.handle, 30000'u32))
  var nativeModel: pointer
  let modelStatus = coreAIApi().jobModel(job.handle, addr nativeModel)
  try:
    checkCoreAI(modelStatus)
  except:
    if nativeModel != nil: coreAIApi().release(nativeModel)
    raise
  result = CoreAIModel(owner: ownCoreAI(nativeModel), policy: policy)

proc checkedAdd(a, b: int64): int64 =
  if b > 0 and a > high(int64) - b:
    descriptorFailure("Core AI tensor address calculation overflow")
  if b < 0 and a < low(int64) - b:
    descriptorFailure("Core AI tensor address calculation overflow")
  a + b

proc checkedMultiply(a, b: int64): int64 =
  if a == 0 or b == 0: return 0
  if a == -1:
    if b == low(int64): descriptorFailure("Core AI tensor address calculation overflow")
    return -b
  if b == -1:
    if a == low(int64): descriptorFailure("Core AI tensor address calculation overflow")
    return -a
  if a > 0:
    if b > 0:
      if a > high(int64) div b: descriptorFailure("Core AI tensor address calculation overflow")
    else:
      if b < low(int64) div a: descriptorFailure("Core AI tensor address calculation overflow")
  else:
    if b > 0:
      if a < low(int64) div b: descriptorFailure("Core AI tensor address calculation overflow")
    else:
      if a < high(int64) div b: descriptorFailure("Core AI tensor address calculation overflow")
  a * b

proc checkedShapeCount(shape: Metadata): int =
  if shape.len > 8 or shape.len > LASER_MAXRANK:
    descriptorFailure("Core AI tensor rank is outside the supported range")
  var prefix = 1
  var hasZero = false
  for dimension in shape:
    if dimension < 0: descriptorFailure("Core AI tensor has a negative dimension")
    if dimension == 0:
      hasZero = true
      prefix = 0
    elif prefix != 0:
      if prefix > high(int) div dimension:
        descriptorFailure("Core AI tensor element count overflows int")
      prefix *= dimension
  var suffix = 1
  for axis in countdown(shape.len - 1, 0):
    let dimension = shape[axis]
    if dimension != 0 and suffix != 0:
      if suffix > high(int) div dimension:
        descriptorFailure("Core AI tensor stride calculation overflows int")
      suffix *= dimension
    elif dimension == 0:
      suffix = 0
  if hasZero: 0 else: prefix

proc checkedByteCount(elementCount, elementWidth: int): uint64 =
  if elementCount < 0 or elementWidth <= 0:
    descriptorFailure("Core AI tensor byte count is invalid")
  let count = uint64(elementCount)
  let width = uint64(elementWidth)
  if count > high(uint64) div width:
    descriptorFailure("Core AI tensor byte count overflows uint64")
  count * width
proc resolveDescriptor*(descriptor: CoreAIArrayDescriptor;
                        shape: Metadata): CoreAIArrayDescriptor =
  ## Resolve through the retained SDK descriptor, preserving its actual layout.
  discard checkedShapeCount(shape)
  if descriptor.shape.len != shape.len:
    descriptorFailure("Core AI resolved shape has the wrong rank")
  if descriptor.strides.len != shape.len:
    descriptorFailure("Core AI descriptor shape and stride ranks differ")
  for axis in 0 ..< shape.len:
    let declared = descriptor.shape[axis]
    if declared >= 0:
      if declared != shape[axis]:
        descriptorFailure("Core AI resolved shape conflicts with a static dimension")
    elif declared != -1 or not descriptor.dynamicShape or descriptor.resolved:
      descriptorFailure("Core AI descriptor contains an invalid dynamic dimension")
  if descriptor.descriptorOwner == nil:
    if not descriptor.resolved:
      descriptorFailure("Core AI descriptor has no SDK handle for dynamic resolution")
    return descriptor
  var nativeShape: array[8, int64]
  for axis in 0 ..< shape.len:
    nativeShape[axis] = int64(shape[axis])
  var native: ACArrayDescriptor
  let status = coreAIApi().descriptorResolve(
    requireOwner(descriptor.descriptorOwner, "descriptor"),
    addr nativeShape[0], uint32(shape.len), addr native)
  try:
    checkCoreAI(status)
    result = toPublicDescriptor(native)
    if not result.resolved or result.shape.len != shape.len:
      descriptorFailure("Core AI did not resolve the requested descriptor shape")
    for axis in 0 ..< shape.len:
      if result.shape[axis] != shape[axis]:
        descriptorFailure("Core AI resolved a different concrete shape")
    if result.descriptorOwner == nil:
      descriptorFailure("Core AI did not retain the resolved SDK descriptor")
  finally:
    freeArrayDescriptor(native)

proc metalTensorDescriptor[T](tensor: MetalTensor[T]): AMTensorDescriptor =
  when T is float32:
    result.dtype = 1
  elif T is int32:
    result.dtype = 6
  else:
    descriptorFailure("Core AI Metal views support only float32 and int32")
  if tensor.storage == nil or tensor.storage.owner == nil or
     tensor.storage.owner.handle == nil:
    descriptorFailure("Cannot borrow an uninitialized Metal tensor")
  if tensor.shape.len != tensor.strides.len:
    descriptorFailure("Metal tensor shape and stride ranks differ")
  let elementCount = checkedShapeCount(tensor.shape)
  result.rank = int32(tensor.shape.len)
  result.offset = int64(tensor.offset)
  result.capacity = tensor.storage.capacity
  var empty = elementCount == 0
  var minimumIndex = result.offset
  var maximumIndex = result.offset
  for axis in 0 ..< tensor.shape.len:
    let dimension = tensor.shape[axis]
    let stride = int64(tensor.strides[axis])
    result.shape[axis] = int64(dimension)
    result.strides[axis] = stride
    if dimension == 0:
      empty = true
    elif not empty:
      let span = checkedMultiply(int64(dimension - 1), stride)
      if span < 0: minimumIndex = checkedAdd(minimumIndex, span)
      else: maximumIndex = checkedAdd(maximumIndex, span)
  if not empty:
    if minimumIndex < 0 or maximumIndex < minimumIndex:
      descriptorFailure("Metal tensor view is outside its storage")
    if maximumIndex == high(int64):
      descriptorFailure("Metal tensor byte extent overflows int64")
    let elements = uint64(maximumIndex + 1)
    let width = uint64(sizeof(T))
    if elements > high(uint64) div width:
      descriptorFailure("Metal tensor byte extent overflows uint64")
    if elements * width > result.capacity:
      descriptorFailure("Metal tensor view exceeds its native buffer capacity")

proc coreAIView*[T](tensor: MetalTensor[T]): CoreAIArray =
  ## Borrow the original shared Metal buffer and metadata without host copying.
  ## Submitted inference holds a whole-buffer read lease until completion;
  ## competing Metal writes raise ``MetalBufferBusyError``. Unsupported dtype,
  ## storage or layout raises ``CoreAIDescriptorError`` instead of packing.
  let descriptor = metalTensorDescriptor(tensor)
  var nativeArray: pointer
  let status = coreAIApi().arrayView(tensor.storage.owner.handle, descriptor,
    addr nativeArray)
  try:
    checkCoreAI(status)
  except:
    if nativeArray != nil: coreAIApi().release(nativeArray)
    raise
  result = CoreAIArray(owner: ownCoreAI(nativeArray))

proc hostDescriptor[T: float32 | int32](tensor: Tensor[T]): AMTensorDescriptor =
  if tensor.storage.isNil or tensor.storage.raw_buffer.isNil:
    descriptorFailure("Cannot import an uninitialized CPU tensor")
  if tensor.shape.len != tensor.strides.len:
    descriptorFailure("CPU tensor shape and stride ranks differ")
  let elementCount = checkedShapeCount(tensor.shape)
  result.rank = int32(tensor.shape.len)
  when T is float32: result.dtype = 1
  else: result.dtype = 6
  result.offset = int64(tensor.offset)
  var empty = elementCount == 0
  var minimumIndex = result.offset
  var maximumIndex = result.offset
  for axis in 0 ..< tensor.shape.len:
    let dimension = tensor.shape[axis]
    let stride = int64(tensor.strides[axis])
    result.shape[axis] = int64(dimension)
    result.strides[axis] = stride
    if dimension == 0:
      empty = true
    elif not empty:
      let span = checkedMultiply(int64(dimension - 1), stride)
      if span < 0: minimumIndex = checkedAdd(minimumIndex, span)
      else: maximumIndex = checkedAdd(maximumIndex, span)
  if empty:
    result.capacity = 0
  else:
    if minimumIndex < 0 or maximumIndex < minimumIndex:
      descriptorFailure("CPU tensor view is outside its storage")
    if maximumIndex == high(int64):
      descriptorFailure("Core AI tensor capacity overflows int64")
    let elements = uint64(maximumIndex + 1)
    let width = uint64(sizeof(T))
    if elements > high(uint64) div width:
      descriptorFailure("Core AI tensor byte capacity overflows uint64")
    result.capacity = elements * width

proc toCoreAI*[T: float32 | int32](tensor: Tensor[T]): CoreAIArray =
  ## Import directly from the CPU storage base; native code copies logical
  ## values using the supplied signed strides and offset before returning.
  ## The source must satisfy the CPU Tensor storage-bounds contract; CpuStorage
  ## does not retain an allocation length to independently validate forged metadata.
  let nativeDescriptor = hostDescriptor(tensor)
  let host = cast[pointer](tensor.get_data_ptr)
  var nativeArray: pointer
  let status = coreAIApi().arrayImport(host, nativeDescriptor, addr nativeArray)
  try:
    checkCoreAI(status)
  except:
    if nativeArray != nil: coreAIApi().release(nativeArray)
    raise
  result = CoreAIArray(owner: ownCoreAI(nativeArray))

proc requiredDtype[T: float32 | int32](): string =
  when T is float32: "float32"
  else: "int32"

proc cpu*[T: float32 | int32](array: CoreAIArray; typ: typedesc[T]): Tensor[T] =
  ## Allocate an independent row-major CPU tensor and copy completed logical values.
  ## The requested dtype must match exactly; no implicit conversion is performed.
  let handle = requireOwner(if array == nil: nil else: array.owner, "array")
  let source = descriptor(array)
  if not source.resolved:
    descriptorFailure("Cannot export a Core AI array with unresolved dimensions")
  if source.dtype != requiredDtype[T]():
    descriptorFailure("Core AI array dtype " & source.dtype & " cannot be copied as " & requiredDtype[T]())
  let elementCount = checkedShapeCount(source.shape)
  let byteCount = checkedByteCount(elementCount, sizeof(T))
  var hostShape = source.shape
  result = newTensorUninit[T](hostShape)
  var destination: AMTensorDescriptor
  destination.rank = int32(hostShape.len)
  when T is float32: destination.dtype = 1
  else: destination.dtype = 6
  destination.capacity = byteCount
  for axis in 0 ..< hostShape.len:
    destination.shape[axis] = int64(hostShape[axis])
  for axis in 0 ..< hostShape.len:
    destination.strides[axis] = int64(result.strides[axis])
  let host = cast[pointer](result.get_data_ptr)
  checkCoreAI(coreAIApi().arrayCopy(handle, host, destination))

proc submitInference*(function: CoreAIFunction; inputs: openArray[CoreAIBinding]): CoreAIJob =
  ## Validate named descriptors and submit native inference while retaining inputs.
  ## This returns a job, not host-ready output arrays.
  let functionHandle = requireOwner(if function == nil: nil else: function.owner, "function")
  if uint64(inputs.len) > uint64(high(uint32)):
    descriptorFailure("Too many Core AI inference inputs")
  var nativeInputs = newSeq[ACBinding](inputs.len)
  for index, input in inputs:
    nativeInputs[index].name = input.name.cstring
    if containsNul(input.name):
      descriptorFailure("Core AI input name contains a null byte")
    nativeInputs[index].array = requireOwner(if input.array == nil: nil else: input.array.owner, "array")
  var inputPointer: ptr ACBinding
  if nativeInputs.len != 0: inputPointer = addr nativeInputs[0]
  var nativeJob: pointer
  let status = coreAIApi().submit(functionHandle, inputPointer, uint32(nativeInputs.len), addr nativeJob)
  try:
    checkCoreAI(status)
  except:
    if nativeJob != nil: coreAIApi().release(nativeJob)
    raise
  result = CoreAIJob(owner: ownCoreAI(nativeJob), function: function)

proc newCoreAIStream*(stream: MetalStream): CoreAIStream =
  ## Create an SDK stream over the same retained native Metal command queue.
  if stream == nil or stream.owner == nil or stream.owner.handle == nil:
    raise newException(CoreAIError, "Invalid or released Metal stream")
  var queue: pointer
  var identity: uint64
  checkCoreAI(coreAIApi().streamInfo(stream.owner.handle, addr queue, addr identity))
  if queue == nil:
    raise newException(CoreAIError, "Metal stream has no native command queue")
  var nativeStream: pointer
  checkCoreAI(coreAIApi().streamCreate(stream.owner.handle, addr nativeStream))
  result = CoreAIStream(owner: ownCoreAI(nativeStream),
    metalStream: stream, streamIdentity: identity)

proc encodeInference*(function: CoreAIFunction;
                       inputs: openArray[CoreAIBinding];
                       outputViews: openArray[CoreAIBinding] = [];
                       stream: CoreAIStream): CoreAIJob =
  ## Encode on the same queue as Metal work; returns before inference completes.
  ## Native work retains arrays and the stream through SDK completion.
  let functionHandle = requireOwner(if function == nil: nil else: function.owner, "function")
  let streamHandle = requireOwner(if stream == nil: nil else: stream.owner, "stream")
  if uint64(inputs.len) > uint64(high(uint32)) or
      uint64(outputViews.len) > uint64(high(uint32)):
    descriptorFailure("Too many Core AI inference bindings")
  var nativeInputs = newSeq[ACBinding](inputs.len)
  var nativeOutputs = newSeq[ACBinding](outputViews.len)
  for index, input in inputs:
    if containsNul(input.name):
      descriptorFailure("Core AI input name contains a null byte")
    nativeInputs[index].name = input.name.cstring
    nativeInputs[index].array = requireOwner(
      if input.array == nil: nil else: input.array.owner, "array")
  for index, output in outputViews:
    if containsNul(output.name):
      descriptorFailure("Core AI output name contains a null byte")
    nativeOutputs[index].name = output.name.cstring
    nativeOutputs[index].array = requireOwner(
      if output.array == nil: nil else: output.array.owner, "array")
  var inputPointer: ptr ACBinding
  var outputPointer: ptr ACBinding
  if nativeInputs.len != 0: inputPointer = addr nativeInputs[0]
  if nativeOutputs.len != 0: outputPointer = addr nativeOutputs[0]
  var nativeJob: pointer
  let status = coreAIApi().encode(functionHandle, inputPointer,
    uint32(nativeInputs.len), outputPointer, uint32(nativeOutputs.len),
    streamHandle, addr nativeJob)
  try:
    checkCoreAI(status)
  except:
    if nativeJob != nil: coreAIApi().release(nativeJob)
    raise
  result = CoreAIJob(owner: ownCoreAI(nativeJob), function: function,
    stream: stream)

proc wait*(job: CoreAIJob; timeoutMs = 30000): seq[CoreAIBinding] =
  ## Wait for this job only and return its named SDK arrays. Zero milliseconds polls.
  ## Timeout does not cancel native work or release its pending dependencies.
  let handle = requireOwner(if job == nil: nil else: job.owner, "job")
  if timeoutMs < 0 or uint64(timeoutMs) > uint64(high(uint32)):
    descriptorFailure("Core AI wait timeout is outside the supported range")
  checkCoreAI(coreAIApi().jobWait(handle, uint32(timeoutMs)))
  var count: uint32
  checkCoreAI(coreAIApi().jobOutputCount(handle, addr count))
  result = newSeqOfCap[CoreAIBinding](int(count))
  for index in 0'u32 ..< count:
    var name: cstring
    var nativeArray: pointer
    let status = coreAIApi().jobOutput(handle, index, addr name, addr nativeArray)
    try:
      checkCoreAI(status)
      if name == nil: raise newException(CoreAIError, "Core AI returned a nil output name")
      let outputName = $name
      let outputArray = CoreAIArray(owner: ownCoreAI(nativeArray))
      nativeArray = nil
      result.add(CoreAIBinding(name: outputName, array: outputArray))
    except:
      if nativeArray != nil: coreAIApi().release(nativeArray)
      raise
    finally:
      releaseCString(name)

proc infer*(function: CoreAIFunction; inputs: openArray[CoreAIBinding]): seq[CoreAIBinding] =
  ## Submit inference and wait up to 30 seconds for its named outputs.
  let job = submitInference(function, inputs)
  result = wait(job)

proc awaitCurrentWorkCompleted*(stream: CoreAIStream;
                                timeoutMs = 30000): CoreAIEvidence {.discardable.} =
  ## Wait for accepted SDK jobs and this queue's SDK/Metal completion boundary.
  ## This is a completion fence, not tensor readback or a fully host-wait-free pipeline.
  let streamHandle = requireOwner(if stream == nil: nil else: stream.owner, "stream")
  if timeoutMs < 0 or uint64(timeoutMs) > uint64(high(uint32)):
    descriptorFailure("Core AI wait timeout is outside the supported range")
  var nativeJob: pointer
  let status = coreAIApi().streamAwait(streamHandle, addr nativeJob)
  try:
    checkCoreAI(status)
  except:
    if nativeJob != nil: coreAIApi().release(nativeJob)
    raise
  let job = CoreAIJob(owner: ownCoreAI(nativeJob), stream: stream)
  checkCoreAI(coreAIApi().jobWait(job.owner.handle, uint32(timeoutMs)))
  result = coreAIJobEvidence(job)
