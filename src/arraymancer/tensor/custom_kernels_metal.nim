# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides explicit custom Metal shader compilation, binding and dispatch.
## Shader indexing and bounds are the caller's responsibility. Bindings retain
## their native storage. Descriptors carry element strides and offset; native
## dispatch applies element width to form byte offsets. Cross-stream read/write
## conflicts require explicit producer synchronization before dispatch;
## native leases remain active until queued work completes.
import std/os
import ./[metal_types, init_metal, metal_runtime]
import ../laser/dynamic_stack_arrays

export metal_types

type
  MetalLibrary* = ref object
    ## Retains both the native library and the device it was created for.
    owner*: MetalOwner
    device*: MetalDevice
  MetalKernel* = ref object
    ## Retains its pipeline independently of the Nim library reference.
    owner*: MetalOwner
    library*: MetalLibrary
  MetalBufferAccess* = enum
    mbaRead, mbaWrite, mbaReadWrite
  MetalBufferBinding* = object
    ## The storage reference keeps the native allocation alive through dispatch.
    storage*: MetalStorage
    descriptor*: AMDescriptor
    access*: MetalBufferAccess

proc requireDevice(device: MetalDevice) {.gcsafe.} =
  if device == nil or device.owner == nil or device.owner.handle == nil:
    raise newException(MetalError, "Metal device is unavailable")

proc requireLibrary(library: MetalLibrary) {.gcsafe.} =
  if library == nil or library.owner == nil or library.owner.handle == nil:
    raise newException(MetalError, "Metal library handle is unavailable")
  requireDevice(library.device)

proc compileMetalShader*(device: MetalDevice; source: string): MetalLibrary {.gcsafe.} =
  ## Compile source using Metal's runtime compiler; diagnostics retain native detail.
  requireDevice(device)
  if source.len == 0: raise newException(MetalShaderCompileError, "Metal shader source is empty")
  var handle: pointer
  checkMetal metalApi().library(device.owner.handle, source.cstring, 0, addr handle)
  if handle == nil: raise newException(MetalShaderCompileError, "Metal compiler returned a nil library")
  result = MetalLibrary(owner: ownMetal(handle), device: device)

proc loadMetalLibrary*(device: MetalDevice; path: string): MetalLibrary {.gcsafe.} =
  ## Load a compiled metallib through Metal's native file loader.
  requireDevice(device)
  if path.len == 0: raise newException(MetalLibraryLoadError, "Metal library path is empty")
  let resolvedPath = absolutePath(path)
  var handle: pointer
  checkMetal metalApi().library(device.owner.handle, resolvedPath.cstring, 1, addr handle)
  if handle == nil: raise newException(MetalLibraryLoadError, "Metal loader returned a nil library")
  result = MetalLibrary(owner: ownMetal(handle), device: device)

proc validateNativeBinding(binding: MetalBufferBinding) =
  let d = binding.descriptor
  if d.rank < 0 or d.rank > 8:
    raise newException(MetalError, "Custom Metal binding rank is invalid")
  if ord(binding.access) < ord(low(MetalBufferAccess)) or
      ord(binding.access) > ord(high(MetalBufferAccess)):
    raise newException(MetalError, "Custom Metal binding access is invalid")
  let elementBytes = case d.dtype
    of 1, 6, 10: 4'u64
    of 2, 3, 5, 9: 2'u64
    of 13: 4'u64
    of 4, 8, 12: 1'u64
    of 7, 11, 14: 8'u64
    else: 0'u64
  if elementBytes == 0:
    raise newException(MetalDtypeError, "Custom Metal binding dtype is invalid")
  var cStride = 1'u64
  var fStride = 1'u64
  var cContiguous = true
  var fContiguous = true
  for axis in countdown(int(d.rank) - 1, 0):
    let dim = d.shape[axis]
    if dim < 0: raise newException(MetalError, "Custom Metal binding has a negative dimension")
    if dim > 1 and (d.strides[axis] < 0 or uint64(d.strides[axis]) != cStride):
      cContiguous = false
    if dim != 0 and cStride > high(uint64) div uint64(dim):
      raise newException(MetalError, "Custom Metal binding shape overflows")
    cStride *= uint64(dim)
  for axis in 0 ..< int(d.rank):
    let dim = d.shape[axis]
    if dim < 0: raise newException(MetalError, "Custom Metal binding has a negative dimension")
    if dim > 1 and (d.strides[axis] < 0 or uint64(d.strides[axis]) != fStride):
      fContiguous = false
    if dim != 0 and fStride > high(uint64) div uint64(dim):
      raise newException(MetalError, "Custom Metal binding shape overflows")
    fStride *= uint64(dim)
  if not (cContiguous or fContiguous):
    raise newException(MetalError, "Custom Metal bindings must be C- or F-contiguous")
  let count = cStride
  if d.offset < 0: raise newException(MetalError, "Custom Metal binding has a negative element offset")
  if uint64(d.offset) > high(uint64) div elementBytes or
      count > high(uint64) div elementBytes:
    raise newException(MetalError, "Custom Metal binding byte range overflows")
  let byteOffset = uint64(d.offset) * elementBytes
  let byteCount = count * elementBytes
  if d.capacity > binding.storage.capacity or byteOffset > d.capacity or
      byteCount > d.capacity - byteOffset:
    raise newException(MetalError, "Custom Metal binding exceeds native storage capacity")


proc metalKernel*(library: MetalLibrary; name: string): MetalKernel {.gcsafe.} =
  ## Resolve and retain a native compute pipeline for a named shader function.
  requireLibrary(library)
  if name.len == 0: raise newException(MetalMissingFunctionError, "Metal kernel name is empty")
  var handle: pointer
  checkMetal metalApi().kernel(library.owner.handle, name.cstring, addr handle)
  if handle == nil: raise newException(MetalMissingFunctionError, "Metal returned a nil kernel for " & name)
  result = MetalKernel(owner: ownMetal(handle), library: library)


proc metalBinding*[T](tensor: MetalTensor[T]; access: MetalBufferAccess): MetalBufferBinding =
  ## Bind a contiguous C/F layout, preserving its checked native descriptor offset.
  if tensor.storage == nil or tensor.storage.owner == nil or
      tensor.storage.owner.handle == nil or tensor.storage.device == nil:
    raise newException(MetalError, "Cannot bind a Metal tensor without native storage")
  discard checkedMetalSize(tensor.shape)
  let d = descriptor(tensor)
  result = MetalBufferBinding(storage: tensor.storage, descriptor: d, access: access)
  validateNativeBinding(result)

proc toNative(grid: array[3, int]): AMGrid =
  for dimension in grid:
    if dimension <= 0:
      raise newException(MetalError, "Metal grid and group dimensions must be positive")
  result.x = uint64(grid[0])
  result.y = uint64(grid[1])
  result.z = uint64(grid[2])


proc dispatchMetalKernel*(kernel: MetalKernel; buffers: openArray[MetalBufferBinding];
    grid, group: array[3, int]; stream: MetalStream = nil) {.gcsafe.} =
  ## Enqueue a user-authored pipeline. Raw shader indexing/bounds are caller-owned.
  ## Grid is the total thread extent, not a count of threadgroups; group is its maximum tile size.
  if kernel == nil or kernel.owner == nil or kernel.owner.handle == nil or kernel.library == nil:
    raise newException(MetalError, "Metal kernel handle is unavailable")
  requireLibrary(kernel.library)
  if buffers.len > 31:
    raise newException(MetalError, "Custom Metal dispatch supports at most 31 direct buffer bindings")
  if stream != nil:
    if stream.owner == nil or stream.owner.handle == nil or stream.device == nil:
      raise newException(MetalError, "Metal stream is unavailable")
    if stream.device.registryId != kernel.library.device.registryId:
      raise newException(MetalError, "Metal stream and kernel belong to different devices")
  for binding in buffers:
    if binding.storage == nil or binding.storage.owner == nil or
        binding.storage.owner.handle == nil or binding.storage.device == nil:
      raise newException(MetalError, "Custom Metal binding has no retained storage")
    if binding.storage.device.registryId != kernel.library.device.registryId:
      raise newException(MetalError, "Custom Metal buffer and kernel belong to different devices")
    validateNativeBinding(binding)
  let nativeGrid = toNative(grid)
  let nativeGroup = toNative(group)
  var nativeBindings {.noinit.}: array[31, AMKernelBinding]
  for index, binding in buffers:
    nativeBindings[index].buffer = binding.storage.owner.handle
    nativeBindings[index].descriptor = binding.descriptor
    nativeBindings[index].access = int32(ord(binding.access))
    nativeBindings[index].reserved = 0

  let bindingPtr = if buffers.len == 0: nil else: addr nativeBindings[0]
  let streamHandle = if stream == nil: nil else: stream.owner.handle
  checkMetal metalApi().dispatch(kernel.owner.handle, bindingPtr,
    uint32(buffers.len), nativeGrid, nativeGroup, streamHandle)
