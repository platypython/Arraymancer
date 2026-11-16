# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides explicit float32 native allocation and logical CPU transfers.
import ./[metal_types, data_structure, init_cpu, shapeshifting]
export metal_types

proc metalDeviceCount*(): int =
  var count: int32
  checkMetal metalApi().deviceCount(addr count)
  int(count)
proc metalIsAvailable*(): bool =
  try: result = metalDeviceCount() > 0
  except MetalUnavailableError: result = false
proc metalDevice*(index = 0): MetalDevice =
  if index < 0 or index > high(int32).int:
    raise newException(MetalUnavailableError, "Metal device index unavailable")
  var handle: pointer
  var identity: uint64
  checkMetal metalApi().device(int32(index), addr handle, addr identity)
  MetalDevice(owner: ownMetal(handle), registryId: identity)

proc requireMetalDtype[T]() =
  when T isnot float32:
    raise newException(MetalDtypeError, "Metal storage currently requires float32")

proc newMetalTensorUninit*[T: KnownSupportsCopyMem](shape: Metadata; device: MetalDevice = nil): MetalTensor[T] =
  ## Allocate shared native storage. Empty shape is scalar; zero dimensions are empty.
  requireMetalDtype[T]()
  let count = checkedMetalSize(shape)
  if count > high(int) div sizeof(T): raise newException(MetalError, "Metal byte size overflow")
  let dev = if device == nil: metalDevice() else: device
  let bytes = uint64(count * sizeof(T))
  var handle: pointer
  checkMetal metalApi().buffer(dev.owner.handle, bytes, addr handle)
  result.storage = MetalStorage(owner: ownMetal(handle), device: dev, capacity: bytes)
  result.shape = shape
  shape_to_strides(shape, rowMajor, result.strides)

proc newMetalTensorUninit*[T: KnownSupportsCopyMem](shape: varargs[int]; device: MetalDevice): MetalTensor[T] =
  if shape.len > min(LASER_MAXRANK, 8): raise newException(MetalError, "Excessive Metal rank")
  var metadata: Metadata
  metadata.len = shape.len
  for i, dim in shape: metadata[i] = dim
  newMetalTensorUninit[T](metadata, device)

proc newMetalTensorUninit*[T: KnownSupportsCopyMem](shape: varargs[int]): MetalTensor[T] =
  newMetalTensorUninit[T](shape, nil)

proc descriptor[T](t: MetalTensor[T]): AMDescriptor =
  result.rank = int32(t.rank)
  result.dtype = 1
  result.capacity = t.storage.capacity
  result.offset = int64(t.offset)
  for i in 0 ..< t.rank:
    result.shape[i] = int64(t.shape[i])
    result.strides[i] = int64(t.strides[i])

proc metal*[T: KnownSupportsCopyMem](t: Tensor[T]; device: MetalDevice = nil): MetalTensor[T] =
  ## Copy logical CPU values; no mutation or aliasing of the source.
  requireMetalDtype[T]()
  if t.storage == nil: raise newException(MetalError, "Cannot upload an uninitialized tensor sentinel")
  result = newMetalTensorUninit[T](t.shape, device)
  if t.size > 0:
    let packed = t.asContiguous
    checkMetal metalApi().transfer(result.storage.owner.handle, descriptor(result), packed.get_offset_ptr, 1)

proc cpu*[T](t: MetalTensor[T]): Tensor[T] =
  ## Wait for this storage's producer before copying logical values to the CPU.
  requireMetalDtype[T]()
  if t.storage == nil: return
  discard checkedMetalSize(t.shape)
  result = newTensorUninit[T](t.shape)
  if t.size > 0:
    checkMetal metalApi().transfer(t.storage.owner.handle, descriptor(t), result.get_offset_ptr, 0)
