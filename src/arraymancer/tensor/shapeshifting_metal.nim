# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides checked metadata aliases and explicit GPU layout materialization.
import ./[metal_types, init_metal, data_structure]

proc validateMetalView*[T](t: MetalTensor[T]) =
  ## Bound signed spans before addition, so danger builds cannot wrap an address.
  if t.storage == nil: raise newException(MetalError, "Uninitialized Metal storage")
  if t.shape.len != t.strides.len: raise newException(MetalError, "Metal shape/stride rank mismatch")
  let count = checkedMetalSize(t.shape)
  if count == 0: return
  let capacity = t.storage.capacity div uint64(sizeof(T))
  if t.offset < 0 or uint64(t.offset) >= capacity:
    raise newException(MetalError, "Metal offset outside storage")
  var below, above: uint64
  for axis in 0 ..< t.rank:
    let stride = t.strides[axis]
    let magnitude = if stride < 0: uint64(-(stride + 1)) + 1 else: uint64(stride)
    let steps = uint64(t.shape[axis] - 1)
    if magnitude != 0 and steps > capacity div magnitude:
      raise newException(MetalError, "Metal view span outside storage")
    let span = magnitude * steps
    if stride < 0:
      if span > uint64(t.offset) - below: raise newException(MetalError, "Metal view precedes storage")
      below += span
    else:
      if span >= capacity - uint64(t.offset) - above: raise newException(MetalError, "Metal view exceeds storage")
      above += span

proc metalView*[T](t: MetalTensor[T]; shape, strides: Metadata; offset: int): MetalTensor[T] =
  ## Share the owner; offset is absolute within its element storage.
  result = MetalTensor[T](shape: shape, strides: strides, offset: offset, storage: t.storage)
  validateMetalView(result)

proc transpose*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Reverse axes without copying data.
  result = t
  for axis in 0 ..< t.rank:
    result.shape[axis] = t.shape[t.rank - 1 - axis]
    result.strides[axis] = t.strides[t.rank - 1 - axis]
  validateMetalView(result)

proc requireMetalMutable*[T](t: MetalTensor[T]) =
  ## A sorted stride/span proof rejects overlapping and broadcast destinations.
  validateMetalView(t)
  if t.size == 0: return
  var axes: array[8, tuple[stride, extent: uint64]]
  var length = 0
  for axis in 0 ..< t.rank:
    if t.shape[axis] <= 1: continue
    let stride = t.strides[axis]
    let magnitude = if stride < 0: uint64(-(stride + 1)) + 1 else: uint64(stride)
    var index = length
    while index > 0 and axes[index - 1].stride > magnitude:
      axes[index] = axes[index - 1]
      dec index
    axes[index] = (magnitude, uint64(t.shape[axis]))
    inc length
  var span = 1'u64
  for i in 0 ..< length:
    if axes[i].stride < span: raise newException(MetalError, "Overlapping Metal mutation layout")
    let extra = axes[i].stride * (axes[i].extent - 1)
    if extra > high(uint64) - span: raise newException(MetalError, "Metal mutation span overflow")
    span += extra

proc asContiguous*[T: KnownSupportsCopyMem](t: MetalTensor[T]; layout = rowMajor; force = false): MetalTensor[T] =
  ## Keep aliases already in the requested ordering; otherwise copy on the GPU.
  validateMetalView(t)
  let matches = if layout == rowMajor: t.is_C_contiguous else: t.is_F_contiguous
  if matches and not force: return t
  result = newMetalTensorUninit[T](t.shape, t.storage.device)
  shape_to_strides(t.shape, layout, result.strides)
  checkMetal metalApi().copy(t.storage.owner.handle, descriptor(t),
    result.storage.owner.handle, descriptor(result), arraymancerAppleLibDir.cstring)
