# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides checked metadata aliases and explicit GPU layout materialization.
import ./[metal_types, init_metal, data_structure]
import ./private/p_shapeshifting

proc checkedMetalShapeSize(shape: Metadata): int =
  if shape.len < 0 or shape.len > min(LASER_MAXRANK, 8):
    raise newException(MetalError, "Invalid or excessive Metal rank")
  checkedMetalSize(shape)


proc validateMetalView*[T](t: MetalTensor[T]) =
  ## Bound signed spans before addition, so danger builds cannot wrap an address.
  if t.storage == nil: raise newException(MetalError, "Uninitialized Metal storage")
  if t.shape.len != t.strides.len: raise newException(MetalError, "Metal shape/stride rank mismatch")
  let count = checkedMetalShapeSize(t.shape)
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
  validateMetalView(t)
  result = t
  for axis in 0 ..< t.rank:
    result.shape[axis] = t.shape[t.rank - 1 - axis]
    result.strides[axis] = t.strides[t.rank - 1 - axis]
  validateMetalView(result)

proc transpose*[T](t: MetalTensor[T]; dim1, dim2: int): MetalTensor[T] =
  ## Return a metadata alias with two axes exchanged; native storage is shared.
  validateMetalView(t)
  if dim1 < 0 or dim1 >= t.rank or dim2 < 0 or dim2 >= t.rank:
    raise newException(MetalError, "Metal transpose dimensions out of range")
  result = t
  swap(result.shape[dim1], result.shape[dim2])
  swap(result.strides[dim1], result.strides[dim2])
  validateMetalView(result)

proc transpose2d*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Return a metadata alias transposing the last two axes; rank must be at least 2.
  validateMetalView(t)
  if t.rank < 2:
    raise newException(MetalError, "Metal transpose2d requires rank at least 2")
  t.transpose(t.rank - 2, t.rank - 1)

proc asContiguous*[T: KnownSupportsCopyMem](t: MetalTensor[T]; layout = rowMajor; force = false): MetalTensor[T] =
  ## Keep aliases already in the requested ordering; otherwise copy on the GPU.
  validateMetalView(t)
  let matches = if layout == rowMajor: t.is_C_contiguous else: t.is_F_contiguous
  if matches and not force: return t
  result = newMetalTensorUninit[T](t.shape, t.storage.device)
  shape_to_strides(t.shape, layout, result.strides)
  checkMetal metalApi().copy(t.storage.owner.handle, descriptor(t),
    result.storage.owner.handle, descriptor(result), arraymancerAppleLibDir.cstring)

proc reshape*[T: KnownSupportsCopyMem](t: MetalTensor[T]; shape: Metadata): MetalTensor[T] =
  ## Reshape by aliasing contiguous storage, or materialize once on the GPU first.
  ## The returned tensor always shares the native owner with its packed source.
  validateMetalView(t)
  let requestedSize = checkedMetalShapeSize(shape)
  if requestedSize != checkedMetalSize(t.shape):
    raise newException(MetalError, "Metal reshape changes the number of elements")
  if t.size == 0 or t.is_C_contiguous:
    result = t
    reshape_no_copy(t, shape, result, rowMajor)
  else:
    let packed = t.asContiguous(rowMajor)
    result = packed
    reshape_no_copy(packed, shape, result, rowMajor)
  validateMetalView(result)

proc reshape*[T: KnownSupportsCopyMem](t: MetalTensor[T]; shape: varargs[int]): MetalTensor[T] =
  ## Varargs form of Metal reshape; data is aliased when possible and otherwise
  ## packed once on the GPU.
  if shape.len > min(LASER_MAXRANK, 8):
    raise newException(MetalError, "Excessive Metal rank")
  var metadata: Metadata
  metadata.len = shape.len
  for axis, dim in shape:
    metadata[axis] = dim
  t.reshape(metadata)

proc broadcast*[T](t: MetalTensor[T]; shape: Metadata): MetalTensor[T] =
  ## Broadcast by sharing the native owner and adding zero-stride metadata.
  ## Missing leading axes align from the right; mutating the returned alias is unsafe.
  validateMetalView(t)
  discard checkedMetalShapeSize(shape)
  if t.rank > shape.len:
    raise newException(MetalError, "Metal broadcast target rank is too small")
  let leading = shape.len - t.rank
  for axis in 0 ..< t.rank:
    let targetDim = shape[leading + axis]
    if t.shape[axis] != targetDim and t.shape[axis] != 1:
      raise newException(MetalError, "Incompatible Metal broadcast shape")

  var target: MetalTensor[T]
  target.shape = shape
  target.strides.len = shape.len
  var expanded: tuple[a, b: MetalTensor[T]]
  broadcast2Impl(t, target, expanded)
  if expanded.a.shape != shape:
    raise newException(MetalError, "Incompatible Metal broadcast shape")
  result = t
  result.shape = expanded.a.shape
  result.strides = expanded.a.strides
  result.offset = expanded.a.offset
  validateMetalView(result)

proc broadcast*[T](t: MetalTensor[T]; shape: varargs[int]): MetalTensor[T] =
  ## Varargs form of leading-axis Metal broadcast; the result is a no-copy alias.
  if shape.len > min(LASER_MAXRANK, 8):
    raise newException(MetalError, "Excessive Metal rank")
  var metadata: Metadata
  metadata.len = shape.len
  for axis, dim in shape:
    metadata[axis] = dim
  t.broadcast(metadata)

proc broadcast2*[T](a, b: MetalTensor[T]): tuple[a, b: MetalTensor[T]] =
  ## Align trailing axes as shared zero-stride views, including scalar and empty inputs.
  validateMetalView(a)
  validateMetalView(b)
  var shape: Metadata
  shape.len = max(a.rank, b.rank)
  for axis in 0 ..< shape.len:
    let ai = axis - (shape.len - a.rank)
    let bi = axis - (shape.len - b.rank)
    let x = if ai < 0: 1 else: a.shape[ai]
    let y = if bi < 0: 1 else: b.shape[bi]
    if x == y: shape[axis] = x
    elif x == 1: shape[axis] = y
    elif y == 1: shape[axis] = x
    else: raise newException(MetalError, "Incompatible Metal broadcast")
  (a.broadcast(shape), b.broadcast(shape))

proc squeeze*[T](t: MetalTensor[T]; axis: int): MetalTensor[T] =
  ## Remove a singleton axis without copying; a one-element vector may become scalar.
  validateMetalView(t)
  if axis < 0 or axis >= t.rank:
    raise newException(MetalError, "Metal squeeze axis out of range")
  result = t
  if result.shape[axis] == 1:
    result.shape.delete(axis)
    result.strides.delete(axis)
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

