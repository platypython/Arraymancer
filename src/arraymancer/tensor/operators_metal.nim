# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides GPU float32 arithmetic with checked trailing-axis broadcasting.
import ./[metal_types, init_metal, shapeshifting_metal, data_structure]

template requireMath(T: typedesc) =
  if metalDtype[T]() != 1:
    raise newException(MetalDtypeError, "Metal arithmetic requires float32")

proc dispatchMetalMath*[T](a, b: MetalTensor[T]; output: MetalTensor[T]; operation: int32;
                          scalar = 0'f32; scalarInput = false; axis = 0) =
  ## Only descriptors cross the boundary; no host arithmetic or tensor readback.
  requireMath(T)
  validateMetalView(output)
  var ah, bh: pointer
  var ad, bd: AMDescriptor
  if a.storage != nil:
    validateMetalView(a)
    if a.storage.device.registryId != output.storage.device.registryId:
      raise newException(MetalError, "Metal device mismatch")
    ah = a.storage.owner.handle
    ad = descriptor(a)
  if b.storage != nil:
    validateMetalView(b)
    if b.storage.device.registryId != output.storage.device.registryId:
      raise newException(MetalError, "Metal device mismatch")
    bh = b.storage.owner.handle
    bd = descriptor(b)
  checkMetal metalApi().math(ah, ad, bh, bd, output.storage.owner.handle, descriptor(output),
    AMMathParameters(operation: operation, scalar: scalar, scalarInput: int32(scalarInput), axis: int32(axis)),
    arraymancerAppleLibDir.cstring)

proc aligned[T](t: MetalTensor[T]; shape: Metadata): MetalTensor[T] =
  var strides: Metadata
  strides.len = shape.len
  let leading = shape.len - t.rank
  for axis in 0 ..< shape.len:
    if axis >= leading:
      let source = axis - leading
      if t.shape[source] == shape[axis]: strides[axis] = t.strides[source]
      elif t.shape[source] != 1: raise newException(MetalError, "Incompatible Metal broadcast")
  t.metalView(shape, strides, t.offset)

proc binary[T](a, b: MetalTensor[T]; operation: int32; broadcastInputs = true): MetalTensor[T] =
  ## Share one dispatch path for equal-shape arithmetic and trailing-axis broadcasting.
  requireMath(T)
  validateMetalView(a)
  validateMetalView(b)
  if not broadcastInputs:
    if a.shape != b.shape:
      raise newException(MetalError, "Metal elementwise operands must have equal shapes")
    result = newMetalTensorUninit[T](a.shape, a.storage.device)
    dispatchMetalMath(a, b, result, operation)
    return
  if a.shape == b.shape:
    result = newMetalTensorUninit[T](a.shape, a.storage.device)
    dispatchMetalMath(a, b, result, operation)
    return
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
  result = newMetalTensorUninit[T](shape, a.storage.device)
  dispatchMetalMath(aligned(a, shape), aligned(b, shape), result, operation)

proc scalarMath[T](t: MetalTensor[T]; scalar: float32; operation: int32): MetalTensor[T] =
  requireMath(T)
  validateMetalView(t)
  result = newMetalTensorUninit[T](t.shape, t.storage.device)
  dispatchMetalMath(t, default(MetalTensor[T]), result, operation, scalar, operation != 5)

proc `+.`*[T](a, b: MetalTensor[T]): MetalTensor[T] = binary(a, b, 1)
proc `-.`*[T](a, b: MetalTensor[T]): MetalTensor[T] = binary(a, b, 2)
proc `*.`*[T](a, b: MetalTensor[T]): MetalTensor[T] = binary(a, b, 3)
proc `/.`*[T](a, b: MetalTensor[T]): MetalTensor[T] = binary(a, b, 4)
proc `+`*[T](a, b: MetalTensor[T]): MetalTensor[T] =
  ## Equal-shape elementwise addition; inputs are read directly without packing.
  binary(a, b, 1, broadcastInputs = false)

proc `-`*[T](a, b: MetalTensor[T]): MetalTensor[T] =
  ## Equal-shape elementwise subtraction; inputs are read directly without packing.
  binary(a, b, 2, broadcastInputs = false)
proc `+`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] = scalarMath(a, b, 1)
proc `-`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] = scalarMath(a, b, 2)
proc `*`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] = scalarMath(a, b, 3)
proc `/`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] = scalarMath(a, b, 4)
proc `+`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] = scalarMath(b, a, 1)
proc `-`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] = scalarMath(b, a, 8)
proc `*`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] = scalarMath(b, a, 3)
proc `/`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] = scalarMath(b, a, 9)
proc `-`*[T](t: MetalTensor[T]): MetalTensor[T] = scalarMath(t, 0, 5)
proc `+.`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] = scalarMath(a, b, 1)
proc `-.`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] = scalarMath(a, b, 2)
proc `*.`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] = scalarMath(a, b, 3)
proc `/.`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] = scalarMath(a, b, 4)
proc `+.`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] = scalarMath(b, a, 1)
proc `-.`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] = scalarMath(b, a, 8)
proc `*.`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] = scalarMath(b, a, 3)
proc `/.`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] = scalarMath(b, a, 9)

proc `*`*[T](a, b: MetalTensor[T]): MetalTensor[T] =
  ## Allocate a row-major 2-D matmul result; tile and accumulate on the GPU.
  requireMath(T)
  validateMetalView(a)
  validateMetalView(b)
  if a.rank != 2 or b.rank != 2 or a.shape[1] != b.shape[0]:
    raise newException(MetalError, "Metal matmul needs compatible 2-D inputs")
  result = newMetalTensorUninit[T]([a.shape[0], b.shape[1]], a.storage.device)
  dispatchMetalMath(a, b, result, 20)

proc fill*[T](t: var MetalTensor[T]; value: T) =
  requireMath(T)
  requireMetalMutable(t)
  when T is float32:
    dispatchMetalMath(default(MetalTensor[T]), default(MetalTensor[T]), t, 6, value, true)

proc mutableBinary[T](a: var MetalTensor[T]; b: MetalTensor[T]; operation: int32; scalar: float32) =
  requireMath(T)
  requireMetalMutable(a)
  if a.shape != b.shape: raise newException(MetalError, "Metal in-place shape mismatch")
  var read = b
  if a.storage == b.storage and (a.offset != b.offset or a.strides != b.strides):
    read = b.clone
  dispatchMetalMath(a, read, a, operation, scalar)

proc `+=`*[T](a: var MetalTensor[T]; b: MetalTensor[T]) = mutableBinary(a, b, 1, 0)
proc scaledSubtract*[T](a: var MetalTensor[T]; b: MetalTensor[T]; scale: float32) =
  ## Preserve both storage identities and b's values; materialize conflicting aliases.
  mutableBinary(a, b, 7, scale)

proc zeros_like*[T](t: MetalTensor[T]): MetalTensor[T] =
  requireMath(T)
  result = newMetalTensorUninit[T](t.shape, t.storage.device)
  when T is float32: result.fill(0'f32)
proc ones_like*(t: MetalTensor[float32]): MetalTensor[float32] =
  result = newMetalTensorUninit[float32](t.shape, t.storage.device)
  result.fill(1'f32)
