# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides GPU float32 arithmetic with checked trailing-axis broadcasting.
import ./[metal_types, init_metal, shapeshifting_metal, data_structure]
import ./private/p_matmul

template requireMath(T: typedesc) =
  if metalDtype[T]() != 1:
    raise newException(MetalDtypeError, "Metal arithmetic requires float32")

proc dispatchValidatedMetalMath[T](a, b: MetalTensor[T]; output: MetalTensor[T]; operation: int32;
                                   scalar = 0'f32; scalarInput = false; axis = 0) =
  ## Internal callers already checked inputs and constructed the owning output.
  ## Native admission still validates every descriptor and device before encoding.
  var ah, bh: pointer
  var ad, bd: AMDescriptor
  if a.storage != nil:
    ah = a.storage.owner.handle
    ad = descriptor(a)
  if b.storage != nil:
    bh = b.storage.owner.handle
    bd = descriptor(b)
  checkMetal metalApi().math(ah, ad, bh, bd, output.storage.owner.handle, descriptor(output),
    AMMathParameters(operation: operation, scalar: scalar, scalarInput: int32(scalarInput), axis: int32(axis)),
    arraymancerAppleLibDir.cstring)

proc dispatchMetalMath*[T](a, b: MetalTensor[T]; output: MetalTensor[T]; operation: int32;
                          scalar = 0'f32; scalarInput = false; axis = 0) =
  ## Only descriptors cross the boundary; no host arithmetic or tensor readback.
  requireMath(T)
  validateMetalView(output)
  if a.storage != nil:
    validateMetalView(a)
    if a.storage.device.registryId != output.storage.device.registryId:
      raise newException(MetalError, "Metal device mismatch")
  if b.storage != nil:
    validateMetalView(b)
    if b.storage.device.registryId != output.storage.device.registryId:
      raise newException(MetalError, "Metal device mismatch")
  dispatchValidatedMetalMath(a, b, output, operation, scalar, scalarInput, axis)

proc binary[T](a, b: MetalTensor[T]; operation: int32; broadcastInputs = true): MetalTensor[T] =
  ## Share one dispatch path for equal-shape arithmetic and trailing-axis broadcasting.
  requireMath(T)
  validateMetalView(a)
  validateMetalView(b)
  if a.storage.device.registryId != b.storage.device.registryId:
    raise newException(MetalError, "Metal device mismatch")
  if not broadcastInputs:
    if a.shape != b.shape:
      raise newException(MetalError, "Metal elementwise operands must have equal shapes")
    result = newMetalTensorUninit[T](a.shape, a.storage.device)
    dispatchValidatedMetalMath(a, b, result, operation)
    return
  if a.shape == b.shape:
    result = newMetalTensorUninit[T](a.shape, a.storage.device)
    dispatchValidatedMetalMath(a, b, result, operation)
    return
  let (left, right) = broadcast2(a, b)
  result = newMetalTensorUninit[T](left.shape, a.storage.device)
  dispatchValidatedMetalMath(left, right, result, operation)

proc scalarMath[T](t: MetalTensor[T]; scalar: float32; operation: int32): MetalTensor[T] =
  requireMath(T)
  validateMetalView(t)
  result = newMetalTensorUninit[T](t.shape, t.storage.device)
  dispatchValidatedMetalMath(t, default(MetalTensor[T]), result, operation, scalar, operation != 5)

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

proc exp*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Element-wise float32 exponential; preserves scalar, empty and strided shapes.
  scalarMath(t, 0, 10)

proc ln*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Element-wise float32 natural logarithm with IEEE domain results.
  scalarMath(t, 0, 11)

proc sqrt*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Element-wise float32 square root with IEEE domain results.
  scalarMath(t, 0, 12)

proc tanh*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Element-wise float32 hyperbolic tangent; preserves scalar, empty and strided shapes.
  scalarMath(t, 0, 14)

proc `*`*[T](a, b: MetalTensor[T]): MetalTensor[T] =
  ## Float32 matrix product with vector promotion and trailing-axis batch broadcasting.
  ## The result owns one GPU allocation; all batches share one native dispatch.
  ## Two vectors return a rank-zero dot product; empty contractions produce zeros.
  requireMath(T)
  validateMetalView(a)
  validateMetalView(b)
  if a.storage.device.registryId != b.storage.device.registryId:
    raise newException(MetalError, "Metal device mismatch")
  if a.rank < 1 or b.rank < 1:
    raise newException(MetalError, "Metal matmul requires operands of rank >= 1")
  let left = promoteMatmulOperand(a, false)
  let right = promoteMatmulOperand(b, true)
  let M = left.shape[left.rank - 2]
  let K = left.shape[left.rank - 1]
  let N = right.shape[right.rank - 1]
  if K != right.shape[right.rank - 2]:
    raise newException(MetalError, "Metal matrix inner dimensions differ")
  let aBatchRank = left.rank - 2
  let bBatchRank = right.rank - 2
  let batchRank = max(aBatchRank, bBatchRank)
  var batchShape: Metadata
  batchShape.len = batchRank
  for i in 0 ..< batchRank:
    let ai = i - (batchRank - aBatchRank)
    let bi = i - (batchRank - bBatchRank)
    let ad = if ai < 0: 1 else: left.shape[ai]
    let bd = if bi < 0: 1 else: right.shape[bi]
    if ad == bd: batchShape[i] = ad
    elif ad == 1: batchShape[i] = bd
    elif bd == 1: batchShape[i] = ad
    else: raise newException(MetalError, "Incompatible Metal matrix batch dimensions")
  var outShape = batchShape
  outShape.add(M)
  outShape.add(N)
  result = newMetalTensorUninit[T](outShape, a.storage.device)
  if batchRank == 0:
    dispatchValidatedMetalMath(left, right, result, 20)
  else:
    var leftShape = outShape
    var rightShape = outShape
    leftShape[batchRank + 1] = K
    rightShape[batchRank] = K
    var leftStrides, rightStrides: Metadata
    leftStrides.len = batchRank + 2
    rightStrides.len = batchRank + 2
    for axis in 0 ..< batchRank:
      let ai = axis - (batchRank - aBatchRank)
      let bi = axis - (batchRank - bBatchRank)
      if ai >= 0 and left.shape[ai] != 1: leftStrides[axis] = left.strides[ai]
      if bi >= 0 and right.shape[bi] != 1: rightStrides[axis] = right.strides[bi]
    leftStrides[batchRank] = left.strides[left.rank - 2]
    leftStrides[batchRank + 1] = left.strides[left.rank - 1]
    rightStrides[batchRank] = right.strides[right.rank - 2]
    rightStrides[batchRank + 1] = right.strides[right.rank - 1]
    let leftView = left.metalView(leftShape, leftStrides, left.offset)
    let rightView = right.metalView(rightShape, rightStrides, right.offset)
    dispatchValidatedMetalMath(leftView, rightView, result, 20)
  if a.rank == 1:
    result.shape.delete(result.rank - 2)
    result.strides.delete(result.strides.len - 2)
  if b.rank == 1:
    result.shape.delete(result.rank - 1)
    result.strides.delete(result.strides.len - 1)

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
