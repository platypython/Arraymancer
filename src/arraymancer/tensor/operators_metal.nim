# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides GPU float32 arithmetic with checked trailing-axis broadcasting.
import ./[metal_types, init_metal, shapeshifting_metal, data_structure]
import ./private/[p_matmul, p_init_metal]

template requireMath(T: typedesc) =
  # The float32 tag is fixed by its type; keep unsupported-type errors lazy.
  when T isnot float32:
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
    initMetalTensorUninit[T](result, a.shape, a.storage.device)
    dispatchValidatedMetalMath(a, b, result, operation)
    return
  if a.shape == b.shape:
    initMetalTensorUninit[T](result, a.shape, a.storage.device)
    dispatchValidatedMetalMath(a, b, result, operation)
    return
  let (left, right) = broadcast2(a, b)
  initMetalTensorUninit[T](result, left.shape, a.storage.device)
  dispatchValidatedMetalMath(left, right, result, operation)

template scalarMath[T](t: MetalTensor[T]; scalarValue: float32; operationCode: int32; output: var MetalTensor[T]) =
  ## Avoid a sentinel tensor and the binary helper's conditional descriptor path.
  requireMath(T)
  validateMetalView(t)
  initMetalTensorUninit[T](output, t.shape, t.storage.device)
  checkMetal metalApi().math(t.storage.owner.handle, descriptor(t), nil, default(AMDescriptor),
    output.storage.owner.handle, descriptor(output),
    AMMathParameters(operation: operationCode, scalar: scalarValue, scalarInput: int32(operationCode != 5)),
    arraymancerAppleLibDir.cstring)

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
proc `+`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] =
  scalarMath(a, b, 1, result)
proc `-`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] =
  scalarMath(a, b, 2, result)
proc `*`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] =
  scalarMath(a, b, 3, result)
proc `/`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] =
  scalarMath(a, b, 4, result)
proc `+`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] =
  scalarMath(b, a, 1, result)
proc `-`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] =
  scalarMath(b, a, 8, result)
proc `*`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] =
  scalarMath(b, a, 3, result)
proc `/`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] =
  scalarMath(b, a, 9, result)
proc `-`*[T](t: MetalTensor[T]): MetalTensor[T] =
  scalarMath(t, 0, 5, result)
proc `+.`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] =
  scalarMath(a, b, 1, result)
proc `-.`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] =
  scalarMath(a, b, 2, result)
proc `*.`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] =
  scalarMath(a, b, 3, result)
proc `/.`*[T](a: MetalTensor[T]; b: float32): MetalTensor[T] =
  scalarMath(a, b, 4, result)
proc `+.`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] =
  scalarMath(b, a, 1, result)
proc `-.`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] =
  scalarMath(b, a, 8, result)
proc `*.`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] =
  scalarMath(b, a, 3, result)
proc `/.`*[T](a: float32; b: MetalTensor[T]): MetalTensor[T] =
  scalarMath(b, a, 9, result)

proc exp*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Element-wise float32 exponential; preserves scalar, empty and strided shapes.
  scalarMath(t, 0, 10, result)

proc ln*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Element-wise float32 natural logarithm with IEEE domain results.
  scalarMath(t, 0, 11, result)

proc sqrt*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Element-wise float32 square root with IEEE domain results.
  scalarMath(t, 0, 12, result)

proc tanh*[T](t: MetalTensor[T]): MetalTensor[T] =
  ## Element-wise float32 hyperbolic tangent; preserves scalar, empty and strided shapes.
  scalarMath(t, 0, 14, result)

proc matmulBatched[T](a, b: MetalTensor[T]; result: var MetalTensor[T]) =
  ## Keep promoted ownership roots off the matrix/vector return path.
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
  initMetalTensorUninit[T](result, outShape, a.storage.device)
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
  if a.rank <= 2 and b.rank <= 2:
    let leftVector = a.rank == 1
    let rightVector = b.rank == 1
    let M = if leftVector: 1 else: a.shape[0]
    let K = a.shape[a.rank - 1]
    let rightK = b.shape[0]
    let N = if rightVector: 1 else: b.shape[1]
    if K != rightK:
      raise newException(MetalError, "Metal matrix inner dimensions differ")

    var outShape: Metadata
    if leftVector and rightVector:
      discard
    elif leftVector:
      outShape.add(N)
    elif rightVector:
      outShape.add(M)
    else:
      outShape.add(M)
      outShape.add(N)
    initMetalTensorUninit[T](result, outShape, a.storage.device)

    var ad = default(AMDescriptor)
    ad.rank = 2
    ad.dtype = metalDtype[T]()
    ad.shape[0] = int64(M)
    ad.shape[1] = int64(K)
    ad.strides[0] = if leftVector: 0 else: int64(a.strides[0])
    ad.strides[1] = int64(a.strides[a.rank - 1])
    ad.offset = int64(a.offset)
    ad.capacity = a.storage.capacity

    var bd = default(AMDescriptor)
    bd.rank = 2
    bd.dtype = metalDtype[T]()
    bd.shape[0] = int64(K)
    bd.shape[1] = int64(N)
    bd.strides[0] = int64(b.strides[0])
    bd.strides[1] = if rightVector: 0 else: int64(b.strides[1])
    bd.offset = int64(b.offset)
    bd.capacity = b.storage.capacity

    var od = default(AMDescriptor)
    od.rank = 2
    od.dtype = metalDtype[T]()
    od.shape[0] = int64(M)
    od.shape[1] = int64(N)
    od.strides[0] = int64(N)
    od.strides[1] = 1
    od.offset = 0
    od.capacity = result.storage.capacity
    checkMetal metalApi().math(a.storage.owner.handle, ad, b.storage.owner.handle, bd,
      result.storage.owner.handle, od, AMMathParameters(operation: 20),
      arraymancerAppleLibDir.cstring)
    return
  matmulBatched(a, b, result)

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
  initMetalTensorUninit[T](result, t.shape, t.storage.device)
  when T is float32: result.fill(0'f32)
proc ones_like*(t: MetalTensor[float32]): MetalTensor[float32] =
  initMetalTensorUninit[float32](result, t.shape, t.storage.device)
  result.fill(1'f32)
