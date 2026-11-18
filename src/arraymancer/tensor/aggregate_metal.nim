# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides deterministic GPU reductions with GPU-only partial buffers.
import ./[metal_types, init_metal, shapeshifting_metal, operators_metal, data_structure]

proc reduce(t: MetalTensor[float32]; normalize: bool;
            multiply = false): MetalTensor[float32] =
  ## Normalize in the final GPU pass to avoid allocating a second scalar.
  validateMetalView(t)
  var input = t
  while true:
    let groups = if input.size <= 65536: 1 else:
      input.size div 256 + int(input.size mod 256 != 0)
    var shape: Metadata
    if groups != 1:
      shape.len = 1
      shape[0] = groups
    let output = newMetalTensorUninit[float32](shape, t.storage.device)
    dispatchMetalMath(input, default(MetalTensor[float32]), output,
      (if multiply: 25'i32 else: 21'i32),
      scalar = float32(t.size), scalarInput = normalize and groups == 1)
    if groups == 1: return output
    input = output

proc sum*(t: MetalTensor[float32]): MetalTensor[float32] =
  ## Full reduction returns a native scalar. Empty input sums to zero.
  reduce(t, false)

proc mean*(t: MetalTensor[float32]): MetalTensor[float32] =
  ## Empty mean is a native scalar NaN (GPU zero divided by zero).
  reduce(t, true)

proc prod*(t: MetalTensor[float32]): MetalTensor[float32] =
  ## Full product returns an independent native scalar; empty products are one.
  ## Packed groups combine adjacent factors before reducing across GPU lanes;
  ## floating-point grouping can differ from host reductions.
  reduce(t, false, true)

proc reduceAxis(t: MetalTensor[float32]; axis: int;
                operation: int32): MetalTensor[float32] =
  ## Resolve scalar and negative axes before creating the output descriptor.
  validateMetalView(t)
  let rank = max(t.rank, 1)
  if axis < -rank or axis >= rank:
    raise newException(MetalError, "Metal reduction axis outside rank")
  if t.rank == 0:
    return reduce(t, operation == 27, operation == 26)
  let resolved = if axis < 0: axis + rank else: axis
  var shape = t.shape
  shape[resolved] = 1
  result = newMetalTensorUninit[float32](shape, t.storage.device)
  dispatchMetalMath(t, default(MetalTensor[float32]), result,
    operation, axis = resolved)

proc sum*(t: MetalTensor[float32]; axis: int): MetalTensor[float32] =
  ## Keep the reduced dimension at length one; accept negative and scalar axes.
  ## Empty-axis sums are zero; unreduced empty dimensions stay empty.
  reduceAxis(t, axis, 22)

proc prod*(t: MetalTensor[float32]; axis: int): MetalTensor[float32] =
  ## Allocate a keepdim product; empty axes produce one. Accept negative axes.
  ## Packed rows combine adjacent factors within SIMD subgroups.
  reduceAxis(t, axis, 26)

proc mean*(t: MetalTensor[float32]; axis: int): MetalTensor[float32] =
  ## Allocate a keepdim mean; empty axes produce NaN. Accept negative axes.
  reduceAxis(t, axis, 27)

proc productGradient*(input, product, seed: MetalTensor[float32];
                      axis: int): MetalTensor[float32] =
  ## Return the product cotangent on the input device without materializing reads.
  validateMetalView(input)
  validateMetalView(product)
  validateMetalView(seed)
  if input.storage.device.registryId != product.storage.device.registryId or
      input.storage.device.registryId != seed.storage.device.registryId:
    raise newException(MetalError, "Metal device mismatch")
  let rank = max(input.rank, 1)
  if axis < -1 or axis >= rank:
    raise newException(MetalError, "Metal product-gradient axis outside rank")

  result = newMetalTensorUninit[float32](input.shape, input.storage.device)
  if input.size == 0: return
  checkMetal metalApi().productGradient(
    input.storage.owner.handle, descriptor(input),
    product.storage.owner.handle, descriptor(product),
    seed.storage.owner.handle, descriptor(seed),
    result.storage.owner.handle, descriptor(result),
    int32(axis), arraymancerAppleLibDir.cstring)
