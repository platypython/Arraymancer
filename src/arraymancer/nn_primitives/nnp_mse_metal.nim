# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides native float32 mean-MSE forward and backward primitives.
import ../tensor/[metal_types, init_metal, shapeshifting_metal, operators_metal, aggregate_metal]

proc mse_forward_metal*(prediction, target: MetalTensor[float32]): MetalTensor[float32] =
  ## Fuse subtraction and squaring on the GPU, then reduce to a scalar mean.
  validateMetalView(prediction)
  validateMetalView(target)
  if prediction.shape != target.shape:
    raise newException(MetalError, "Metal MSE prediction and target shapes differ")
  let squaredDifference = newMetalTensorUninit[float32](prediction.shape, prediction.storage.device)
  dispatchMetalMath(prediction, target, squaredDifference, 23)
  mean(squaredDifference)

proc mse_backward_metal*(upstream, prediction, target: MetalTensor[float32]): MetalTensor[float32] =
  ## Read the scalar upstream gradient in a fused GPU kernel; never materialize it on host.
  validateMetalView(upstream)
  validateMetalView(prediction)
  validateMetalView(target)
  if upstream.rank != 0 or upstream.size != 1:
    raise newException(MetalError, "Metal MSE upstream gradient must be a scalar")
  if prediction.shape != target.shape:
    raise newException(MetalError, "Metal MSE prediction and target shapes differ")

  result = newMetalTensorUninit[float32](prediction.shape, prediction.storage.device)
  checkMetal metalApi().mseBackward(
    upstream.storage.owner.handle, descriptor(upstream),
    prediction.storage.owner.handle, descriptor(prediction),
    target.storage.owner.handle, descriptor(target),
    result.storage.owner.handle, descriptor(result), arraymancerAppleLibDir.cstring)
