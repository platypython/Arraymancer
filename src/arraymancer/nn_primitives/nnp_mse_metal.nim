# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides native float32 mean-MSE forward and backward primitives.
import ../tensor/[metal_types, init_metal, shapeshifting_metal, operators_metal, data_structure]

proc mse_forward_metal*(prediction, target: MetalTensor[float32]): MetalTensor[float32] =
  ## Reduce squared differences directly to a native scalar, without an error tensor.
  validateMetalView(prediction)
  validateMetalView(target)
  if prediction.shape != target.shape:
    raise newException(MetalError, "Metal MSE prediction and target shapes differ")
  result = newMetalTensorUninit[float32]([], prediction.storage.device)
  dispatchMetalMath(prediction, target, result, 24)

proc mse_backward_reduced_metal*(upstream, prediction, target: MetalTensor[float32];
                                target_shape: Metadata): MetalTensor[float32] =
  ## Sum broadcast contributions directly into the requested original operand shape.
  validateMetalView(upstream)
  validateMetalView(prediction)
  validateMetalView(target)
  if upstream.rank != 0 or upstream.size != 1:
    raise newException(MetalError, "Metal MSE upstream gradient must be a scalar")
  if prediction.shape != target.shape:
    raise newException(MetalError, "Metal MSE prediction and target shapes differ")

  result = newMetalTensorUninit[float32](target_shape, prediction.storage.device)
  checkMetal metalApi().mseBackward(
    upstream.storage.owner.handle, descriptor(upstream),
    prediction.storage.owner.handle, descriptor(prediction),
    target.storage.owner.handle, descriptor(target),
    result.storage.owner.handle, descriptor(result), arraymancerAppleLibDir.cstring)

proc mse_backward_metal*(upstream, prediction, target: MetalTensor[float32]): MetalTensor[float32] =
  ## Read the scalar seed on-device and return the full prediction-shaped gradient.
  mse_backward_reduced_metal(upstream, prediction, target, prediction.shape)
