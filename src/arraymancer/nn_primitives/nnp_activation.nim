# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import std / math
import  ../tensor,
        ./private/p_activation

## Provides tensor activation functions and cached-output derivatives.

# ##################################################################################################
# Forward

proc sigmoid*[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noinit.}=
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value
  ## Please use sigmoid_cross_entropy for the final layer for better stability and performance

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)

  result = map_inline(t):
    sigmoid(x)

proc relu*[T](t: Tensor[T]): Tensor[T] {.noinit.}=
  ## Allocate a ReLU result, preserving NaN, signed zero and input shape.
  t.map_inline relu(x)

proc tanh*[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noinit.}=
  t.map_inline tanh(x)

proc gelu*[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noinit.}=
  ## Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016),
  ## :math:`f(x) = x/2 (1 + erf(x/sqrt(2)))`, matching PyTorch's default GELU.
  result = map_inline(t):
    0.5.T * x * (1.T + erf(x / sqrt(2.T)))

proc quick_gelu*[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noinit.}=
  ## Sigmoid approximation to GELU (OpenAI CLIP), :math:`f(x) = x \sigma(1.702 x)`.
  result = map_inline(t):
    x * sigmoid(1.702.T * x)

# ##################################################################################################
# In-place forward

proc msigmoid*[T: SomeFloat](t: var Tensor[T]) =
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)
  apply_inline(t):
    sigmoid(x)

proc mrelu*[T](t: var Tensor[T]) =
  ## Zero negative elements in place, preserving NaN and signed zero.
  t.apply_inline relu(x)

proc mtanh*[T: SomeFloat](t: var Tensor[T]) =
  t.apply_inline tanh(x)

# ##################################################################################################
# Backward

proc sigmoid_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.noinit.}=
  result = map2_inline(cached_tensor, gradient):
    x * (1 - x) * y

proc relu_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.noinit.}=
  result = map2_inline(cached_tensor, gradient):
    if x <= 0.T:
      0.T
    else:
      y

proc tanh_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.noinit.}=
  result = map2_inline(cached_tensor, gradient):
    y * (1 - x * x)

proc gelu_backward*[T: SomeFloat](gradient: Tensor[T], cached_input: Tensor[T]): Tensor[T]{.noinit.}=
  ## GELU derivative, :math:`f'(x) = 1/2(1 + erf(x/\sqrt{2})) + x e^{-x^2/2}/\sqrt{2\pi}`.
  result = map2_inline(cached_input, gradient):
    y * (0.5.T * (1.T + erf(x / sqrt(2.T))) + x * exp(-0.5.T * x * x) / sqrt(2.T * T(PI)))

proc quick_gelu_backward*[T: SomeFloat](gradient: Tensor[T], cached_input: Tensor[T]): Tensor[T]{.noinit.}=
  ## QuickGELU derivative,
  ## :math:`f'(x) = \sigma(1.702 x) (1 + 1.702 x (1 - \sigma(1.702 x)))`.
  result = map2_inline(cached_input, gradient):
    let s = sigmoid(1.702.T * x)
    y * s * (1.T + 1.702.T * x * (1.T - s))

when defined(metal) or defined(coreai):
  template requireMetalActivation(T: typedesc) =
    if metalDtype[T]() != 1:
      raise newException(MetalDtypeError, "Metal activations require float32")

  proc metalActivation[T](t: MetalTensor[T]; operation: int32): MetalTensor[T] =
    ## Validate before allocating; the kernel reads arbitrary supported views.
    requireMetalActivation(T)
    validateMetalView(t)
    result = newMetalTensorUninit[T](t.shape, t.storage.device)
    dispatchMetalMath(t, default(MetalTensor[T]), result, operation, scalarInput = true)

  proc metalActivationBackward[T](gradient, cached_tensor: MetalTensor[T];
                                  operation: int32): MetalTensor[T] =
    ## Fuse the derivative with its upstream gradient instead of allocating intermediates.
    requireMetalActivation(T)
    validateMetalView(gradient)
    validateMetalView(cached_tensor)
    if gradient.shape != cached_tensor.shape:
      raise newException(MetalError, "Metal activation gradients require equal shapes")
    if gradient.storage.device.registryId != cached_tensor.storage.device.registryId:
      raise newException(MetalError, "Metal device mismatch")
    result = newMetalTensorUninit[T](gradient.shape, gradient.storage.device)
    dispatchMetalMath(gradient, cached_tensor, result, operation)

  proc metalActivationInplace[T](t: var MetalTensor[T]; operation: int32) =
    ## Each thread must own its destination, including reversed strided views.
    requireMetalActivation(T)
    requireMetalMutable(t)
    dispatchMetalMath(t, default(MetalTensor[T]), t, operation, scalarInput = true)

  proc sigmoid*[T](t: MetalTensor[T]): MetalTensor[T] =
    ## Float32 sigmoid with one GPU allocation; preserves scalar, empty and strided shapes.
    metalActivation(t, 16)

  proc relu*[T](t: MetalTensor[T]): MetalTensor[T] =
    ## Float32 ReLU with one GPU allocation; preserves shape, NaN and signed zero.
    metalActivation(t, 18)

  proc msigmoid*[T](t: var MetalTensor[T]) =
    ## Mutate a non-overlapping float32 Metal view without reallocating.
    metalActivationInplace(t, 16)

  proc mrelu*[T](t: var MetalTensor[T]) =
    ## Mutate a non-overlapping float32 Metal view, preserving NaN and signed zero.
    metalActivationInplace(t, 18)

  proc mtanh*[T](t: var MetalTensor[T]) =
    ## Mutate a non-overlapping float32 Metal view without reallocating.
    metalActivationInplace(t, 14)

  proc sigmoid_backward*[T](gradient, cached_tensor: MetalTensor[T]): MetalTensor[T] =
    ## Allocate a float32 VJP from equal-shaped gradients and cached sigmoid outputs.
    metalActivationBackward(gradient, cached_tensor, 17)

  proc relu_backward*[T](gradient, cached_tensor: MetalTensor[T]): MetalTensor[T] =
    ## Allocate a float32 VJP from equal-shaped gradients and cached ReLU outputs.
    metalActivationBackward(gradient, cached_tensor, 19)

  proc tanh_backward*[T](gradient, cached_tensor: MetalTensor[T]): MetalTensor[T] =
    ## Allocate a float32 VJP from equal-shaped gradients and cached tanh outputs.
    metalActivationBackward(gradient, cached_tensor, 15)

# ####################################################################################################
# Documentation

# Sigmoid implementation doc:
# 1. Canonical sigmoid "f(x) = 1 / (1 + exp(-x))" is unstable
# for negative values < 709 (for float64)
#
# 2. Alternative expression stable for negative but unstable for positive is
# "f(x) = exp(x) / (1 + exp(x))"
#
# 3. Introducing branching would be very costly.
#
# 4. Using tanh as 0.5 * (tanh(0.5 * x) + 1) is better than branching
# but slow as well
#
# 5. Another alternative would be to clip x to max (-500, x) to avoid this instability
#
# Benchmarks available in the benchmark folder
#
