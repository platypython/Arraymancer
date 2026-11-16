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

import  ../../tensor/higher_order_applymap,
        ../../tensor,
        ../../autograd,
        math

# ############################################################
#
#             SGD: Stochastic Gradient Descent
#
# ############################################################

type
  SGD*[TT] = object
    ## Stochastic gradient descent without momentum.
    params*: seq[Variable[TT]]
    lr*: TT.T # Learning rate. T is the generic parameter of Tensor[T]

proc newSGD*[T](params: varargs[Variable[Tensor[T]]], learning_rate: T): SGD[Tensor[T]] {.deprecated: "Use the optimizer macro instead".}=
  SGD[Tensor[T]](params: @params, lr: learning_rate)

proc update*(self: SGD) =
  ## Performs an optimization update.
  ##
  ## Parameters:
  ## - ``self`` A SGD optimizer to update.
  ##
  ## This proc will update the weights in the model associated with the input
  ## optimizer according to the following rule:
  ##    `w = w - lr * gradient`

  # Update the params with formula Value -= lr * gradient
  # Note: SGD expects gradient to be scaled by batchsize (done by default in Arraymancer)
  for v in self.params:
    # v.value -= learning rate * grad
    if v.requires_grad:
      apply2_inline(v.value, v.grad):
        x - self.lr * y
      # Zero the gradient
      v.grad = v.value.zeros_like # TODO "setZero" instead of a new allocation

func collectSgdParams[TT](params: var seq[Variable[TT]]; layer: auto) =
  ## Stop at variables so backend mismatches cannot disappear into model fields.
  when layer is Variable[TT]:
    if not layer.isNil: params.add(layer)
  elif layer is Variable:
    raise newException(ValueError, "SGD model contains a different tensor backend or dtype")
  elif layer is seq or layer is array:
    for item in layer: collectSgdParams(params, item)
  elif layer is object or layer is tuple:
    for field in fields(layer): collectSgdParams(params, field)

func optimizer*[M, T](
        model: M,
        OptimizerKind: typedesc[SGD],
        learning_rate: T): SGD[Tensor[T]] =
  ## Create a SGD optimizer that will update the model weight
  ##
  ## Parameters:
  ## - ``model`` Model to optimize.
  ## - ``learning_rate`` Learning rate.
  ##
  ## Returns:
  ## - A SGD optimizer with the given learning rate.

  result.params = @[]
  result.lr = learning_rate

  collectSgdParams(result.params, model)

when defined(metal) or defined(coreai):
  func optimizer*[M, TT](model: M; OptimizerKind: typedesc[SGD];
                         learning_rate: float32; tensorType: typedesc[TT]): SGD[TT] =
    ## Select native float32 Metal SGD explicitly; all collected variables must match.
    when TT isnot MetalTensor[float32]:
      {.error: "Explicit native SGD requires MetalTensor[float32]".}
    else:
      result.lr = learning_rate
      collectSgdParams(result.params, model)

  proc validateSgdParameter(v: Variable[MetalTensor[float32]]) =
    validateMetalView(v.value)
    validateMetalView(v.grad)
    if v.value.shape != v.grad.shape:
      raise newException(MetalError, "SGD parameter and gradient shapes differ")
    if v.value.storage.device.registryId != v.grad.storage.device.registryId:
      raise newException(MetalError, "SGD parameter and gradient devices differ")
    if v.value.storage == v.grad.storage:
      raise newException(MetalError, "SGD parameter and gradient must not share storage")

  proc step*(self: SGD[MetalTensor[float32]]) =
    ## Update weights in place; gradients and both native storage owners are preserved.
    for v in self.params:
      if v.requires_grad:
        validateSgdParameter(v)
        v.value.scaledSubtract(v.grad, self.lr)

  proc zeroGrads*(self: SGD[MetalTensor[float32]]) =
    ## Clear trainable gradients in place, never allocating or rebinding buffers.
    for v in self.params:
      if v.requires_grad:
        validateSgdParameter(v)
        v.grad.fill(0'f32)

  proc update*(self: SGD[MetalTensor[float32]]) =
    ## Existing compound behavior: take a step, then clear gradients.
    self.step()
    self.zeroGrads()

func optimizerSGD*[M, T](model: M, learning_rate: T): SGD[Tensor[T]] {.deprecated: "Use optimizer(model, SGD, learning_rate) instead."} =
  return optimizer(model, SGD, learning_rate)

# ############################################################
#
#    SGDMomentum: Stochastic Gradient Descent with Momentum
#
# ############################################################

type
  SGDMomentum*[TT] = object
    ## Stochastic gradient descent with momentum.
    ## Details on Nesterov momentum can be found in
    ## `Sutskever et. al. 2013<http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf>`_
    params*: seq[Variable[TT]]
    lr*: TT.T                ## Learning rate
    momentum*: TT.T          ## Value of the momentum
    moments: seq[TT]         ## Moments for momentum
    decay: TT.T              ## Learning rate decay
    nesterov: bool           ## Flag for Nesterov momentum

proc update*(self: var SGDMomentum) =
  ## Performs an optimization update.
  ##
  ## Parameters:
  ## - ``self`` A SGDMomentum optimizer to update.
  ##
  ## This proc will update the weights in the model associated with the input
  ## optimizer according to the following rule:
  ##    `w = w - lr * gradient + m * moment`
  ## If nesterov is set to true then the following rule is applied instead:
  ##    `w = w - lr * gradient + m * v`
  ##
  ##    `v = - lr * gradient + m * moment`
  ## Where in both cases the `moment` is the gradient change applied in the
  ## previous update step and `m` is the momentum.
  ##
  ## If `decay` is greater than 0, the learning rate will be modified each
  ## call according to the following:
  ##      `lr = lr * 1/(1 + decay)`

  # Decay of the learning rate.
  self.lr *= 1 / (self.decay + 1)
  # Keeps track of decay without having to keep track of iterations.
  # Each update call is counted as one iteration.
  self.decay += self.decay
  for i in 0 ..< self.params.len:
    let v = self.params[i]
    if v.requires_grad:
      # This implementation of both kinds of momentum follows that of Tensorflow
      # and closely mirrors the formulation of Sustkever et. al.
      # Update the moments with the previous update.
      apply2_inline(self.moments[i], v.grad):
        self.momentum * x - self.lr * y

      # When momentum = 0 this acts identically to SGD without momentum.
      if self.nesterov:
        # Update the params with formula Value = value - lr * gradient + momentum * v
        # where v = - lr * gradient + momentum * moment
        apply3_inline(v.value, v.grad, self.moments[i]):
          x - self.lr * y + self.momentum * z
      else:
        # Update the params with formula Value = value - lr * gradient + momentum * moment
        apply2_inline(v.value, self.moments[i]):
          x + y

      # Zero the gradient
      v.grad = v.value.zeros_like # TODO "setZero" instead of a new allocation

proc optimizer*[M, T](
        model: M,
        OptimizerKind: typedesc[SGDMomentum],
        learning_rate: T,
        momentum: T = T(0.0),
        decay: T = T(0.0),
        nesterov = false): SGDMomentum[Tensor[T]] =
  ## Create a SGD optimizer with optional momentum that will update the model weight
  ##
  ## Parameters:
  ## - ``model`` Model to optimize.
  ## - ``learning_rate`` Learning rate.
  ## - ``momentum`` Momentum.
  ## - ``decay`` How much the learning rate will decay each update.
  ## - ``nesterov`` Whether to use Nesterov momentum or not.
  ##
  ## Returns:
  ## - A SGD optimizer with momentum with the given parameters.

  result.params = @[]
  result.lr = learning_rate
  result.momentum = momentum
  result.decay = decay
  result.nesterov = nesterov

  proc addLayer(result: var SGDMomentum[Tensor[T]], layer: auto) =
    when layer is Variable:
      if not layer.isNil:
        result.params.add layer
        result.moments.add layer.grad.zeros_like
    elif layer is seq or layer is array:
      for item in layer:
        result.addLayer(item)
    elif layer is object or layer is tuple:
      for field in fields(layer):
        result.addLayer(field)

  result.addLayer(model)

proc optimizerSGDMomentum*[M, T](model: M, learning_rate: T, momentum = T(0.0), decay = T(0.0), nesterov = false): SGDMomentum[Tensor[T]] {.deprecated: "Use optimizer(model, SGDMomentum, learning_rate) instead."} =
  return optimizer(model, SGDMomentum, learning_rate, momentum, decay, nesterov)

# ############################################################
#
#             Adam: Adaptative Moment Estimation
#
# ############################################################

type
  Adam*[TT] = object
    ## Adaptative Moment Estimation
    params: seq[Variable[TT]]       ## Learnable weights
    learning_rate: TT.T
    beta1, beta2: TT.T              ## Decays on first and second moment
    beta1_t, beta2_t: TT.T          ## Current decay
    first_moments: seq[TT]          ## Exponential moving averages (mean estimation)
    second_moments: seq[TT]         ## Exponential moving averages squared (uncentered variance)
    epsilon: TT.T                   ## Epsilon for numerical stability when dividing

proc update*(self: var Adam) =
  # We use the second formulation of Adam from Kingma et al similar to Tensorflow

  # Bias corrected learning rate
  let lr_t = self.learning_rate * sqrt(1 - self.beta2_t) / (1 - self.beta1_t)

  # Raise β1^t and β2^t for next update
  self.beta1_t *= self.beta1
  self.beta2_t *= self.beta2

  for i in 0 ..< self.params.len:
    let v = self.params[i]
    if v.requires_grad:
      # Update biaised first moment estimate
      apply2_inline(self.first_moments[i], v.grad):
        self.beta1 * x + (1 - self.beta1) * y
      # Update biaised second moment estimate
      apply2_inline(self.second_moments[i], v.grad):
        self.beta2 * x + (1 - self.beta2) * y * y
      # Adjust weight
      apply3_inline(v.value, self.first_moments[i], self.second_moments[i]):
        x - lr_t * y / (z.sqrt + self.epsilon)

      # Zero the gradient
      v.grad = v.value.zeros_like # TODO "setZero" instead of a new allocation

proc optimizer*[M, T](
        model: M,
        OptimizerKind: typedesc[Adam],
        learning_rate: T = T(0.001),
        beta1: T = T(0.9),
        beta2: T = T(0.999),
        eps: T = T(1e-8)
      ): Adam[Tensor[T]] =
  ## Create a Adam optimizer that will update the model weight

  result.params = @[]
  result.learning_rate = learning_rate
  result.beta1 = beta1
  result.beta1_t = beta1
  result.beta2 = beta2
  result.beta2_t = beta2
  result.epsilon = eps

  proc addLayer(result: var Adam[Tensor[T]], layer: auto) =
    when layer is Variable:
      if not layer.isNil:
        result.params.add layer
        result.first_moments.add layer.grad.zeros_like
        result.second_moments.add layer.grad.zeros_like
    elif layer is seq or layer is array:
      for item in layer:
        result.addLayer(item)
    elif layer is object or layer is tuple:
      for field in fields(layer):
        result.addLayer(field)

  result.addLayer(model)

proc optimizerAdam*[M, T](model: M, learning_rate: T, beta1: T = T(0.9), beta2: T = T(0.999), eps: T = T(1e-8)): Adam[Tensor[T]] {.deprecated: "Use optimizer(model, SGDMomentum, learning_rate) instead."} =
  return optimizer(model, Adam, learning_rate, beta1, beta2, eps)

# ############################################################
#
#                 Generic optimizers
#
# ############################################################

type
  Optimizer*[TT] = SGD[TT] or Adam[TT]

proc zeroGrads*(o: Optimizer) =
  # Reset the gradients of the optimized params
  # TODO setZero instead of allocating.
  for v in o.params:
    v.grad = v.value.zeros_like
