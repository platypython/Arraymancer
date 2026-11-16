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

import  ../../tensor,
        ../../ml,
        ../../autograd

when defined(metal) or defined(coreai):
  import ../../nn_primitives/nnp_mse_metal

proc mseForward[TT](prediction, target: TT): TT =
  let loss = mean_squared_error(prediction, target)
  result = newTensorUninit[typeof(loss)]([])
  result.unsafe_raw_offset[0] = loss

proc mseBackward[TT](upstream, prediction, target: TT): TT =
  assert upstream.rank == 0
  let grad = upstream.unsafe_raw_offset[0]
  type T = typeof(grad)
  let norm = grad * T(2) / T(prediction.size)
  result = map2_inline(prediction, target):
    norm * (x - y)

when defined(metal) or defined(coreai):
  proc mseForward(prediction, target: MetalTensor[float32]): MetalTensor[float32] =
    mse_forward_metal(prediction, target)

  proc mseBackward(upstream, prediction, target: MetalTensor[float32]): MetalTensor[float32] =
    mse_backward_metal(upstream, prediction, target)

type MSELoss*[TT] {.final.} = ref object of Gate[TT]
  target: TT
  cache: Variable[TT]

proc mse_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = MSELoss[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = mseBackward(gradient, self.cache.value, self.target)

proc mse_cache[TT](result: Variable[TT], input: Variable[TT], target: TT) =
  ## We expect input with shape [batch_size, features]

  # Gate
  var gate: MSELoss[TT]
  new gate
  gate.cache = input
  gate.target = target

  # Result setup
  result.grad = zeros_like result.value
  result.requires_grad = true

  # Add to graph
  register_node(
    "Mean Squared Error",
    gate,
    mse_backward_ag[TT],
    result,
    input
  )

proc mse_loss*[TT](input: Variable[TT], target: TT): Variable[TT] =
  ## Mean square error loss function.
  ## Input:
  ##   - An input variable of predicted values of shape [batch_size, features]
  ##   - The ground truth of the same shape
  ## The returned value is an initialized rank-zero tensor (shape `[]`) containing one scalar.
  ## The loss averages over every input element; backward uses the same divisor.

  # Resulting var
  new result
  result.context = input.context
  result.value = mseForward(input.value, target)

  # Caching for backprop
  if input.is_grad_needed:
    result.mse_cache(input, target)
