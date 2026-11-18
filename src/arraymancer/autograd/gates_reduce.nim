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

import  ../private/ast_utils,
        ../tensor,
        ./autograd_common,
        sequtils

proc reductionTensor[T](value: T): Tensor[T] =
  ## Wrap a host reduction in an initialized scalar without intermediate storage.
  result = newTensorUninit[T]([])
  result.unsafe_raw_offset[0] = value

when defined(metal) or defined(coreai):
  proc reductionTensor[T](value: MetalTensor[T]): MetalTensor[T] =
    value

proc reductionAxis(rank, axis: int): int =
  ## Normalize before indexing metadata, including release builds.
  if rank == 0:
    if axis == 0 or axis == -1: return 0
  elif axis >= -rank and axis < rank:
    return if axis < 0: axis + rank else: axis
  raise newException(ValueError, "Reduction axis is outside the input rank")

proc sumAlongAxis[T](value: Tensor[T], axis: int): Tensor[T] =
  if value.size == 0:
    var shape = value.shape
    shape[axis] = 1
    return zeros[T](shape)
  value.sum(axis)

proc meanAlongAxis[T](value: Tensor[T], axis: int): Tensor[T] =
  when T is SomeInteger:
    sumAlongAxis(value, axis) div T(value.shape[axis])
  else:
    sumAlongAxis(value, axis) / T(value.shape[axis])

when defined(metal) or defined(coreai):
  proc sumAlongAxis(value: MetalTensor[float32], axis: int): MetalTensor[float32] =
    value.sum(axis)

  proc meanAlongAxis(value: MetalTensor[float32], axis: int): MetalTensor[float32] =
    value.mean(axis)

type MeanGate*[TT] {.final.} = ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  cached_input_shape: Metadata
  axis: int


proc mean_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = MeanGate[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = gradient / getSubType(TT)(self.cached_input_shape.shapeProduct)

  # We create a shape of 1 dimension that we will expand with broadcast
  let z_shape = newSeqWith(self.cached_input_shape.len, 1)
  result[0] = result[0].reshape(z_shape).broadcast(self.cached_input_shape)

proc mean_with_axis_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = MeanGate[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  var shape = self.cached_input_shape
  shape[self.axis] = 1
  result[0] = (gradient.reshape(shape) / getSubType(TT)(self.cached_input_shape[self.axis])).broadcast(self.cached_input_shape)

proc mean_cache[TT](result: Variable[TT], a: Variable[TT]) =
  # Gate
  var gate: MeanGate[TT]
  new gate
  gate.cached_input_shape = a.value.shape

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Caching for backprop
  register_node(
    "Mean",
    gate,
    mean_backward_ag[TT],
    result,
    a
  )

proc mean_cache[TT](result: Variable[TT], a: Variable[TT], axis: int) =
  # Gate
  var gate: MeanGate[TT]
  new gate
  gate.cached_input_shape = a.value.shape
  gate.axis = axis

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Caching for backprop
  register_node(
    "Mean",
    gate,
    mean_with_axis_backward_ag[TT],
    result,
    a
  )

proc mean*[TT](a: Variable[TT]): Variable[TT] =
  ## Average all elements, returning an initialized rank-zero tensor.
  # Resulting var
  new result
  result.context = a.context
  result.value = reductionTensor(a.value.mean)

  # Caching for backprop
  if a.is_grad_needed:
    result.mean_cache(a)

proc mean*[TT](a: Variable[TT], axis: int, keepdim = true): Variable[TT] =
  ## Average along a positive or negative ``axis``.
  ## Retain the reduced dimension by default for compatibility; ``keepdim = false``
  ## removes it. Backward divides by the length of the reduced axis.
  let normalized = reductionAxis(a.value.rank, axis)
  if a.value.rank == 0: return a.mean()
  new result
  result.context = a.context
  result.value = meanAlongAxis(a.value, normalized)
  if not keepdim:
    result.value = result.value.squeeze(normalized)
  if a.is_grad_needed:
    result.mean_cache(a, normalized)

type SumGate*[TT]{.final.}= ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  cached_input_shape: Metadata
  axis: int

proc sum_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = SumGate[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)

  var shape = self.cached_input_shape
  for i in 0 ..< shape.len:
    if self.axis < 0 or i == self.axis: shape[i] = 1
  result[0] = gradient.reshape(shape).broadcast(self.cached_input_shape)

proc sum_cache[TT](result: Variable[TT], a: Variable[TT], axis = -1) =
  # Gate
  var gate: SumGate[TT]
  new gate
  gate.cached_input_shape = a.value.shape
  gate.axis = axis

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  register_node(
    "Sum",
    gate,
    sum_backward_ag[TT],
    result,
    a
  )

proc sum*[TT](a: Variable[TT]): Variable[TT] =
  ## Sum all elements, returning an initialized rank-zero tensor.

  # Resulting var
  new result
  result.context = a.context
  result.value = reductionTensor(a.value.sum)

  # Caching for backprop
  if a.is_grad_needed:
    result.sum_cache(a)

proc sum*[TT](a: Variable[TT], axis: int, keepdim = true): Variable[TT] =
  ## Sum along a positive or negative ``axis``.
  ## Retain the reduced dimension by default, matching tensor axis reductions;
  ## ``keepdim = false`` removes it, including rank-one inputs becoming scalars.
  let normalized = reductionAxis(a.value.rank, axis)
  if a.value.rank == 0: return a.sum()
  new result
  result.context = a.context
  result.value = sumAlongAxis(a.value, normalized)
  if not keepdim:
    result.value = result.value.squeeze(normalized)
  if a.is_grad_needed:
    result.sum_cache(a, normalized)
