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
        ../../autograd,
        ../../autograd/private/p_broadcast,
        ../../tensor/private/p_inner_spans

when defined(metal) or defined(coreai):
  import ../../nn_primitives/nnp_mse_metal

type MseReduction* = enum
  ## Reduction over broadcasted squared errors.
  mseMean, mseSum, mseNone

proc mseForward[T: SomeFloat](prediction, target: Tensor[T], reduction: MseReduction): Tensor[T] =
  let (p, t) = broadcast2(prediction, target)
  if reduction == mseNone:
    return map2_inline(p, t):
      let difference = x - y
      difference * difference
  if p.size == 0:
    return newTensorWith[T]([], if reduction == mseMean: T(0) / T(p.size) else: T(0))
  var loss = T(0)
  # Independent accumulators permit SIMD without relaxing IEEE arithmetic.
  var partials: array[4, T]
  let strides = [p.strides, t.strides]
  let span = innerSpan(p.shape, strides)
  let origins = [p.offset, t.offset]
  let predictionData = p.storage.raw_buffer
  let targetData = t.storage.raw_buffer
  for _, offsets in spanOffsets(p.shape, strides, origins,
      span.outerRank, p.size div span.length):
    let predictionOffset = offsets[0]
    let targetOffset = offsets[1]
    let count = uint(span.length)
    withSpanStep(span.steps[0], predictionStep):
      withSpanStep(span.steps[1], targetStep):
        var cursor = 0'u
        while cursor + 4'u <= count:
          for lane in 0 ..< 4:
            let i = cast[int](cursor) +% lane
            let difference = predictionData[predictionOffset +% (i *% predictionStep)] -
              targetData[targetOffset +% (i *% targetStep)]
            partials[lane] += difference * difference
          cursor += 4'u
        while cursor < count:
          let i = cast[int](cursor)
          let difference = predictionData[predictionOffset +% (i *% predictionStep)] -
            targetData[targetOffset +% (i *% targetStep)]
          loss += difference * difference
          inc cursor
  for partial in partials: loss += partial
  if reduction == mseMean: loss /= T(p.size)
  result = newTensorUninit[T]([])
  result.unsafe_raw_offset[0] = loss

proc mseBackwardCpu[T: SomeFloat](upstream, prediction, target: Tensor[T],
    reduction: MseReduction, needPrediction, needTarget: bool):
    tuple[prediction, target: Tensor[T]] =
  ## Accumulate each requested gradient directly in its original shape.
  let (p, t) = broadcast2(prediction, target)
  if needPrediction: result.prediction = zeros[T](prediction.shape)
  if needTarget: result.target = zeros[T](target.shape)
  if p.size == 0: return
  let gradient = broadcast2(upstream, p).a
  var predictionStrides, targetStrides: Metadata
  predictionStrides.len = p.rank
  targetStrides.len = p.rank
  var predictionGradient, targetGradient: ptr UncheckedArray[T]
  if needPrediction:
    predictionStrides = broadcast2(result.prediction, p).a.strides
    predictionGradient = result.prediction.storage.raw_buffer
  if needTarget:
    targetStrides = broadcast2(result.target, p).a.strides
    targetGradient = result.target.storage.raw_buffer
  let strides = [p.strides, t.strides, gradient.strides, predictionStrides, targetStrides]
  let origins = [p.offset, t.offset, gradient.offset, 0, 0]
  let span = innerSpan(p.shape, strides)
  let predictionData = p.storage.raw_buffer
  let targetData = t.storage.raw_buffer
  let upstreamData = gradient.storage.raw_buffer
  let norm = if reduction == mseMean: T(2) / T(p.size) else: T(2)
  for _, offsets in spanOffsets(p.shape, strides, origins,
      span.outerRank, p.size div span.length):
    for cursor in 0'u ..< uint(span.length):
      let i = cast[int](cursor)
      let difference = predictionData[offsets[0] +% (i *% span.steps[0])] -
        targetData[offsets[1] +% (i *% span.steps[1])]
      let value = norm * difference * upstreamData[offsets[2] +% (i *% span.steps[2])]
      if needPrediction:
        predictionGradient[offsets[3] +% (i *% span.steps[3])] += value
      if needTarget:
        targetGradient[offsets[4] +% (i *% span.steps[4])] -= value

when defined(metal) or defined(coreai):
  proc mseForward(prediction, target: MetalTensor[float32],
                  reduction: MseReduction): MetalTensor[float32] =
    let (p, t) = broadcast2(prediction, target)
    if reduction == mseMean: return mse_forward_metal(p, t)
    let squaredDifference = newMetalTensorUninit[float32](p.shape, p.storage.device)
    dispatchMetalMath(p, t, squaredDifference, 23)
    if reduction == mseSum: squaredDifference.sum else: squaredDifference

  proc mseBackward(upstream, prediction, target: MetalTensor[float32],
                   reduction: MseReduction): MetalTensor[float32] =
    let (p, t) = broadcast2(prediction, target)
    if reduction == mseMean: return mse_backward_metal(upstream, p, t)
    result = p -. t
    dispatchMetalMath(result, upstream.broadcast(p.shape), result, 3)
    dispatchMetalMath(result, default(MetalTensor[float32]), result, 3, 2'f32, true)

type MSELoss*[TT] {.final.} = ref object of Gate[TT]
  target: TT
  cache: Variable[TT]
  targetVariable: Variable[TT]
  reduction: MseReduction

proc mse_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = MSELoss[TT](self)
  let targetNeedsGrad = not self.targetVariable.isNil and self.targetVariable.requires_grad
  result = newDiffs[TT](if self.targetVariable.isNil: 1 else: 2)
  when TT is Tensor:
    let gradients = mseBackwardCpu(payload.variable.grad, self.cache.value, self.target,
                                  self.reduction, self.cache.requires_grad, targetNeedsGrad)
    if self.cache.requires_grad: result[0] = gradients.prediction
    if targetNeedsGrad: result[1] = gradients.target
  else:
    if self.cache.requires_grad:
      let gradient = mseBackward(payload.variable.grad, self.cache.value, self.target, self.reduction)
      result[0] = reduce_broadcast_dims(gradient, self.cache.value.shape)
      if targetNeedsGrad:
        result[1] = reduce_broadcast_dims(-gradient, self.target.shape)
    elif targetNeedsGrad:
      let gradient = mseBackward(payload.variable.grad, self.target, self.cache.value, self.reduction)
      result[1] = reduce_broadcast_dims(gradient, self.target.shape)

proc mseLoss[TT](input: Variable[TT], target: TT, targetVariable: Variable[TT],
                 reduction: MseReduction): Variable[TT] =
  new result
  result.context = input.context
  result.value = mseForward(input.value, target, reduction)
  if input.is_grad_needed or (not targetVariable.isNil and targetVariable.is_grad_needed):
    let gate = MSELoss[TT](cache: input, target: target,
                           targetVariable: targetVariable, reduction: reduction)
    result.grad = result.value.zeros_like
    result.requires_grad = true
    if targetVariable.isNil:
      register_node("Mean Squared Error", gate, mse_backward_ag[TT], result, input)
    else:
      register_node("Mean Squared Error", gate, mse_backward_ag[TT], result, input, targetVariable)

proc mse_loss*[TT](input: Variable[TT], target: TT, reduction = mseMean): Variable[TT] =
  ## Squared-error loss with trailing-axis broadcasting.
  ## ``mseMean`` averages all broadcasted errors; ``mseSum`` sums them.
  ## Both return rank-zero tensors. ``mseNone`` retains the broadcasted shape.
  ## Backward reduces prediction gradients to the original input shape.
  mseLoss(input, target, nil, reduction)

proc mse_loss*[TT](input, target: Variable[TT], reduction = mseMean): Variable[TT] =
  ## Differentiate squared errors with respect to either or both variables.
  ## Inputs must share a context; reduction and broadcasting match the tensor-target
  ## overload. Each trainable gradient is reduced to its original operand shape.
  check_ctx(input, target)
  mseLoss(input, target.value, target, reduction)
