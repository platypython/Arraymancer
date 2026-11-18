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
  value.mean(axis)

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

proc fullProductValue[T](value: Tensor[T]): Tensor[T] =
  reductionTensor(value.product)

proc axisProductValue[T](value: Tensor[T], axis: int): Tensor[T] =
  value.product(axis)

when defined(metal) or defined(coreai):
  proc fullProductValue(value: MetalTensor[float32]): MetalTensor[float32] =
    value.prod

  proc axisProductValue(value: MetalTensor[float32], axis: int): MetalTensor[float32] =
    value.prod(axis)

func cpuOffsetAtFlat[T](value: Tensor[T], index: int): int {.inline.} =
  result = value.offset
  var remaining = index
  for axis in countdown(value.rank - 1, 0):
    let coordinate = remaining mod value.shape[axis]
    remaining = remaining div value.shape[axis]
    result += coordinate * value.strides[axis]

func cpuFlatValue[T](value: Tensor[T], index: int): T {.inline.} =
  value.storage.raw_buffer[cpuOffsetAtFlat(value, index)]

proc cpuFullProductGradient[T: SomeFloat](
    input, cachedProduct, seed: Tensor[T], packed: static[bool]): Tensor[T] =
  ## Compute one full-reduction VJP into one packed input-shaped allocation.
  result = newTensorUninit[T](input.shape)
  if result.size == 0: return
  let output = result.storage.raw_buffer
  if input.rank == 0:
    output[0] = cpuFlatValue(seed, 0)
    return

  let inputData = input.storage.raw_buffer
  let count = uint(input.size)
  when packed:
    let inputOffset = uint(input.offset)
  template inputAt(index: uint): T =
    when packed: inputData[inputOffset + index]
    else: inputData[cpuOffsetAtFlat(input, int(index))]
  let upstream = cpuFlatValue(seed, 0)
  var zeroCount = 0'u
  for index in 0'u ..< count:
    zeroCount += uint(ord(inputAt(index) == T(0)))
  if zeroCount > 1:
    for index in 0'u ..< count: output[index] = T(0)
    return
  if zeroCount == 0:
    let weighted = upstream * cpuFlatValue(cachedProduct, 0)
    for index in 0'u ..< count: output[index] = weighted / inputAt(index)
    return
  var prefix = T(1)
  for index in 0'u ..< count:
    output[index] = prefix
    prefix *= inputAt(index)
  var suffix = T(1)
  var remaining = count
  while remaining > 0:
    dec remaining
    output[remaining] = output[remaining] * suffix * upstream
    suffix *= inputAt(remaining)

proc cpuAxisProductGradient[T: SomeFloat](
    input, cachedProduct, seed: Tensor[T], axis: int;
    packed: static[bool]): Tensor[T] =
  ## Packed offsets are bounded by input storage, avoiding hot overflow checks.
  ## Generic reads retain signed strides while all gradient writes stay packed.
  result = newTensorUninit[T](input.shape)
  if result.size == 0: return
  if input.rank == 0:
    result.storage.raw_buffer[0] = cpuFlatValue(seed, 0)
    return

  let axisLength = uint(input.shape[axis])
  var outer = 1'u
  for dim in 0 ..< axis: outer *= uint(input.shape[dim])
  var inner = 1'u
  for dim in axis + 1 ..< input.rank: inner *= uint(input.shape[dim])
  let inputData = input.storage.raw_buffer
  let output = result.storage.raw_buffer
  for group in 0'u ..< outer * inner:
    let outputBase = (group div inner) * axisLength * inner + group mod inner
    when packed:
      let baseOffset = uint(input.offset) + outputBase
    else:
      var baseOffset = input.offset
      var coordinates = int(group)
      for dim in countdown(input.rank - 1, 0):
        if dim == axis: continue
        let coordinate = coordinates mod input.shape[dim]
        coordinates = coordinates div input.shape[dim]
        baseOffset += coordinate * input.strides[dim]
    template inputAt(index: uint): T =
      when packed: inputData[baseOffset + index * inner]
      else: inputData[baseOffset + int(index) * input.strides[axis]]

    let seedValue = cpuFlatValue(seed, int(group))
    var zeroCount = 0'u
    for index in 0'u ..< axisLength:
      zeroCount += uint(ord(inputAt(index) == T(0)))
    if zeroCount == 0:
      let weighted = seedValue * cpuFlatValue(cachedProduct, int(group))
      for index in 0'u ..< axisLength:
        output[outputBase + index * inner] = weighted / inputAt(index)
      continue
    var prefix = T(1)
    for index in 0'u ..< axisLength:
      output[outputBase + index * inner] = prefix
      prefix *= inputAt(index)
    var suffix = T(1)
    var remaining = axisLength
    while remaining > 0:
      dec remaining
      let outputIndex = outputBase + remaining * inner
      output[outputIndex] = output[outputIndex] * suffix * seedValue
      suffix *= inputAt(remaining)

type ProductGate[TT] {.final.} = ref object of Gate[TT]
  cached_input: TT
  cached_product: TT
  axis: int
  fullReduction: bool
  keepdim: bool

proc product_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = ProductGate[TT](self)
  result = newDiffs[TT](1)
  when TT is Tensor[float32] or TT is Tensor[float64]:
    if self.fullReduction:
      if self.cached_input.is_C_contiguous:
        result[0] = cpuFullProductGradient(
          self.cached_input, self.cached_product, payload.variable.grad, true)
      else:
        result[0] = cpuFullProductGradient(
          self.cached_input, self.cached_product, payload.variable.grad, false)
    else:
      if self.cached_input.is_C_contiguous:
        result[0] = cpuAxisProductGradient(
          self.cached_input, self.cached_product, payload.variable.grad, self.axis, true)
      else:
        result[0] = cpuAxisProductGradient(
          self.cached_input, self.cached_product, payload.variable.grad, self.axis, false)
  elif defined(metal) or defined(coreai):
    when TT is MetalTensor[float32]:
      var seed = payload.variable.grad
      if not self.fullReduction and not self.keepdim and self.cached_input.rank > 0:
        seed = seed.unsqueeze(self.axis)
      result[0] = productGradient(
        self.cached_input, self.cached_product, seed,
        if self.fullReduction: -1 else: self.axis)
    else:
      {.error: "Variable product gradients support CPU float32/float64 and Metal float32".}
  else:
    {.error: "Variable product gradients support CPU float32/float64 and Metal float32".}

proc product_cache[TT](
    result: Variable[TT], a: Variable[TT], cachedProduct: TT,
    axis: int, fullReduction, keepdim: bool) =
  var gate: ProductGate[TT]
  new gate
  gate.cached_input = a.value
  gate.cached_product = cachedProduct
  gate.axis = axis
  gate.fullReduction = fullReduction
  gate.keepdim = keepdim

  result.grad = zeros_like(result.value)
  result.requires_grad = true
  register_node(
    "Product",
    gate,
    product_backward_ag[TT],
    result,
    a
  )

proc product*[TT](a: Variable[TT]): Variable[TT] =
  ## Multiply all elements, returning an initialized rank-zero tensor.
  ## Supports CPU float32/float64 and Metal float32; empty products are one.
  ## Backward allocates an independent input-shaped first-order gradient and
  ## preserves zero/nonfinite derivatives rather than blindly dividing by zero.
  new result
  result.context = a.context
  result.value = fullProductValue(a.value)
  if a.is_grad_needed:
    result.product_cache(a, result.value, -1, true, true)

proc product*[TT](a: Variable[TT], axis: int, keepdim = true): Variable[TT] =
  ## Multiply along a positive or negative axis, retaining its dimension by default.
  ## ``keepdim = false`` removes that axis; scalar inputs remain rank zero.
  ## CPU float32/float64 and Metal float32 support noncontiguous/broadcast reads.
  let normalized = reductionAxis(a.value.rank, axis)
  let cachedProduct = axisProductValue(a.value, normalized)
  new result
  result.context = a.context
  result.value = if keepdim: cachedProduct else: cachedProduct.squeeze(normalized)
  if a.is_grad_needed:
    result.product_cache(a, cachedProduct, normalized, false, keepdim)
