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
        ./private/p_extrema_gradient,
        ./private/p_logsumexp_gradient,
        std/[complex, sequtils]
when defined(macosx) and defined(arm64):
  import ../tensor/private/p_accelerate

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

proc cpuReverseCumsumGradient[T: SomeFloat](
    seed: Tensor[T], inputShape: Metadata, axis: int): Tensor[T] =
  ## Reverse inclusive scan of the seed into one packed input-shaped output.
  result = newTensorUninit[T](inputShape)
  if result.size == 0:
    return
  if inputShape.len == 0:
    result.storage.raw_buffer[0] = seed.storage.raw_buffer[seed.offset]
    return

  var outer = 1'u
  for dim in 0 ..< axis:
    outer *= uint(inputShape[dim])
  var inner = 1'u
  for dim in axis + 1 ..< inputShape.len:
    inner *= uint(inputShape[dim])
  let axisLength = uint(inputShape[axis])
  let groupCount = outer * inner
  let seedData = seed.storage.raw_buffer
  let output = result.storage.raw_buffer

  if seed.is_C_contiguous:
    when defined(macosx) and defined(arm64) and T is float32:
      scanPackedFloat32(addr output[0], unsafeAddr seedData[seed.offset],
                        outer, axisLength, inner, false, true)
    else:
      # Backprop validates the seed shape before this packed traversal.
      let seedOffset = uint(seed.offset)
      for group in 0'u ..< groupCount:
        let outputBase = (group div inner) * axisLength * inner + group mod inner
        var accumulated = 0.0
        var remaining = axisLength
        while remaining > 0:
          dec remaining
          let flat = outputBase + remaining * inner
          accumulated += float64(seedData[seedOffset + flat])
          output[flat] = T(accumulated)
    return

  var coordinates: Metadata
  coordinates.len = inputShape.len
  for dim in 0 ..< inputShape.len:
    coordinates[dim] = 0
  var seedBase = seed.offset
  var group = 0'u
  while group < groupCount:
    let outputBase = (group div inner) * axisLength * inner + group mod inner
    when T is float32:
      var accumulated: float64
      var hasValue = false
      var index = axisLength
      while index > 0:
        dec index
        let value = float64(seedData[seedBase + int(index) * seed.strides[axis]])
        if hasValue:
          accumulated += value
        else:
          accumulated = value
          hasValue = true
        output[outputBase + index * inner] = T(accumulated)
    else:
      var accumulated: T
      var hasValue = false
      var index = axisLength
      while index > 0:
        dec index
        let value = seedData[seedBase + int(index) * seed.strides[axis]]
        if hasValue:
          accumulated = accumulated + value
        else:
          accumulated = value
          hasValue = true
        output[outputBase + index * inner] = accumulated

    if group + 1'u < groupCount:
      for dim in countdown(inputShape.len - 1, 0):
        if dim == axis:
          continue
        if coordinates[dim] < inputShape[dim] - 1:
          inc coordinates[dim]
          seedBase += seed.strides[dim]
          break
        coordinates[dim] = 0
        seedBase -= (inputShape[dim] - 1) * seed.strides[dim]
    inc group

type CumsumGate[TT] {.final.} = ref object of Gate[TT]
  cached_input_shape: Metadata
  axis: int

proc cumsum_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = CumsumGate[TT](self)
  result = newDiffs[TT](1)
  when TT is Tensor[float32]:
    result[0] = cpuReverseCumsumGradient(
      payload.variable.grad, self.cached_input_shape, self.axis)
  elif TT is Tensor[float64]:
    result[0] = cpuReverseCumsumGradient(
      payload.variable.grad, self.cached_input_shape, self.axis)
  elif defined(metal) or defined(coreai):
    when TT is MetalTensor[float32]:
      result[0] = cumsumGradient(payload.variable.grad, self.axis)
    else:
      {.error: "Variable cumsum gradients support CPU float32/float64 and Metal float32".}
  else:
    {.error: "Variable cumsum gradients support CPU float32/float64 and Metal float32".}

proc cumsum_cache[TT](result: Variable[TT], a: Variable[TT], axis: int) =
  var gate: CumsumGate[TT]
  new gate
  gate.cached_input_shape = a.value.shape
  gate.axis = axis
  result.grad = zeros_like(result.value)
  result.requires_grad = true
  register_node(
    "Cumsum",
    gate,
    cumsum_backward_ag[TT],
    result,
    a
  )

proc cumsum*[TT](a: Variable[TT], axis: int = 0): Variable[TT] =
  ## Inclusive cumulative sum along a positive or negative axis.
  ## Backward is a reverse inclusive scan of the upstream gradient.
  let normalized = reductionAxis(a.value.rank, axis)
  new result
  result.context = a.context
  result.value = a.value.cumsum(normalized)
  if a.is_grad_needed:
    result.cumsum_cache(a, normalized)

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
  when packed and T is float64 and defined(macosx) and defined(arm64):
    if inner >= 2 and cachedProduct.is_C_contiguous and seed.is_C_contiguous:
      if productGradientPackedColumnsFloat64(
          addr output[0], unsafeAddr inputData[input.offset],
          cachedProduct.get_offset_ptr, seed.get_offset_ptr, outer, axisLength, inner):
        return
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

proc cpuCumprodGradient[T: SomeFloat](
    input, prefixes, seed: Tensor[T], axis: int,
    packed: static[bool]): Tensor[T] =
  ## Keep the pinned first-order zero masks; omitted-product formulas differ at Inf.
  if input.size <= 1 or input.shape[axis] == 1:
    return seed.clone
  result = newTensorUninit[T](input.shape)
  var outer = 1'u
  var inner = 1'u
  for dim in 0 ..< axis: outer *= uint(input.shape[dim])
  for dim in axis + 1 ..< input.rank: inner *= uint(input.shape[dim])
  let length = uint(input.shape[axis])
  let inputData = input.storage.raw_buffer
  let prefixData = prefixes.storage.raw_buffer
  let seedData = seed.storage.raw_buffer
  let output = result.storage.raw_buffer
  when packed and T is float32 and defined(macosx) and defined(arm64):
    if inner >= 4 and cumprodGradientPackedColumnsFloat32(
        addr output[0], unsafeAddr inputData[input.offset],
        unsafeAddr prefixData[prefixes.offset], unsafeAddr seedData[seed.offset],
        outer, length, inner):
      return
  let prefixOffset = uint(prefixes.offset)
  when packed:
    let inputOffset = uint(input.offset)
    let seedOffset = uint(seed.offset)
  else:
    var coordinates: Metadata
    coordinates.len = input.rank
    for dim in 0 ..< input.rank: coordinates[dim] = 0
    var inputBase = input.offset
    var seedBase = seed.offset
  let groups = outer * inner
  for group in 0'u ..< groups:
    let base = (group div inner) * length * inner + group mod inner
    template inputAt(index: uint): T =
      when packed: inputData[inputOffset + base + index * inner]
      else: inputData[inputBase + int(index) * input.strides[axis]]
    template seedAt(index: uint): T =
      when packed: seedData[seedOffset + base + index * inner]
      else: seedData[seedBase + int(index) * seed.strides[axis]]
    template prefixAt(index: uint): T =
      prefixData[prefixOffset + base + index * inner]
    var firstZero = length
    for index in 0'u ..< length:
      if inputAt(index) == T(0):
        firstZero = index
        break
    for index in firstZero ..< length:
      output[base + index * inner] = T(0)
    var accumulated = 0.0
    var remaining = firstZero
    while remaining > 0:
      dec remaining
      let weight = prefixAt(remaining) * seedAt(remaining)
      accumulated += float64(weight)
      output[base + remaining * inner] = T(accumulated) / inputAt(remaining)
    if firstZero < length:
      var suffixProduct = 1.0
      var zeroSum = T(0)
      for index in firstZero ..< length:
        if index > firstZero:
          let value = inputAt(index)
          if value == T(0):
            # Masked gradients still multiply the carried Inf/NaN by zero.
            let maskedWeight = T(suffixProduct) * T(0)
            zeroSum = zeroSum + maskedWeight
            break
          suffixProduct *= float64(value)
        let weight = T(suffixProduct) * seedAt(index)
        zeroSum = zeroSum + weight
      let beforeZero = if firstZero == 0: T(1) else: prefixAt(firstZero - 1)
      output[base + firstZero * inner] = zeroSum * beforeZero
    when not packed:
      if group + 1 < groups:
        for dim in countdown(input.rank - 1, 0):
          if dim == axis: continue
          if coordinates[dim] < input.shape[dim] - 1:
            inc coordinates[dim]
            inputBase += input.strides[dim]
            seedBase += seed.strides[dim]
            break
          coordinates[dim] = 0
          inputBase -= (input.shape[dim] - 1) * input.strides[dim]
          seedBase -= (input.shape[dim] - 1) * seed.strides[dim]

type CumprodGate[TT] {.final.} = ref object of Gate[TT]
  cached_input: TT
  cached_prefixes: TT
  axis: int

proc cumprod_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = CumprodGate[TT](self)
  result = newDiffs[TT](1)
  when TT is Tensor[float32] or TT is Tensor[float64]:
    if self.cached_input.is_C_contiguous and payload.variable.grad.is_C_contiguous:
      result[0] = cpuCumprodGradient(
        self.cached_input, self.cached_prefixes, payload.variable.grad, self.axis, true)
    else:
      result[0] = cpuCumprodGradient(
        self.cached_input, self.cached_prefixes, payload.variable.grad, self.axis, false)
  elif defined(metal) or defined(coreai):
    when TT is MetalTensor[float32]:
      result[0] = cumprodGradient(
        self.cached_input, self.cached_prefixes, payload.variable.grad, self.axis)
    else:
      {.error: "Variable cumulative products support CPU float32/float64 and Metal float32".}
  else:
    {.error: "Variable cumulative products support CPU float32/float64 and Metal float32".}

proc cumprod*[TT](a: Variable[TT], axis: int = 0): Variable[TT] =
  ## CPU float32/float64 and Metal float32 inclusive product with seeded VJPs.
  ## Negative/scalar axes, exact empty shapes and signed/broadcast reads are
  ## supported. Backward uses rounded forward prefixes and zero-aware masks;
  ## CPU float32 reverse sums keep wide carries, while Metal stays float32.
  ## Output and gradient storage are independent; saved input/prefix storage
  ## must not be mutated before backward.
  let normalized = reductionAxis(a.value.rank, axis)
  new result
  result.context = a.context
  result.value = a.value.cumprod(normalized)
  if a.is_grad_needed:
    var gate: CumprodGate[TT]
    new gate
    gate.cached_input = a.value
    gate.cached_prefixes = result.value
    gate.axis = normalized
    result.grad = zeros_like(result.value)
    result.requires_grad = true
    register_node("Cumprod", gate, cumprod_backward_ag[TT], result, a)

type ExtremaGate[T] {.final.} = ref object of Gate[Tensor[T]]
  cachedInput, cachedExtrema: Tensor[T]
  reduced: Metadata

proc extrema_backward_ag[T: SomeFloat](self: Gate[Tensor[T]];
    payload: Payload[Tensor[T]]): SmallDiffs[Tensor[T]] =
  let gate = ExtremaGate[T](self)
  result = newDiffs[Tensor[T]](1)
  result[0] = cpuExtremaGradient(gate.cachedInput, gate.cachedExtrema,
    payload.variable.grad, gate.reduced)

proc extremaVariable[T: SomeFloat](a: Variable[Tensor[T]], axes: openArray[int];
    keepdim: bool; maximum: static[bool]): Variable[Tensor[T]] =
  new result
  result.context = a.context
  when maximum:
    result.value = a.value.amax(axes, keepdim)
  else:
    result.value = a.value.amin(axes, keepdim)
  if a.is_grad_needed:
    var gate: ExtremaGate[T]
    new gate
    gate.cachedInput = a.value
    gate.cachedExtrema = result.value
    gate.reduced.len = a.value.rank
    if axes.len == 0:
      for axis in 0 ..< a.value.rank: gate.reduced[axis] = 1
    elif a.value.rank > 0:
      for axis in axes:
        let normalized = if axis < 0: axis + a.value.rank else: axis
        gate.reduced[normalized] = 1
    result.grad = zeros_like(result.value)
    result.requires_grad = true
    when maximum:
      register_node("Amax", gate, extrema_backward_ag[T], result, a)
    else:
      register_node("Amin", gate, extrema_backward_ag[T], result, a)

proc amin*[T: SomeFloat](a: Variable[Tensor[T]], keepdim = false): Variable[Tensor[T]] =
  ## Minimum of all CPU float32/float64 values, returning a rank-zero tensor.
  ## ``keepdim`` retains all reduced dimensions. First-order backward splits
  ## the seed evenly between equal minima, preserving masked NaN/-0 arithmetic.
  ## Saved input/output storage must not be mutated before backward.
  extremaVariable(a, [], keepdim, false)

proc amax*[T: SomeFloat](a: Variable[Tensor[T]], keepdim = false): Variable[Tensor[T]] =
  ## Maximum of all CPU float32/float64 values, with tie-splitting first-order VJPs.
  ## Output and input-shaped gradients own independent storage; ``keepdim`` retains
  ## reduced axes. Saved input/output storage must not change before backward.
  extremaVariable(a, [], keepdim, true)

proc amin*[T: SomeFloat](a: Variable[Tensor[T]], axis: int;
    keepdim = false): Variable[Tensor[T]] =
  ## Minimum along a positive/negative axis; drop it unless ``keepdim`` is true.
  ## Scalar and empty-input rules match Tensor ``amin``; backward splits ties.
  extremaVariable(a, [axis], keepdim, false)

proc amax*[T: SomeFloat](a: Variable[Tensor[T]], axis: int;
    keepdim = false): Variable[Tensor[T]] =
  ## Maximum along a positive/negative axis, with independent tie-split gradients.
  extremaVariable(a, [axis], keepdim, true)

proc amin*[T: SomeFloat](a: Variable[Tensor[T]], axes: openArray[int];
    keepdim = false): Variable[Tensor[T]] =
  ## Minimum across distinct axes; an empty list reduces all dimensions.
  ## Backward divides each upstream value by the number of equal minima.
  extremaVariable(a, axes, keepdim, false)

proc amax*[T: SomeFloat](a: Variable[Tensor[T]], axes: openArray[int];
    keepdim = false): Variable[Tensor[T]] =
  ## Maximum across distinct axes, with dtype-preserving tie-split seeded VJPs.
  extremaVariable(a, axes, keepdim, true)

type PairedExtremaGate[T] {.final.} = ref object of Gate[Tensor[T]]
  cachedIndices: Tensor[int64]
  inputShape: Metadata
  axis: int

proc paired_extrema_backward_ag[T: SomeFloat](self: Gate[Tensor[T]];
    payload: Payload[Tensor[T]]): SmallDiffs[Tensor[T]] =
  let gate = PairedExtremaGate[T](self)
  result = newDiffs[Tensor[T]](1)
  result[0] = cpuPairedExtremaGradient(gate.cachedIndices,
    payload.variable.grad, gate.inputShape, gate.axis)

proc pairedExtremaVariable[T: SomeFloat](a: Variable[Tensor[T]], axis: int;
    keepdim: bool; maximum: static[bool]):
    tuple[values: Variable[Tensor[T]], indices: Tensor[int64]] =
  let output =
    when maximum: a.value.maxWithIndices(axis, keepdim)
    else: a.value.minWithIndices(axis, keepdim)
  new result.values
  result.values.context = a.context
  result.values.value = output.values
  result.indices = output.indices
  if a.is_grad_needed:
    var gate: PairedExtremaGate[T]
    new gate
    gate.cachedIndices = result.indices
    gate.inputShape = a.value.shape
    gate.axis = if axis < 0: axis + max(a.value.rank, 1) else: axis
    result.values.grad = zeros_like(output.values)
    result.values.requires_grad = true
    when maximum:
      register_node("MaxWithIndices", gate, paired_extrema_backward_ag[T],
        result.values, a)
    else:
      register_node("MinWithIndices", gate, paired_extrema_backward_ag[T],
        result.values, a)

proc minWithIndices*[T: SomeFloat](a: Variable[Tensor[T]], axis: int;
    keepdim = false): tuple[values: Variable[Tensor[T]], indices: Tensor[int64]] =
  ## CPU minimum values and nondifferentiable int64 first indices along an axis.
  ## First-order backward scatters the seed only to each chosen logical position;
  ## inactive gradients are +0 even for nonfinite/-0 seeds. Output and dense
  ## input-shaped gradients own storage. Saved indices must not change before
  ## backward. Scalar/negative axes and ``keepdim`` match the Tensor overload.
  pairedExtremaVariable(a, axis, keepdim, false)

proc maxWithIndices*[T: SomeFloat](a: Variable[Tensor[T]], axis: int;
    keepdim = false): tuple[values: Variable[Tensor[T]], indices: Tensor[int64]] =
  ## CPU maximum values and nondifferentiable int64 first indices along an axis.
  ## Backward routes each seed to the first tie/NaN, without tie splitting or
  ## nonfinite mask arithmetic. Saved indices must not change before backward;
  ## signed/broadcast input views still receive independent dense gradients.
  pairedExtremaVariable(a, axis, keepdim, true)

type LogsumexpGate[T] {.final.} = ref object of Gate[Tensor[T]]
  cachedInput, cachedNormalizer: Tensor[T]
  reduced: Metadata

proc logsumexp_backward_ag[T: SomeFloat or Complex](self: Gate[Tensor[T]];
    payload: Payload[Tensor[T]]): SmallDiffs[Tensor[T]] =
  let gate = LogsumexpGate[T](self)
  result = newDiffs[Tensor[T]](1)
  result[0] = cpuLogsumexpGradient(gate.cachedInput, gate.cachedNormalizer,
    payload.variable.grad, gate.reduced)

proc logsumexpVariable[T: SomeFloat or Complex](a: Variable[Tensor[T]];
    axes: openArray[int]; whole: static[bool]; keepdim: bool): Variable[Tensor[T]] =
  new result
  result.context = a.context
  when whole:
    result.value = a.value.logsumexp(keepdim)
  else:
    result.value = a.value.logsumexp(axes, keepdim)
  if a.is_grad_needed:
    var gate: LogsumexpGate[T]
    new gate
    gate.cachedInput = a.value
    gate.cachedNormalizer = result.value
    gate.reduced.len = a.value.rank
    when whole:
      for axis in 0 ..< a.value.rank: gate.reduced[axis] = 1
    else:
      if axes.len == 0:
        for axis in 0 ..< a.value.rank: gate.reduced[axis] = 1
      elif a.value.rank > 0:
        for axis in axes:
          let normalized = if axis < 0: axis + a.value.rank else: axis
          gate.reduced[normalized] = 1
    result.grad = zeros_like(result.value)
    result.requires_grad = true
    register_node("Logsumexp", gate, logsumexp_backward_ag[T], result, a)

proc logsumexp*[T: SomeFloat or Complex](a: Variable[Tensor[T]];
    keepdim = false): Variable[Tensor[T]] =
  ## Stable CPU float32/64 and complex64/128 reduction to a scalar by default.
  ## First-order backward computes seed * conj(exp(input - rounded output)),
  ## with one independent dense input-shaped gradient. Saved input and output
  ## storage must not be mutated before backward. ``keepdim`` retains all axes.
  logsumexpVariable(a, [], true, keepdim)

proc logsumexp*[T: SomeFloat or Complex](a: Variable[Tensor[T]]; axis: int;
    keepdim = false): Variable[Tensor[T]] =
  ## Stable logarithmic reduction along a signed axis, with seeded first-order VJPs.
  ## Signed/offset/broadcast inputs and seeds are read without conversion tensors.
  ## Scalar/empty rules and independent output storage match Tensor ``logsumexp``.
  logsumexpVariable(a, [axis], false, keepdim)

proc logsumexp*[T: SomeFloat or Complex](a: Variable[Tensor[T]];
    axes: openArray[int]; keepdim = false): Variable[Tensor[T]] =
  ## Reduce distinct signed axes; explicit empty lists retain the Tensor contract.
  ## Gradients use the rounded forward result, not an unrounded softmax.
  ## This CPU API does not provide higher-order, Metal or output-buffer overloads.
  logsumexpVariable(a, axes, false, keepdim)
