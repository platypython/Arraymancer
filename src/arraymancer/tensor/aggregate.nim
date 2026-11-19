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

import  ./data_structure,
        ./init_cpu,
        ./higher_order_foldreduce,
        ./operators_broadcasted,
        ./operators_comparison,
        ./higher_order_applymap,
        ./math_functions,
        ./accessors,
        ./accessors_macros_syntax,
        ./algorithms,
        ./private/p_empty_tensors,
        ./private/p_accelerate,
        ./private/p_extrema,
        ./private/p_argextrema,
        ./private/p_aminmax
import std / [math, macros]
import complex except Complex64, Complex32
import ../laser/compiler_optim_hints
from ./private/p_complex import warnComplexToRealCast

# ### Standard aggregate functions
# TODO consider using stats from Nim standard lib: https://nim-lang.org/docs/stats.html#standardDeviation,RunningStat

template sumProductElement(T: typedesc, promoteIntegers: static[bool] = true): typedesc =
  when promoteIntegers and (T is SomeInteger or T is bool): int64
  else: T

template promotedInteger(value: untyped): int64 =
  when typeof(value) is bool: int64(ord(value))
  elif typeof(value) is SomeUnsignedInt: cast[int64](uint64(value))
  else: int64(value)

template combineInteger(acc, value: untyped, multiply: static[bool]) =
  when multiply: acc = acc *% promotedInteger(value)
  else: acc = acc +% promotedInteger(value)

proc reducePackedIntegers[T](input: ptr UncheckedArray[T], base, count: uint;
                             multiply: static[bool]): int64 =
  ## Independent modulo-64 carries shorten the dependency chain without rounding.
  ## Tensor shape bounds permit unsigned indices without checked hot-loop sums.
  const identity = when multiply: 1'i64 else: 0'i64
  var
    a = identity
    b = identity
    c = identity
    d = identity
  let bulk = count and not 3'u
  var index = 0'u
  while index < bulk:
    combineInteger(a, input[base + index], multiply)
    combineInteger(b, input[base + index + 1], multiply)
    combineInteger(c, input[base + index + 2], multiply)
    combineInteger(d, input[base + index + 3], multiply)
    index += 4
  result = a
  combineInteger(result, b, multiply)
  combineInteger(result, c, multiply)
  combineInteger(result, d, multiply)
  while index < count:
    combineInteger(result, input[base + index], multiply)
    inc index

template integerOutputIdentity(U: typedesc; multiply: static[bool]): untyped =
  when U is bool: multiply
  else: U(ord(multiply))

template requireIntegerReductionDtype(U: typedesc; size: int) =
  when U is SomeUnsignedInt and sizeof(U) > 1:
    if size != 0:
      raise newException(ValueError, "CPU reductions do not support this unsigned output dtype")

template combineIntegerOutput(acc, value: untyped; U: typedesc;
                              multiply: static[bool]) =
  # Low-bit stores commute with integer sums/products, including narrowed inputs.
  when U is bool:
    let truth = convertedReductionValue(value, bool)
    when multiply: acc = (ord(acc) and ord(truth)) != 0
    else: acc = (ord(acc) or ord(truth)) != 0
  elif sizeof(U) <= 4:
    var word = convertedIntegerWord(acc, uint32)
    when multiply: word *= convertedIntegerWord(value, uint32)
    else: word += convertedIntegerWord(value, uint32)
    acc = cast[U](word)
  else:
    var word = convertedIntegerWord(acc, uint64)
    when multiply: word *= convertedIntegerWord(value, uint64)
    else: word += convertedIntegerWord(value, uint64)
    acc = cast[U](word)

proc reducePackedIntegerOutput[T, U](input: ptr UncheckedArray[T], base, count: uint;
                                     multiply: static[bool]; dtype: typedesc[U]): U =
  ## Numeric zero and decisive boolean values finish these exact reductions.
  when U is bool:
    when T is Complex and multiply:
      result = reducePackedComplexTruth(input, base, count)
    elif T is SomeFloat and multiply:
      result = reducePackedFloatTruth(input, base, count)
    else:
      result = multiply
      for index in 0'u ..< count:
        if convertedReductionValue(input[base + index], bool) != multiply:
          return not multiply
  elif multiply:
    var word = 1'i64
    for index in 0'u ..< count:
      when T is bool:
        if not input[base + index]: return U(0)
      else:
        combineIntegerOutput(word, input[base + index], int64, true)
        if cast[U](word) == U(0): return U(0)
    result = cast[U](word)
  else:
    result = reduceConvertedIntegerSumSpan(input, base, count, U)

proc reduceIntegers[T, U](arg: Tensor[T], multiply: static[bool];
                          dtype: typedesc[U]): U =
  ## Cast on read; narrowing and carries preserve the requested low bits.
  requireIntegerReductionDtype(U, arg.size)
  result = integerOutputIdentity(U, multiply)
  if arg.size == 0: return
  if arg.is_C_contiguous:
    let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
    return reducePackedIntegerOutput(input, 0'u, arg.size.uint, multiply, U)
  for value in arg:
    when U is bool:
      if convertedReductionValue(value, bool) != multiply:
        return not multiply
    else:
      combineIntegerOutput(result, value, U, multiply)
      when multiply:
        if result == U(0): return

proc reduceIntegerAxis[T, U](arg: Tensor[T], axis: int;
                             multiply: static[bool]; dtype: typedesc[U]): Tensor[U] =
  ## Only the final output is allocated; its low bits are the running carries.
  withCompilerOptimHints()
  requireIntegerReductionDtype(U, arg.size)
  const identity = integerOutputIdentity(U, multiply)
  if arg.rank == 0:
    if arg.size == 0: return newTensor[U](0)
    let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
    return newTensorWith[U]([], convertedReductionValue(input[0], U))
  var shape = arg.shape
  shape[axis] = 1
  let axisLength = arg.shape[axis]
  if axisLength == 0: return newTensorWith[U](shape, identity)
  result = newTensorUninit[U](shape)
  if result.size == 0: return
  let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
  let output {.restrict.} = cast[ptr UncheckedArray[U]](result.get_offset_ptr)
  if arg.is_C_contiguous:
    var outer = 1'u
    for dimension in 0 ..< axis: outer *= arg.shape[dimension].uint
    var inner = 1'u
    for dimension in axis + 1 ..< arg.rank: inner *= arg.shape[dimension].uint
    let length = axisLength.uint
    when U is bool and T is Complex and multiply:
      if inner > 1:
        reduceComplexTruthColumns(result.get_offset_ptr, arg.get_offset_ptr,
          outer, length, inner)
        return
    for batch in 0'u ..< outer:
      let base = batch * length * inner
      if inner == 1:
        output[batch] = reducePackedIntegerOutput(input, base, length, multiply, U)
      else:
        let destination = batch * inner
        when U is bool and T is Complex:
          # Each logical column can finish at its first decisive component pair.
          for column in 0'u ..< inner:
            var value = identity
            for position in 0'u ..< length:
              if convertedReductionValue(input[base + position * inner + column], bool) != multiply:
                value = not multiply
                break
            output[destination + column] = value
        else:
          for column in 0'u ..< inner:
            output[destination + column] = convertedReductionValue(input[base + column], U)
          for position in 1'u ..< length:
            let source = base + position * inner
            for column in 0'u ..< inner:
              let index = destination + column
              combineIntegerOutput(output[index], input[source + column], U, multiply)
    return
  for index in 0 ..< result.size:
    var remainder = index
    var base = 0
    for dimension in countdown(arg.rank - 1, 0):
      if dimension != axis:
        base += (remainder mod shape[dimension]) * arg.strides[dimension]
        remainder = remainder div shape[dimension]
    var value = identity
    for position in 0 ..< axisLength:
      combineIntegerOutput(value, input[base + position * arg.strides[axis]], U, multiply)
    output[index] = value

func reductionAxis(rank, axis: int): int =
  ## Scalars accept axis 0 and -1 without acquiring a dimension.
  let dimensions = max(1, rank)
  if axis < -dimensions or axis >= dimensions:
    raise newException(IndexDefect, "Reduction axis out of bounds")
  if axis < 0: axis + dimensions else: axis

func productIdentity[T](): T =
  when T is Complex:
    type F = T.T
    complex(1.F, 0.F)
  else:
    1.T

when defined(macosx):
  proc packedFloat32Axis(arg: Tensor[float32], axis: int;
                        multiply, average: static[bool]): Tensor[float32] =
    ## The packed path allocates only its final output, including for means.
    var shape = arg.shape
    shape[axis] = 1
    result = newTensorUninit[float32](shape)
    var outer = 1
    for dimension in 0 ..< axis: outer *= arg.shape[dimension]
    var inner = 1
    for dimension in axis + 1 ..< arg.rank: inner *= arg.shape[dimension]
    reducePackedFloat32(result.get_offset_ptr, arg.get_offset_ptr,
                       outer, arg.shape[axis], inner, multiply, average)
proc packedFloat64Axis(arg: Tensor[float64], axis: int;
                      multiply, average: static[bool]): Tensor[float64] =
  ## The packed path allocates only its final output, including for means.
  var shape = arg.shape
  shape[axis] = 1
  result = newTensorUninit[float64](shape)
  var outer = 1'u
  for dimension in 0 ..< axis: outer *= uint(arg.shape[dimension])
  var inner = 1'u
  for dimension in axis + 1 ..< arg.rank: inner *= uint(arg.shape[dimension])
  reducePackedFloat64Axis(result.get_offset_ptr, arg.get_offset_ptr,
                         outer, uint(arg.shape[axis]), inner, multiply, average)

proc sumAlongAxis[T](arg: Tensor[T], axis: int): Tensor[T] =
  if arg.rank == 0:
    if arg.size == 0: return newTensor[T](0)
    return arg.clone
  if arg.size == 0:
    var shape = arg.shape
    shape[axis] = 1
    return newTensor[T](shape)
  when T is float64:
    if arg.is_C_contiguous:
      return packedFloat64Axis(arg, axis, false, false)
  elif defined(macosx) and T is float32:
    if arg.is_C_contiguous:
      return packedFloat32Axis(arg, axis, false, false)
  arg.reduce_axis_inline(axis):
    x+=y

proc reduceCascadedTotal[T, U](arg: Tensor[T]; dtype: typedesc[U];
                              average: static[bool]): U
proc reduceCascadedTensorAxis[T, U](arg: Tensor[T], axis: int;
    dtype: typedesc[U]; average: static[bool]): Tensor[U]

proc sum*[T](arg: Tensor[T]): sumProductElement(T) =
  ## Compute the sum of all elements.
  ## Integers and booleans promote to int64 before modulo-2^64 arithmetic.
  when T is SomeInteger or T is bool:
    reduceIntegers(arg, false, int64)
  elif T is Complex:
    reduceCascadedTotal(arg, T, false)
  else:
    when T is float64:
      if arg.size == 0: return 0.0
      if arg.is_C_contiguous:
        reducePackedFloat64(addr result, arg.get_offset_ptr, uint(arg.size), false)
        return
    elif defined(macosx) and T is float32:
      if arg.size == 0: return 0'f32
      if arg.is_C_contiguous:
        vDSP_sve(arg.get_offset_ptr, 1.clong, addr result, arg.size.culong)
        return
    arg.reduce_inline():
      x+=y
proc sum*[T](arg: Tensor[T], axis: int): Tensor[sumProductElement(T)] {.noinit.} =
  ## Sum along a positive or negative axis, retaining its dimension.
  ## Empty reductions produce zeros; scalar results are independent.
  ## Integers and booleans promote to int64 with modulo-2^64 arithmetic.
  let axis = reductionAxis(arg.rank, axis)
  when T is SomeInteger or T is bool:
    reduceIntegerAxis(arg, axis, false, int64)
  elif T is Complex:
    reduceCascadedTensorAxis(arg, axis, T, false)
  else:
    sumAlongAxis(arg, axis)

proc reduceComplexProductTotal[T, U](arg: Tensor[T]; dtype: typedesc[U]): U
proc reduceComplexProductTensorAxis[T, U](arg: Tensor[T], axis: int;
                                         dtype: typedesc[U]): Tensor[U]

proc product*[T](arg: Tensor[T]): sumProductElement(T) =
  ## Compute the product of all elements; an empty product is one.
  ## Integers and booleans promote to int64 before modulo-2^64 arithmetic.
  ## Packed float32/float64 products use partial carries; rounding and overflow
  ## can depend on their grouping rather than a left-to-right multiplication order.
  when T is SomeInteger or T is bool:
    reduceIntegers(arg, true, int64)
  elif T is Complex:
    reduceComplexProductTotal(arg, T)
  else:
    if arg.size == 0: return productIdentity[T]()
    when T is float64:
      if arg.is_C_contiguous:
        reducePackedFloat64(addr result, arg.get_offset_ptr, uint(arg.size), true)
        return
    elif defined(macosx) and T is float32:
      if arg.is_C_contiguous:
        reducePackedFloat32(addr result, arg.get_offset_ptr, 1, arg.size, 1, true, false)
        return
    arg.reduce_inline():
      x*=y

proc product*[T](arg: Tensor[T], axis: int): Tensor[sumProductElement(T)] {.noinit.}=
  ## Product along a positive or negative axis, retaining its dimension.
  ## Empty reductions produce ones; scalar results are independent copies.
  ## Integers and booleans promote to int64 with modulo-2^64 arithmetic.
  let axis = reductionAxis(arg.rank, axis)
  when T is SomeInteger or T is bool:
    reduceIntegerAxis(arg, axis, true, int64)
  elif T is Complex:
    reduceComplexProductTensorAxis(arg, axis, T)
  else:
    if arg.rank == 0:
      if arg.size == 0: return newTensor[T](0)
      return arg.clone
    if arg.size == 0:
      var shape = arg.shape
      shape[axis] = 1
      return newTensorWith[T](shape, productIdentity[T]())
    when T is float64:
      if arg.is_C_contiguous:
        return packedFloat64Axis(arg, axis, true, false)
    elif defined(macosx) and T is float32:
      if arg.is_C_contiguous:
        return packedFloat32Axis(arg, axis, true, false)
    arg.reduce_axis_inline(axis):
      x.melwise_mul(y)

proc mean*[T: SomeInteger or bool](arg: Tensor[T]): T =
  ## Integer and boolean means require an explicit floating ``dtype``.
  ## Without one, raise ``ValueError`` rather than perform integer division.
  raise newException(ValueError, "Mean requires a floating-point or complex dtype")

proc mean*[T: SomeInteger or bool](arg: Tensor[T], axis: int): Tensor[T] =
  ## Integer and boolean means require an explicit floating ``dtype``.
  ## Dtype rejection precedes axis validation, including for empty inputs.
  raise newException(ValueError, "Mean requires a floating-point or complex dtype")

proc mean*[T: SomeFloat](arg: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements
  arg.sum / arg.size.T


proc mean*[T: SomeFloat](arg: Tensor[T], axis: int): Tensor[T] {.noinit,inline.}=
  ## Mean along a positive or negative axis, retaining its dimension.
  ## Empty reductions produce NaNs; unreduced empty dimensions are preserved.
  let axis = reductionAxis(arg.rank, axis)
  if arg.rank == 0: return sumAlongAxis(arg, axis)
  when T is float64:
    if arg.size > 0 and arg.is_C_contiguous:
      return packedFloat64Axis(arg, axis, false, true)
  elif defined(macosx) and T is float32:
    if arg.size > 0 and arg.is_C_contiguous:
      return packedFloat32Axis(arg, axis, false, true)
  sumAlongAxis(arg, axis) / arg.shape[axis].T



proc sumConvertedBlock[T, U](input: ptr UncheckedArray[T], base, count: uint;
                              dtype: typedesc[U]): U =
  ## Reduce at most 1024 values; input casts precede any rounding.
  when defined(macosx) and defined(arm64) and U is float32 and
       (T is int32 or T is uint32 or T is float64):
    sumConvertedFloat32Block(input, base, count)
  elif T is SomeInteger and
       (sizeof(T) <= 2 or (U is float64 and sizeof(T) <= 4)):
    # These inputs are exactly representable in U; bounded integer sums cannot overflow.
    U(reducePackedIntegers(input, base, count, false))
  else:
    var a, b, c, d: U
    let bulk = count and not 3'u
    var index = 0'u
    while index < bulk:
      a += convertedReductionValue(input[base + index], U)
      b += convertedReductionValue(input[base + index + 1], U)
      c += convertedReductionValue(input[base + index + 2], U)
      d += convertedReductionValue(input[base + index + 3], U)
      index += 4
    result = ((a + b) + c) + d
    while index < count:
      result += convertedReductionValue(input[base + index], U)
      inc index

template mergeConvertedPartial(bins, blocks, partial: untyped) =
  # Occupied binary-counter slots were written by an earlier partial.
  var value = partial
  var occupied = blocks
  var level = 0
  while (occupied and 1'u) != 0:
    value = bins[level] + value
    occupied = occupied shr 1
    inc level
  bins[level] = value
  inc blocks

template finishConvertedPartials(bins, blocks: untyped): untyped =
  block:
    var value: typeof(bins[0])
    var occupied = blocks
    var level = 0
    while occupied != 0:
      if (occupied and 1'u) != 0:
        value = bins[level] + value
      occupied = occupied shr 1
      inc level
    value

proc sumConvertedSpan[T, U](input: ptr UncheckedArray[T], base, count: uint;
                             dtype: typedesc[U]): U =
  ## Bounded carries avoid losing low bits in long float32 input conversions.
  when T is bool:
    # Boolean counts fit int64 and need only one final floating conversion.
    return U(reducePackedIntegers(input, base, count, false))
  else:
    if count <= 1024:
      return sumConvertedBlock(input, base, count, U)
    var bins {.noinit.}: array[sizeof(int) * 8, U]
    var blocks = 0'u
    var index = 0'u
    while index < count:
      let length = min(1024'u, count - index)
      mergeConvertedPartial(bins, blocks,
                            sumConvertedBlock(input, base + index, length, U))
      index += length
    finishConvertedPartials(bins, blocks)

template sumConvertedValues(values: untyped; U: typedesc): untyped =
  block:
    var bins {.noinit.}: array[sizeof(int) * 8, U]
    var partials: array[4, U]
    var blocks = 0'u
    var position = 0'u
    for value in values:
      partials[position and 3'u] += convertedReductionValue(value, U)
      inc position
      if position == 128:
        mergeConvertedPartial(bins, blocks,
                              ((partials[0] + partials[1]) + partials[2]) + partials[3])
        for partial in partials.mitems: partial = default(U)
        position = 0
    if position != 0:
      mergeConvertedPartial(bins, blocks,
                            ((partials[0] + partials[1]) + partials[2]) + partials[3])
    finishConvertedPartials(bins, blocks)

iterator stridedReductionValues[T](input: ptr UncheckedArray[T], base, count, stride: int): T =
  ## Negative and broadcast strides keep their signed address arithmetic.
  for position in 0 ..< count:
    yield input[base + position * stride]

template productConvertedValues(values: untyped; U: typedesc): untyped =
  block:
    var product = productIdentity[U]()
    for value in values:
      when typeof(value) is bool:
        if not value:
          product = default(U)
          break
      else:
        multiplyProductValue(product, convertedReductionValue(value, U))
    product

proc productConvertedSpan[T, U](input: ptr UncheckedArray[T], base, count: uint;
                                 dtype: typedesc[U]): U =
  ## Cast on read; float32 overflow and underflow remain float32 operations.
  when T is bool:
    productConvertedValues(stridedReductionValues(input, base.int, count.int, 1), U)
  elif defined(macosx):
    reduceConvertedProductSpan(unsafeAddr input[base], count, U)
  else:
    productConvertedValues(stridedReductionValues(input, base.int, count.int, 1), U)

proc reduceComplexProductTotal[T, U](arg: Tensor[T]; dtype: typedesc[U]): U =
  ## Dense storage follows the packed tree; signed vectors keep logical order.
  if arg.size == 0: return productIdentity[U]()
  let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
  if arg.is_C_contiguous or arg.is_F_contiguous:
    return productComplexSpan(input, 0'u, arg.size.uint, 1'u, U)
  if arg.rank == 1:
    return productComplexSpan(input, 0, arg.size.uint, arg.strides[0], U)
  productConvertedValues(arg, U)

proc reduceComplexProductTensorAxis[T, U](arg: Tensor[T], axis: int;
                                         dtype: typedesc[U]): Tensor[U] =
  ## Cast directly into bounded carries and allocate only the final output.
  withCompilerOptimHints()
  let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
  if arg.rank == 0:
    if arg.size == 0: return newTensor[U](0)
    var value = productIdentity[U]()
    multiplyProductValue(value, convertedReductionValue(input[0], U))
    return newTensorWith[U]([], value)
  var shape = arg.shape
  shape[axis] = 1
  let length = arg.shape[axis]
  if length == 0: return newTensorWith[U](shape, productIdentity[U]())
  result = newTensorUninit[U](shape)
  if result.size == 0: return
  if arg.is_C_contiguous:
    var outer = 1'u
    var inner = 1'u
    for dimension in 0 ..< axis: outer *= arg.shape[dimension].uint
    for dimension in axis + 1 ..< arg.rank: inner *= arg.shape[dimension].uint
    productComplexAxis(result.get_offset_ptr, arg.get_offset_ptr,
      outer, length.uint, inner)
    return
  let output {.restrict.} = cast[ptr UncheckedArray[U]](result.get_offset_ptr)
  for index in 0 ..< result.size:
    var remainder = index
    var base = 0
    for dimension in countdown(arg.rank - 1, 0):
      if dimension != axis:
        base += (remainder mod shape[dimension]) * arg.strides[dimension]
        remainder = remainder div shape[dimension]
    if arg.strides[axis] == 1 or arg.strides[axis] == -1:
      output[index] = productComplexSpan(input, base, length.uint, arg.strides[axis], U)
    else:
      output[index] = productConvertedValues(
        stridedReductionValues(input, base, length, arg.strides[axis]), U)

proc reduceConvertedAxis[T, U](arg: Tensor[T], axis: int;
                               dtype: typedesc[U]; average: static[bool];
                               multiply: static[bool] = false): Tensor[U] =
  ## The fresh output is the only buffer; signed input strides remain intact.
  static: doAssert not (average and multiply)
  withCompilerOptimHints()
  if arg.rank == 0:
    if arg.size == 0: return newTensor[U](0)
    let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
    when multiply: return newTensorWith[U]([], convertedReductionValue(input[0], U))
    else: return newTensorWith[U]([], U(0) + convertedReductionValue(input[0], U))
  var shape = arg.shape
  shape[axis] = 1
  let axisLength = arg.shape[axis]
  if axisLength == 0:
    when average: return newTensorWith[U](shape, U(NaN))
    elif multiply: return newTensorWith[U](shape, U(1))
    else: return newTensor[U](shape)
  result = newTensorUninit[U](shape)
  if result.size == 0: return
  let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
  let output {.restrict.} = cast[ptr UncheckedArray[U]](result.get_offset_ptr)
  when average:
    let divisor = U(axisLength)
  template normalized(value: untyped): untyped =
    when average: value / divisor
    else: value
  template reduced(values: untyped): untyped =
    when multiply: productConvertedValues(values, U)
    else: normalized(sumConvertedValues(values, U))
  if arg.is_C_contiguous:
    var outer = 1'u
    for dimension in 0 ..< axis: outer *= arg.shape[dimension].uint
    var inner = 1'u
    for dimension in axis + 1 ..< arg.rank: inner *= arg.shape[dimension].uint
    let length = axisLength.uint
    when defined(macosx) and defined(arm64) and multiply and not (T is bool):
      if inner > 1:
        reduceConvertedProductColumns(result.get_offset_ptr, arg.get_offset_ptr,
                                      outer, length, inner)
        return
    when defined(macosx) and defined(arm64) and U is float32 and
         (T is bool or T is int8) and not multiply:
      if inner > 1 and length <= 128:
        reduceConvertedFloat32Columns(result.get_offset_ptr, input,
                                      outer, length, inner, average)
        return
    for batch in 0'u ..< outer:
      let base = batch * length * inner
      if inner == 1:
        when multiply: output[batch] = productConvertedSpan(input, base, length, U)
        else: output[batch] = normalized(sumConvertedSpan(input, base, length, U))
      elif multiply or length > 128:
        let destination = batch * inner
        for column in 0'u ..< inner:
          output[destination + column] = reduced(
            stridedReductionValues(input, int(base + column), axisLength, int(inner)))
      else:
        let destination = batch * inner
        for column in 0'u ..< inner:
          output[destination + column] = U(0)
        let bulk = length and not 3'u
        var position = 0'u
        while position < bulk:
          let source = base + position * inner
          for column in 0'u ..< inner:
            let a = convertedReductionValue(input[source + column], U)
            let b = convertedReductionValue(input[source + inner + column], U)
            let c = convertedReductionValue(input[source + 2'u * inner + column], U)
            let d = convertedReductionValue(input[source + 3'u * inner + column], U)
            output[destination + column] += ((a + b) + c) + d
          position += 4
        while position < length:
          let source = base + position * inner
          for column in 0'u ..< inner:
            output[destination + column] += convertedReductionValue(input[source + column], U)
          inc position
        when average:
          for column in 0'u ..< inner:
            output[destination + column] /= divisor
    return
  for index in 0 ..< result.size:
    var remainder = index
    var base = 0
    for dimension in countdown(arg.rank - 1, 0):
      if dimension != axis:
        base += (remainder mod shape[dimension]) * arg.strides[dimension]
        remainder = remainder div shape[dimension]
    output[index] = reduced(
      stridedReductionValues(input, base, axisLength, arg.strides[axis]))

proc reduceCascadedTotal[T, U](arg: Tensor[T]; dtype: typedesc[U];
                              average: static[bool]): U =
  ## Dense views reduce in storage order; other views retain signed strides.
  warnComplexToRealCast(T, U, arg.size)
  if arg.size == 0:
    when average:
      when U is Complex: return finishCascadedReduction(default(U), 0, true)
      else: return U(NaN)
    else:
      return default(U)
  let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
  var total: U
  if arg.is_C_contiguous or arg.is_F_contiguous:
    total = sumCascadedSpan(input, 0'u, arg.size.uint, 1'u, U)
  elif arg.rank == 1:
    total = sumCascadedSpan(input, 0, arg.size.uint, arg.strides[0], U)
  else:
    total = sumConvertedValues(arg, U)
  finishCascadedReduction(total, arg.size, average)

proc reduceCascadedTensorAxis[T, U](arg: Tensor[T], axis: int;
    dtype: typedesc[U]; average: static[bool]): Tensor[U] =
  ## Only the independent final output is allocated, including scalar results.
  withCompilerOptimHints()
  warnComplexToRealCast(T, U, arg.size)
  let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
  if arg.rank == 0:
    if arg.size == 0: return newTensor[U](0)
    let total = default(U) + convertedReductionValue(input[0], U)
    return newTensorWith[U]([], finishCascadedReduction(total, 1, average))
  var shape = arg.shape
  shape[axis] = 1
  let length = arg.shape[axis]
  if length == 0:
    when average:
      when U is Complex:
        return newTensorWith[U](shape, finishCascadedReduction(default(U), 0, true))
      else:
        return newTensorWith[U](shape, U(NaN))
    else:
      return newTensor[U](shape)
  result = newTensorUninit[U](shape)
  if result.size == 0: return
  if arg.is_C_contiguous:
    var outer = 1'u
    var inner = 1'u
    for dimension in 0 ..< axis: outer *= arg.shape[dimension].uint
    for dimension in axis + 1 ..< arg.rank: inner *= arg.shape[dimension].uint
    reduceCascadedAxis(result.get_offset_ptr, arg.get_offset_ptr,
      outer, length.uint, inner, average)
    return
  let output {.restrict.} = cast[ptr UncheckedArray[U]](result.get_offset_ptr)
  let stride = arg.strides[axis]
  for index in 0 ..< result.size:
    var remainder = index
    var base = 0
    for dimension in countdown(arg.rank - 1, 0):
      if dimension != axis:
        base += (remainder mod shape[dimension]) * arg.strides[dimension]
        remainder = remainder div shape[dimension]
    let total =
      if stride == 1 or stride == -1:
        sumCascadedSpan(input, base, length.uint, stride, U)
      else:
        sumCascadedSpan(input, base, length.uint, stride, U, false)
    output[index] = finishCascadedReduction(total, length, average)

proc mean*[T: SomeNumber or bool, U: SomeFloat](arg: Tensor[T];
                                                dtype: typedesc[U]): U =
  ## Mean after casting each real input to the requested floating ``dtype``.
  ## Unsigned inputs convert numerically, including the uint64 upper half.
  ## Empty inputs produce NaN. No converted input tensor is allocated.
  when T is U:
    if arg.is_C_contiguous: return arg.mean
  if arg.size == 0: return U(NaN)
  if arg.is_C_contiguous:
    let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
    return sumConvertedSpan(input, 0'u, arg.size.uint, U) / U(arg.size)
  sumConvertedValues(arg, U) / U(arg.size)

proc mean*[T: SomeNumber or bool, U: SomeFloat](arg: Tensor[T], axis: int;
                                                dtype: typedesc[U]): Tensor[U] =
  ## Mean in the requested floating dtype, retaining the reduced dimension.
  ## Negative/scalar axes follow ``mean``; empty reductions produce NaNs.
  ## Views are cast on read. Only the independent final output is allocated.
  let axis = reductionAxis(arg.rank, axis)
  when T is U and (T is float64 or (defined(macosx) and T is float32)):
    if arg.rank > 0 and arg.size > 0 and arg.is_C_contiguous:
      return arg.mean(axis)
  reduceConvertedAxis(arg, axis, U, true)

proc mean*[T: SomeNumber or bool or Complex; F: SomeFloat](arg: Tensor[T];
    dtype: typedesc[Complex[F]]): Complex[F] =
  ## Cast before component-width CPU summation and rounded reciprocal division.
  ## Empty means have two NaN components. No converted input tensor is allocated.
  reduceCascadedTotal(arg, Complex[F], true)

proc mean*[T: SomeNumber or bool or Complex; F: SomeFloat](arg: Tensor[T], axis: int;
    dtype: typedesc[Complex[F]]): Tensor[Complex[F]] =
  ## Mean in the requested complex dtype, retaining the reduced dimension.
  ## Signed views and negative axes are supported; only the output is allocated.
  let axis = reductionAxis(arg.rank, axis)
  reduceCascadedTensorAxis(arg, axis, Complex[F], true)

proc mean*[T: Complex[float32] or Complex[float64]](arg: Tensor[T]): T {.inline.} =
  ## Sum in the input component width, then apply complex reciprocal division.
  reduceCascadedTotal(arg, T, true)

proc mean*[T: Complex[float32] or Complex[float64]](arg: Tensor[T], axis: int): Tensor[T] {.inline.} =
  ## Complex mean retaining its axis; scalar sums execute the zero identity.
  let axis = reductionAxis(arg.rank, axis)
  reduceCascadedTensorAxis(arg, axis, T, true)

proc mean*[F: SomeFloat; U: SomeFloat](arg: Tensor[Complex[F]];
    dtype: typedesc[U]): U =
  ## Drop imaginary components before real accumulation; warn once if nonempty.
  reduceCascadedTotal(arg, U, true)

proc mean*[F: SomeFloat; U: SomeFloat](arg: Tensor[Complex[F]], axis: int;
    dtype: typedesc[U]): Tensor[U] =
  ## Cast real components on read, retaining the axis in an independent output.
  let axis = reductionAxis(arg.rank, axis)
  reduceCascadedTensorAxis(arg, axis, U, true)

proc mean*[F: SomeFloat; U: SomeInteger or bool](arg: Tensor[Complex[F]];
    dtype: typedesc[U]): U =
  ## Integer/bool mean dtypes reject before casts or warnings, including empties.
  raise newException(ValueError, "Mean output dtype must be floating point or complex")

proc mean*[F: SomeFloat; U: SomeInteger or bool](arg: Tensor[Complex[F]], axis: int;
    dtype: typedesc[U]): Tensor[U] =
  ## Dtype rejection precedes axis validation and discarded-imaginary warnings.
  raise newException(ValueError, "Mean output dtype must be floating point or complex")

proc sum*[T: SomeNumber or bool or Complex; F: SomeFloat](arg: Tensor[T];
    dtype: typedesc[Complex[F]]): Complex[F] =
  ## Cast on read before component-width CPU summation; empty sums are zero.
  ## No converted input tensor is allocated, including unsigned numeric casts.
  reduceCascadedTotal(arg, Complex[F], false)

proc sum*[T: SomeNumber or bool or Complex; F: SomeFloat](arg: Tensor[T], axis: int;
    dtype: typedesc[Complex[F]]): Tensor[Complex[F]] =
  ## Sum in the requested complex dtype, retaining the reduced dimension.
  ## Signed views and negative/scalar axes allocate only the independent output.
  let axis = reductionAxis(arg.rank, axis)
  reduceCascadedTensorAxis(arg, axis, Complex[F], false)

proc sum*[T: SomeNumber or bool or Complex, U: SomeFloat](arg: Tensor[T];
                                                          dtype: typedesc[U]): U =
  ## Sum after casting each input to the requested floating ``dtype``.
  ## Unsigned inputs convert numerically; empty inputs produce zero.
  ## Complex inputs discard imaginary components and warn once if nonempty.
  ## No converted input tensor is allocated.
  when T is Complex:
    reduceCascadedTotal(arg, U, false)
  else:
    if arg.size == 0: return U(0)
    when T is U:
      if arg.is_C_contiguous: return arg.sum
    if arg.is_C_contiguous:
      let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
      return sumConvertedSpan(input, 0'u, arg.size.uint, U)
    sumConvertedValues(arg, U)

proc sum*[T: SomeNumber or bool or Complex, U: SomeFloat](arg: Tensor[T], axis: int;
                                                          dtype: typedesc[U]): Tensor[U] =
  ## Sum in the requested floating dtype, retaining the reduced dimension.
  ## Empty reductions produce zeros; scalar axes 0 and -1 stay rank zero.
  ## Views are cast on read. Only the independent final output is allocated.
  let axis = reductionAxis(arg.rank, axis)
  when T is Complex:
    reduceCascadedTensorAxis(arg, axis, U, false)
  else:
    when T is U and (T is float64 or (defined(macosx) and T is float32)):
      if arg.rank > 0 and arg.size > 0 and arg.is_C_contiguous:
        return arg.sum(axis)
    reduceConvertedAxis(arg, axis, U, false)

proc product*[T: SomeNumber or bool or Complex; F: SomeFloat](arg: Tensor[T];
    dtype: typedesc[Complex[F]]): Complex[F] =
  ## Cast on read, then multiply in the complex output's component width.
  ## Empty products are 1+0i. Packed vectors fold before scalar tails;
  ## grouping can change rounding and intermediate overflow or underflow.
  reduceComplexProductTotal(arg, Complex[F])

proc product*[T: SomeNumber or bool or Complex; F: SomeFloat](arg: Tensor[T], axis: int;
    dtype: typedesc[Complex[F]]): Tensor[Complex[F]] =
  ## Retain the reduced dimension, with one independent final allocation.
  ## Empty axes produce 1+0i; scalar axes multiply the identity before storing.
  ## Signed and broadcast views are read in place without a conversion tensor.
  let axis = reductionAxis(arg.rank, axis)
  reduceComplexProductTensorAxis(arg, axis, Complex[F])

proc product*[T: SomeNumber or bool or Complex, U: SomeFloat](arg: Tensor[T];
                                                              dtype: typedesc[U]): U =
  ## Cast on read and multiply in the requested floating dtype.
  ## Unsigned inputs convert numerically; complex inputs discard imaginary parts
  ## with the shared cast warning. Empty products are one.
  warnComplexToRealCast(T, U, arg.size)
  if arg.size == 0: return U(1)
  when T is U:
    if arg.is_C_contiguous: return arg.product
  if arg.is_C_contiguous:
    let input = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
    return productConvertedSpan(input, 0'u, arg.size.uint, U)
  productConvertedValues(arg, U)

proc product*[T: SomeNumber or bool or Complex, U: SomeFloat](arg: Tensor[T], axis: int;
                                                              dtype: typedesc[U]): Tensor[U] =
  ## Retained-axis products allocate only the independent final output.
  ## Empty axes produce one; scalar signs and signed views are preserved.
  ## Complex numeric casts discard imaginary parts with the shared warning.
  let axis = reductionAxis(arg.rank, axis)
  warnComplexToRealCast(T, U, arg.size)
  when T is U and (T is float64 or (defined(macosx) and T is float32)):
    if arg.rank > 0 and arg.size > 0 and arg.is_C_contiguous:
      return arg.product(axis)
  reduceConvertedAxis(arg, axis, U, false, true)

proc mean*[T: SomeNumber or bool, U: SomeInteger or bool](arg: Tensor[T];
                                                         dtype: typedesc[U]): U =
  ## Integer and boolean output dtypes are invalid for means, including empties.
  raise newException(ValueError, "Mean output dtype must be floating point")

proc mean*[T: SomeNumber or bool, U: SomeInteger or bool](arg: Tensor[T], axis: int;
                                                         dtype: typedesc[U]): Tensor[U] =
  ## Integer and boolean output dtypes are invalid for axis means.
  raise newException(ValueError, "Mean output dtype must be floating point")

proc sum*[T: SomeNumber or bool or Complex, U: SomeInteger or bool](arg: Tensor[T];
                                                                  dtype: typedesc[U]): U =
  ## Cast on read and sum modulo the output bit width; bool sums are logical OR.
  ## Numeric outputs support uint8 and signed integers. Wider unsigned outputs
  ## raise ValueError unless the input is empty. Empty sums are zero/false.
  ## Floating inputs truncate; out-of-range/nonfinite integer casts are undefined.
  ## Complex numeric casts drop imaginary parts and share the cast warning.
  warnComplexToRealCast(T, U, arg.size)
  reduceIntegers(arg, false, U)

proc sum*[T: SomeNumber or bool or Complex, U: SomeInteger or bool](arg: Tensor[T], axis: int;
                                                                  dtype: typedesc[U]): Tensor[U] =
  ## Retained-axis integer sums or boolean ORs, with only an output allocation.
  ## Views and negative/scalar axes follow sum. Numeric carries wrap to dtype;
  ## wider unsigned outputs reject nonempty inputs, even scalars.
  ## Floating inputs must truncate to a value representable in dtype.
  let axis = reductionAxis(arg.rank, axis)
  warnComplexToRealCast(T, U, arg.size)
  reduceIntegerAxis(arg, axis, false, U)

proc product*[T: SomeNumber or bool or Complex, U: SomeInteger or bool](arg: Tensor[T];
                                                                       dtype: typedesc[U]): U =
  ## Cast on read and multiply modulo dtype; bool products are logical AND.
  ## Empty products are one/true. Wider unsigned outputs reject nonempty inputs.
  ## Floating inputs must truncate to a value representable in dtype.
  ## Complex numeric casts warn; bool checks both components without warning.
  warnComplexToRealCast(T, U, arg.size)
  reduceIntegers(arg, true, U)

proc product*[T: SomeNumber or bool or Complex, U: SomeInteger or bool](arg: Tensor[T], axis: int;
                                                                       dtype: typedesc[U]): Tensor[U] =
  ## Retained-axis wrapping products or boolean ANDs; views are read in place.
  ## Empty reduced axes yield one/true. Wider unsigned outputs reject nonempty
  ## inputs, including scalars. Floating casts must fit the requested dtype.
  let axis = reductionAxis(arg.rank, axis)
  warnComplexToRealCast(T, U, arg.size)
  reduceIntegerAxis(arg, axis, true, U)

proc amin*[T: SomeNumber or bool or Complex](arg: Tensor[T];
                                            keepdim: bool = false): Tensor[T] =
  ## Reduce all dimensions to an independent tensor with the input dtype.
  ## ``keepdim`` retains singleton dimensions; scalars stay rank zero.
  ## Empty inputs have no minimum identity and raise ``ValueError``.
  ## Nonempty complex and wider unsigned inputs are unsupported.
  reduceExtrema(arg, [], keepdim, false)

proc amin*[T: SomeNumber or bool or Complex](arg: Tensor[T], axis: int;
                                            keepdim: bool = false): Tensor[T] =
  ## Reduce one axis, accepting negative axes and scalar axes 0/-1.
  ## A zero-length reduced axis raises ``IndexDefect``; empty outputs are valid.
  ## Reads strided views without copying and allocates independent output storage.
  reduceExtrema(arg, [axis], keepdim, false)

proc amin*[T: SomeNumber or bool or Complex](arg: Tensor[T], axes: openArray[int];
                                            keepdim: bool = false): Tensor[T] =
  ## Reduce distinct axes without intermediate tensors or dtype promotion.
  ## Negative axes normalize before duplicate checks; an empty list reduces all.
  ## NaNs propagate. Native vector and scalar ties can retain different zero signs.
  ## Invalid axes, empty reductions and duplicates reject before unsupported dtypes.
  reduceExtrema(arg, axes, keepdim, false)

proc amax*[T: SomeNumber or bool or Complex](arg: Tensor[T];
                                            keepdim: bool = false): Tensor[T] =
  ## Maximum counterpart of ``amin``; dtype, shape and allocation rules match.
  reduceExtrema(arg, [], keepdim, true)

proc amax*[T: SomeNumber or bool or Complex](arg: Tensor[T], axis: int;
                                            keepdim: bool = false): Tensor[T] =
  ## Reduce one axis to independent maxima, optionally retaining that dimension.
  ## Negative/scalar axes, NaNs and empty-output rules match ``amin``.
  reduceExtrema(arg, [axis], keepdim, true)

proc amax*[T: SomeNumber or bool or Complex](arg: Tensor[T], axes: openArray[int];
                                            keepdim: bool = false): Tensor[T] =
  ## Reduce distinct axes to independent maxima, preserving the input dtype.
  ## An empty list reduces all dimensions; validation and view rules match ``amin``.
  reduceExtrema(arg, axes, keepdim, true)

proc min*[T](arg: Tensor[T]): T =
  ## Compute the min of all elements
  reduce_inline(arg):
    x = min(x,y)

proc min*[T](arg: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the min along an axis
  returnEmptyIfEmpty(arg)
  arg.reduce_axis_inline(axis):
    for ex, ey in mzip(x,y):
      ex = min(ex,ey)

proc max*[T](arg: Tensor[T]): T =
  ## Compute the max of all elements
  arg.reduce_inline():
    x = max(x,y)

proc max*[T](arg: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the max along an axis
  returnEmptyIfEmpty(arg)
  arg.reduce_axis_inline(axis):
    for ex, ey in mzip(x,y):
      ex = max(ex,ey)

proc variance*[T: SomeFloat](arg: Tensor[T]): T =
  ## Compute the sample variance of all elements
  ## The normalization is by (n-1), also known as Bessel's correction,
  ## which partially correct the bias of estimating a population variance from a sample of this population.
  let mean = arg.mean()
  result = arg.fold_inline() do:
    # Initialize to the first element
    x = square(y - mean)
  do:
    # Fold in parallel by summing remaning elements
    x += square(y - mean)
  do:
    # Merge parallel folds
    x += y
  result /= (arg.size-1).T

proc variance*[T: SomeFloat](arg: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the variance of all elements
  ## The normalization is by the (n-1), like in the formal definition
  returnEmptyIfEmpty(arg)
  let mean = arg.mean(axis)
  result = arg.fold_axis_inline(Tensor[T], axis) do:
    # Initialize to the first element
    x = square(y - mean)
  do:
    # Fold in parallel by summing remaning elements
    for ex, ey, em in mzip(x,y,mean):
      ex += square(ey - em)
  do:
    # Merge parallel folds
    x += y
  result /= (arg.shape[axis]-1).T

proc std*[T: SomeFloat](arg: Tensor[T]): T {.inline.} =
  ## Compute the standard deviation of all elements
  ## The normalization is by the (n-1), like in the formal definition
  sqrt(arg.variance())

proc std*[T: SomeFloat](arg: Tensor[T], axis: int): Tensor[T] {.noinit,inline.} =
  ## Compute the standard deviation of all elements
  ## The normalization is by the (n-1), like in the formal definition
  returnEmptyIfEmpty(arg)
  sqrt(arg.variance(axis))

proc maxWithIndices*[T: SomeNumber or bool or Complex](arg: Tensor[T], axis: int;
    keepdim = false): tuple[values: Tensor[T], indices: Tensor[int64]] =
  ## Return independent maximum values and int64 indices within ``axis``.
  ## Values retain the input dtype and the first selected source bits.
  ## First ties and NaNs win; negative axes and signed/broadcast views are valid.
  ## The reduced axis is removed unless ``keepdim``; scalars remain rank zero.
  ## Bool is supported. Complex inputs reject after axis/empty-axis validation.
  ## Wider unsigned inputs support only scalars and valid empty outputs.
  reducePairedExtrema(arg, axis, keepdim, true)

proc argmax*[T: SomeNumber or bool or Complex](arg: Tensor[T];
    keepdim = false): Tensor[int64] =
  ## Return independent int64 indices of the first maximum in logical flat order.
  ## NaNs win over numeric values; the first NaN wins over later NaNs.
  ## With ``keepdim``, every input dimension remains as a singleton.
  ## Bool/complex inputs and empty whole reductions reject.
  ## Wider unsigned whole inputs reject, including scalars.
  reduceArgExtrema(arg, 0, true, keepdim, true)

proc argmax*[T: SomeNumber or bool or Complex](arg: Tensor[T], axis: int;
    keepdim = false): Tensor[int64] =
  ## Return independent int64 indices of the first maximum along ``axis``.
  ## Negative axes normalize; the reduced axis is removed unless ``keepdim``.
  ## Signed/broadcast views are read without copying. Empty reduced axes reject;
  ## unreduced empty axes produce empty outputs. Bool/complex inputs reject.
  ## Wider unsigned inputs only support empty outputs or explicit singleton axes.
  reduceArgExtrema(arg, axis, false, keepdim, true)

proc minWithIndices*[T: SomeNumber or bool or Complex](arg: Tensor[T], axis: int;
    keepdim = false): tuple[values: Tensor[T], indices: Tensor[int64]] =
  ## Minimum counterpart of ``maxWithIndices``, with values first and indices second.
  ## Both outputs have independent storage. First equal values/NaNs win, preserving
  ## the selected signed-zero bits. Dtype, axis and empty-output rules match.
  reducePairedExtrema(arg, axis, keepdim, false)

proc aminmax*[T: SomeNumber or bool or Complex](arg: Tensor[T];
    keepdim = false): tuple[min, max: Tensor[T]] =
  ## Return independent, dtype-preserving minimum and maximum tensors.
  ## Whole reductions return scalars, or singleton input dimensions with ``keepdim``.
  ## Signed/broadcast views are read without materializing the input. Native vector
  ## folds determine floating zero ties and NaN payload order; scalar bits survive.
  ## Empty whole reductions, complex and wider unsigned whole inputs reject.
  reduceAminmax(arg, 0, true, keepdim)

proc aminmax*[T: SomeNumber or bool or Complex](arg: Tensor[T], axis: int;
    keepdim = false): tuple[min, max: Tensor[T]] =
  ## Return minimum and maximum along one axis, removing it unless ``keepdim``.
  ## Negative axes normalize; scalars accept 0/-1 and stay scalar. First equal
  ## values and first NaN source bits win. Outputs share neither input nor each other.
  ## Empty reduced axes and complex inputs reject; retained empty axes are valid.
  ## Wider unsigned inputs support explicit scalar axes only, not empty outputs.
  reduceAminmax(arg, axis, false, keepdim)

proc argmin*[T: SomeNumber or bool or Complex](arg: Tensor[T];
    keepdim = false): Tensor[int64] =
  ## Return independent int64 indices of the first minimum in logical flat order.
  ## NaNs win over numeric values; the first NaN wins over later NaNs.
  ## With ``keepdim``, every input dimension remains as a singleton.
  ## Bool/complex inputs and empty whole reductions reject.
  ## Wider unsigned whole inputs reject, including scalars.
  reduceArgExtrema(arg, 0, true, keepdim, false)

proc argmin*[T: SomeNumber or bool or Complex](arg: Tensor[T], axis: int;
    keepdim = false): Tensor[int64] =
  ## Return independent int64 indices of the first minimum along ``axis``.
  ## Negative axes normalize; the reduced axis is removed unless ``keepdim``.
  ## Signed/broadcast views are read without copying. Empty reduced axes reject;
  ## unreduced empty axes produce empty outputs. Bool/complex inputs reject.
  ## Wider unsigned inputs only support empty outputs or explicit singleton axes.
  reduceArgExtrema(arg, axis, false, keepdim, false)

proc percentile*[T](arg: Tensor[T], p: int, isSorted = false): float =
  ## statistical percentile value of ``t``, where ``p`` percentile value
  ## is between ``0`` and ``100`` inclusively,
  ## and ``p=0`` gives the min value, ``p=100`` gives the max value
  ## and ``p=50`` gives the median value.
  ##
  ## If the input percentile does not match an element of `t` exactly
  ## the result is the linear interpolation between the neighbors.
  ##
  ## ``t`` does not need to be sorted, because ``percentile`` sorts
  ## a copy of the data itself. If ``isSorted`` is ``true`` however,
  ## no sorting is done.
  # TODO: we could in principle also return `T`, but then we cannot do
  # interpolation between values. Hm.
  if arg.size == 0: result = 0.0
  elif p <= 0: result = min(arg).float
  elif p >= 100: result = max(arg).float
  else:
    let a =
      if isSorted:
        (if arg.rank == 0: arg.reshape(arg.size.int) else: arg)
      else:
        sorted(arg.reshape(arg.size.int))
    let f = (arg.size - 1) * p / 100
    let i = floor(f).int
    if f == i.float: result = a[i].float
    else:
      # interpolate linearly
      let frac = f - i.float
      result = (a[i].float + (a[i+1] - a[i]).float * frac)

proc median*[T](arg: Tensor[T], isSorted = false): float {.inline.} =
  ## Compute the median of all elements (same as `arg.percentile(50)`)
  percentile(arg, 50, isSorted)

proc iqr*[T](arg: Tensor[T]): float =
  ## Returns the interquartile range of the 1D tensor `t`.
  ##
  ## The interquartile range (IQR) is the distance between the
  ## 25th and 75th percentile
  let tS = arg.sorted
  result = percentile(tS, 75, isSorted = true) -
           percentile(tS, 25, isSorted = true)

func cumulativeAxis(rank, axis: int): int =
  ## Scalars keep rank zero while accepting the same two axis spellings.
  let dimensions = max(1, rank)
  if axis < -dimensions or axis >= dimensions:
    raise newException(ValueError, "Cumulative axis is outside input rank")
  if axis < 0: axis + dimensions else: axis

proc scanCPU[T, U](arg: Tensor[T], axis: int, multiply: static[bool],
                   dtype: typedesc[U]; promoteIntegers: static[bool] = true;
                   reverse: bool = false): Tensor[U] =
  ## Cast on read; floating and complex carries survive rounded output stores.
  withCompilerOptimHints()
  mixin _
  template scanValue(value: untyped): untyped =
    when U is SomeFloat or U is Complex: convertedReductionValue(value, U)
    elif not promoteIntegers: value
    elif U is SomeInteger:
      when T is Complex: promotedInteger(value.re)
      else: promotedInteger(value)
    else: value
  when U is SomeFloat:
    const identity = when multiply: 1.0 else: 0.0
  template accumulate(acc, value: untyped) =
    when promoteIntegers and U is SomeInteger:
      when multiply: acc = acc *% value
      else: acc = acc +% value
    else:
      when multiply: acc = acc * value
      else: acc = acc + value
  let rank = arg.rank
  let scanAxis = cumulativeAxis(rank, axis)
  warnComplexToRealCast(T, U, arg.size)

  result = newTensorUninit[U](arg.shape)
  if arg.size == 0:
    return
  if rank == 0:
    when U is SomeInteger:
      result.storage.raw_buffer[0] = convertedReductionValue(arg.storage.raw_buffer[arg.offset], U)
    else:
      result.storage.raw_buffer[0] = scanValue(arg.storage.raw_buffer[arg.offset])
    return

  var outer = 1'u
  for dim in 0 ..< scanAxis:
    outer *= uint(arg.shape[dim])
  var inner = 1'u
  for dim in scanAxis + 1 ..< rank:
    inner *= uint(arg.shape[dim])
  let axisLength = uint(arg.shape[scanAxis])
  template scanCoordinate(step: uint): uint =
    if reverse: axisLength - 1'u - step else: step
  let groupCount = outer * inner

  let inputData = arg.storage.raw_buffer
  let output = result.storage.raw_buffer
  template scanOutput(value: untyped): untyped =
    when promoteIntegers and U is SomeInteger: cast[U](value)
    else: value
  template writeOutput(index: uint; value: untyped) =
    when U is KnownSupportsCopyMem:
      output[index] = scanOutput(value)
    else:
      output[int(index)] = scanOutput(value)

  when U is SomeFloat:
    if arg.is_C_contiguous:
      when defined(macosx) and defined(arm64) and U is float32:
        if reverse:
          scanPackedFloat32(addr output[0], unsafeAddr inputData[arg.offset],
                            outer, axisLength, inner, multiply, true)
        else:
          scanPackedFloat32(addr output[0], unsafeAddr inputData[arg.offset],
                            outer, axisLength, inner, multiply, false)
      elif defined(macosx) and defined(arm64) and U is float64:
        if reverse:
          scanPackedFloat64(addr output[0], unsafeAddr inputData[arg.offset],
                            outer, axisLength, inner, multiply, true)
        else:
          scanPackedFloat64(addr output[0], unsafeAddr inputData[arg.offset],
                            outer, axisLength, inner, multiply, false)
      else:
        # Valid packed tensors bound the unsigned storage offsets.
        let inputOffset = uint(arg.offset)
        for group in 0'u ..< groupCount:
          let outputBase = (group div inner) * axisLength * inner + group mod inner
          var accumulated = identity
          for step in 0'u ..< axisLength:
            let coordinate = scanCoordinate(step)
            let flat = outputBase + coordinate * inner
            accumulate(accumulated, float64(scanValue(inputData[inputOffset + flat])))
            output[flat] = U(accumulated)
      return
  elif U is Complex:
    if arg.is_C_contiguous:
      if reverse:
        scanPackedComplex(addr output[0], unsafeAddr inputData[arg.offset],
                          outer, axisLength, inner, multiply, true)
      else:
        scanPackedComplex(addr output[0], unsafeAddr inputData[arg.offset],
                          outer, axisLength, inner, multiply, false)
      return
  elif T is SomeInteger or T is bool or U is SomeInteger:
    if arg.is_C_contiguous:
      let inputOffset = uint(arg.offset)
      if inner == 1:
        for group in 0'u ..< outer:
          let base = group * axisLength
          let first = base + scanCoordinate(0'u)
          var accumulated = scanValue(inputData[inputOffset + first])
          writeOutput(first, accumulated)
          var step = 1'u
          while step < axisLength:
            when multiply and promoteIntegers and U is SomeInteger:
              if cast[U](accumulated) == U(0):
                # Integer zero is absorbing even when a wider carry had other bits.
                let start = if reverse: base else: base + step
                zeroMem(addr output[start], int(axisLength - step) * sizeof(U))
                break
            let flat = base + scanCoordinate(step)
            accumulate(accumulated, scanValue(inputData[inputOffset + flat]))
            writeOutput(flat, accumulated)
            inc step
      elif promoteIntegers and U is SomeInteger:
        # Stored integer prefixes are exact modulo dtype, so rows can stream.
        let integerOutput {.restrict.} = cast[ptr UncheckedArray[U]](result.get_offset_ptr)
        for group in 0'u ..< outer:
          let base = group * axisLength * inner
          var previous = base + scanCoordinate(0'u) * inner
          for column in 0'u ..< inner:
            integerOutput[previous + column] = cast[U](
              scanValue(inputData[inputOffset + previous + column]))
          for step in 1'u ..< axisLength:
            let current = base + scanCoordinate(step) * inner
            for column in 0'u ..< inner:
              var accumulated = integerOutput[previous + column]
              combineIntegerOutput(accumulated, inputData[inputOffset + current + column],
                                   U, multiply)
              integerOutput[current + column] = accumulated
            previous = current
      else:
        for group in 0'u ..< outer:
          let base = group * axisLength * inner
          for column in 0'u ..< inner:
            let first = base + scanCoordinate(0'u) * inner + column
            var accumulated = scanValue(inputData[inputOffset + first])
            writeOutput(first, accumulated)
            for step in 1'u ..< axisLength:
              let coordinate = scanCoordinate(step)
              let flat = base + coordinate * inner + column
              accumulate(accumulated, scanValue(inputData[inputOffset + flat]))
              writeOutput(flat, accumulated)
      return

  var coordinates: Metadata
  coordinates.len = rank
  for dim in 0 ..< rank:
    coordinates[dim] = 0
  var inputBase = arg.offset
  var group = 0'u
  while group < groupCount:
    let outputBase = (group div inner) * axisLength * inner + group mod inner
    when U is SomeFloat:
      var accumulated = identity
      for step in 0'u ..< axisLength:
        let coordinate = scanCoordinate(step)
        accumulate(accumulated,
                   float64(scanValue(inputData[inputBase + int(coordinate) * arg.strides[scanAxis]])))
        writeOutput(outputBase + coordinate * inner, U(accumulated))
    elif U is Complex:
      var accumulated: Complex[float64]
      when multiply: accumulated.re = 1.0
      for step in 0'u ..< axisLength:
        let coordinate = scanCoordinate(step)
        let value = convertedReductionValue(
          inputData[inputBase + int(coordinate) * arg.strides[scanAxis]], U)
        accumulateComplexScan(accumulated, value, multiply)
        type Component = typeof(default(U).re)
        writeOutput(outputBase + coordinate * inner,
                    U(re: Component(accumulated.re), im: Component(accumulated.im)))
    else:
      when promoteIntegers and U is SomeInteger:
        var accumulated: int64
      else:
        var accumulated: U
      var hasValue = false
      for step in 0'u ..< axisLength:
        let coordinate = scanCoordinate(step)
        let value = scanValue(inputData[inputBase + int(coordinate) * arg.strides[scanAxis]])
        if hasValue:
          accumulate(accumulated, value)
        else:
          accumulated = value
          hasValue = true
        writeOutput(outputBase + coordinate * inner, accumulated)

    if group + 1'u < groupCount:
      for dim in countdown(rank - 1, 0):
        if dim == scanAxis:
          continue
        if coordinates[dim] < arg.shape[dim] - 1:
          inc coordinates[dim]
          inputBase += arg.strides[dim]
          break
        coordinates[dim] = 0
        inputBase -= (arg.shape[dim] - 1) * arg.strides[dim]
    inc group

proc cumsum*[T](arg: Tensor[T], axis: int = 0;
                reverse: bool = false): Tensor[sumProductElement(T)] =
  ## Return independent, same-shaped inclusive cumulative sums along ``axis``.
  ## Negative axes count from the end; scalars accept axes 0 and -1. Empty
  ## tensors retain their exact shape. Invalid axes raise ``ValueError``.
  ##
  ## Offset, reversed and broadcast views are read through their strides.
  ## ``reverse`` accumulates from the final coordinate toward the first.
  ## Float32 prefixes accumulate in float64 before rounding each output;
  ## complex prefixes use complex128 carries, including identity arithmetic.
  ## Integers and booleans promote to int64 with modulo-2^64 arithmetic;
  ## other element types retain their type and ordinary ``+`` operation.
  scanCPU(arg, axis, false, sumProductElement(T), reverse = reverse)

proc cumprod*[T](arg: Tensor[T], axis: int = 0;
                 reverse: bool = false): Tensor[sumProductElement(T)] =
  ## Return independent, same-shaped inclusive cumulative products along ``axis``.
  ## Negative axes count from the end; scalars accept axes 0 and -1. Empty
  ## tensors retain their exact shape. Invalid axes raise ``ValueError``.
  ##
  ## Offset, reversed and broadcast views are read through their strides.
  ## ``reverse`` accumulates from the final coordinate toward the first.
  ## Float32 prefixes accumulate in float64 and complex prefixes in complex128,
  ## so rounded output overflow or underflow does not replace the wider carry.
  ## Complex scans multiply by the identity even at the first vector element;
  ## rank-zero scans copy instead. Integers and booleans promote to int64
  ## with modulo-2^64 arithmetic; other types retain their type and ``*``.
  scanCPU(arg, axis, true, sumProductElement(T), reverse = reverse)

proc cumsum*[T: SomeNumber or bool, U: SomeFloat](arg: Tensor[T], axis: int = 0;
    dtype: typedesc[U]; reverse: bool = false): Tensor[U] =
  ## Inclusive sums after casting each real input to the floating ``dtype``.
  ## Float64 carries survive rounded stores; unsigned values convert numerically.
  ## Only the independent output is allocated. Views, scalar/negative axes,
  ## empty shapes and ``reverse`` follow the default-dtype cumulative sum.
  scanCPU(arg, axis, false, U, reverse = reverse)

proc cumprod*[T: SomeNumber or bool, U: SomeFloat](arg: Tensor[T], axis: int = 0;
    dtype: typedesc[U]; reverse: bool = false): Tensor[U] =
  ## Inclusive products after casting each real input to the floating ``dtype``.
  ## Float64 carries survive rounded output overflow and underflow.
  ## Unsigned values convert numerically. Only the independent output is allocated.
  ## Views, scalar/negative axes, empty shapes and ``reverse`` follow ``cumprod``.
  scanCPU(arg, axis, true, U, reverse = reverse)

proc scanIntegerCPU[T, U](arg: Tensor[T], axis: int, multiply: static[bool];
                           dtype: typedesc[U]; reverse: bool): Tensor[U] =
  ## Unsupported scan kernels still permit scalar copies and empty outputs.
  when U is bool or (U is SomeUnsignedInt and sizeof(U) > 1):
    discard cumulativeAxis(arg.rank, axis)
    warnComplexToRealCast(T, U, arg.size)
    if arg.rank > 0 and arg.size > 0:
      raise newException(ValueError, "CPU cumulative scans do not support this output dtype")
    result = newTensorUninit[U](arg.shape)
    if arg.size > 0:
      result.storage.raw_buffer[0] = convertedReductionValue(arg.storage.raw_buffer[arg.offset], U)
  else:
    result = scanCPU(arg, axis, multiply, U, reverse = reverse)

proc cumsum*[T: SomeNumber or bool, U: SomeInteger or bool](arg: Tensor[T], axis: int = 0;
    dtype: typedesc[U]; reverse: bool = false): Tensor[U] =
  ## Cast on read and retain dtype with modulo-width inclusive sums.
  ## Numeric kernels support uint8 and signed outputs; bool and wider unsigned
  ## outputs support only scalars and empties. Invalid axes raise ValueError.
  ## Floating casts must fit dtype. Only the independent output is allocated.
  scanIntegerCPU(arg, axis, false, U, reverse)

proc cumprod*[T: SomeNumber or bool, U: SomeInteger or bool](arg: Tensor[T], axis: int = 0;
    dtype: typedesc[U]; reverse: bool = false): Tensor[U] =
  ## Cast on read and retain dtype with modulo-width inclusive products.
  ## Bool and wider unsigned outputs support only scalars and empty tensors;
  ## uint8 and signed outputs support views, negative axes and reverse scans.
  ## Floating casts must fit dtype. Only the independent output is allocated.
  scanIntegerCPU(arg, axis, true, U, reverse)

proc cumsum*[T: SomeNumber or bool or Complex; F: SomeFloat](arg: Tensor[T],
    axis: int = 0; dtype: typedesc[Complex[F]]; reverse: bool = false): Tensor[Complex[F]] =
  ## Cast both components to dtype before ordered complex128 accumulation.
  ## Real inputs gain a positive-zero imaginary component. Only the independent
  ## output is allocated; views, scalar/negative axes and reverse are supported.
  scanCPU(arg, axis, false, Complex[F], reverse = reverse)

proc cumprod*[T: SomeNumber or bool or Complex; F: SomeFloat](arg: Tensor[T],
    axis: int = 0; dtype: typedesc[Complex[F]]; reverse: bool = false): Tensor[Complex[F]] =
  ## Cast to the complex dtype before multiplying in an ordered complex128 carry.
  ## Rounded prefix overflow/underflow never replaces the carry. Scalar scans
  ## cast and copy; vector scans multiply by the identity at the first element.
  scanCPU(arg, axis, true, Complex[F], reverse = reverse)

proc cumsum*[F: SomeFloat; U: SomeFloat](arg: Tensor[Complex[F]],
    axis: int = 0; dtype: typedesc[U]; reverse: bool = false): Tensor[U] =
  ## Discard imaginary components and cast before ordered float64 accumulation.
  ## Nonempty complex-to-real casts warn once per process; empties do not warn.
  scanCPU(arg, axis, false, U, reverse = reverse)

proc cumprod*[F: SomeFloat; U: SomeFloat](arg: Tensor[Complex[F]],
    axis: int = 0; dtype: typedesc[U]; reverse: bool = false): Tensor[U] =
  ## Discard imaginary components, cast, then multiply with a float64 carry.
  ## Nonempty complex-to-real casts warn once; only the output is allocated.
  scanCPU(arg, axis, true, U, reverse = reverse)

proc cumsum*[F: SomeFloat; U: SomeInteger or bool](arg: Tensor[Complex[F]],
    axis: int = 0; dtype: typedesc[U]; reverse: bool = false): Tensor[U] =
  ## Truncate real components before modulo-width integer accumulation.
  ## Boolean scalar copies test both components. Bool and wider unsigned
  ## outputs accept only scalars and empties. Truncated real values must fit.
  scanIntegerCPU(arg, axis, false, U, reverse)

proc cumprod*[F: SomeFloat; U: SomeInteger or bool](arg: Tensor[Complex[F]],
    axis: int = 0; dtype: typedesc[U]; reverse: bool = false): Tensor[U] =
  ## Multiply truncated real components modulo the integer output width.
  ## Complex-to-numeric casts warn once, except bool keeps both components.
  ## Bool and wider unsigned outputs accept only scalars and empty tensors.
  scanIntegerCPU(arg, axis, true, U, reverse)

proc diff_discrete*[T](arg: Tensor[T], n=1, axis: int = -1): Tensor[T] =
  ## Calculate the n-th discrete difference along the given axis.
  ##
  ## The first difference is given by `out[i] = a[i+1] - a[i]` along the given axis.
  ## Higher differences are calculated by using diff recursively.
  ##
  ## Input:
  ##   - A tensor
  ##   - n: The number of times values are differenced.
  ##        If zero, the input is returned as-is.
  ##   - axis: The axis along which the difference is taken,
  ##           default is the last axis.
  ## Returns:
  ##   - A tensor with the n-th discrete difference along the given axis.
  ##     It's size along that axis will be reduced by one.
  ##   - The code in this function is heavily based upon and equivalent
  ##   to numpy's `diff()` function.
  mixin `_`
  assert n >= 0, "diff order (" & $n & ") cannot be negative"
  if n == 0 or arg.size == 0:
    return arg
  let axis = if axis == -1:
    arg.shape.len + axis
  else:
    axis
  assert axis < arg.shape.len,
    "diff axis (" & $axis & ") cannot be greater than input shape length (" & $arg.shape.len & ")"
  var result_shape = arg.shape
  result_shape[axis] -= 1
  result = zeros[T](result_shape)
  for i, tAxis in enumerateAxis(arg, axis):
    if unlikely(i == 0):
      continue
    var temp = result.atAxisIndex(axis, i-1)
    when T is bool:
      temp[_] = tAxis != arg.atAxisIndex(axis, i-1)
    else:
      temp[_] = tAxis -. arg.atAxisIndex(axis, i-1)
  if n > 1:
    result = diff_discrete(result, n=n-1, axis=axis)

proc unwrap_period*[T: SomeNumber](t: Tensor[T], discont: T = -1, axis = -1, period: T = default(T)): Tensor[T] {.noinit.} =
    # Unwrap a tensor by taking the complement of large deltas with respect to a period.
    #
    # This unwraps a tensor `t` by changing elements which have an absolute
    # difference from their predecessor of more than ``max(discont, period/2)``
    # to their `period`-complementary values.
    #
    # For the default case where `period` is `2*PI` and `discont` is
    # `PI`, this unwraps a radian phase `t` such that adjacent differences
    # are never greater than `PI` by adding `2*k*PIi` for some integer `k`.
    #
    # Inputs:
    #   - t: Input Tensor.
    #   - discont: Maximum discontinuity between values. Default is `period/2`.
    #       Values below `period/2` are treated as if they were `period/2`.
    #       To have an effect different than the default, `discont` must be
    #       larger than `period/2`.
    #   - axis: Axis along which the unwrap will be done. Default is the last axis.
    #   - period: Size of the range over which the input wraps.
    #             By default, it is ``2*PI``.
    # Return:
    #   - Output Tensor.
    #
    # Notes:
    #   - If the discontinuity in `t` is smaller than ``period/2``,
    #   but larger than `discont`, no unwrapping is done because taking
    #   the complement would only make the discontinuity larger.
    #   - The code in this function is heavily based upon and equivalent
    #   to numpy's `unwrap()` function.
  mixin `_`
  let axis = if axis == -1:
    t.shape.len + axis
  else:
    axis
  let td = t.diff_discrete(axis=axis)
  let period: T = if period == default(T):
    when T is int:
      raise newException(ValueError, "unwrap period must be specified for integer types")
    else:
      2.0 * PI
  else:
    period
  let discont = if discont == -1:
    T(period/2)
  else:
    discont
  when T is int:
    when (NimMajor, NimMinor, NimPatch) >= (2, 0, 0):
      let (interval_high, rem) = divmod(period, 2)
    else:
      let interval_high = period div 2
      let rem = period mod 2
    let boundary_ambiguous = rem == 0
  else:
    let interval_high = period / 2
    let boundary_ambiguous = true
  let interval_low = -interval_high
  var tdmod = (td -. interval_low).floorMod(period) +. interval_low
  if boundary_ambiguous:
    const zero: T = T(0)
    tdmod[(tdmod ==. interval_low) and (td >. zero)] = interval_high
  var ph_correct = tdmod - td
  ph_correct[abs(td) <. discont] = 0
  result = t.clone()
  # Phase correction retains unwrap's input dtype without a conversion buffer.
  let ph_correct_cumsum = scanCPU(ph_correct, axis, false, T, false)
  if t.rank == 1:
    result[1.._] = t[1.._] +. ph_correct_cumsum
  else:
    for i, tAxis in enumerateAxis(t, axis):
      if unlikely(i < 1):
        continue
      let pAxis = ph_correct_cumsum.atAxisIndex(axis, i-1)
      var temp = result.atAxisIndex(axis, i)
      temp[_] = tAxis +. pAxis

when (NimMajor, NimMinor, NimPatch) > (1, 6, 0):
  import std / atomics
proc nonzero*[T](arg: Tensor[T]): Tensor[int] =
  ## Returns the indices, which are non zero as a `Tensor[int]`.
  ##
  ## The resulting tensor is 2 dimensional and has one element for each
  ## dimension in ``t``. Each of those elements contains the indicies along
  ## the corresponding axis (element 0 == axis 0), which are non zero.
  ##
  ## Input:
  ##   - A tensor
  ##
  ## Returns:
  ##   - A 2D tensor with N elements, where N is the rank of ``t``
  ##
  ## Example:
  ##   .. code:: nim
  ##      let a = [[3, 0, 0],
  ##               [0, 4, 0],
  ##               [5, 6, 0]].toTensor()
  ##      assert a.nonzero == [[0, 1, 2, 2], [0, 1, 0, 1]].toTensor
  ##      #                    ^-- indices.. ^ ..for  axis 0
  ##      #                                  ∟-- indices for axis 1
  ##      # axis 0: [0, 1, 2, 2] refers to:
  ##      # - 0 -> 3 in row 0
  ##      # - 1 -> 4 in row 1
  ##      # - 2 -> 5 in row 2
  ##      # - 2 -> 6 in row 2
  ##      # axis 1: [0, 1, 0, 1] refers to:
  ##      # - 0 -> 3 in col 0
  ##      # - 1 -> 4 in col 1
  ##      # - 0 -> 5 in col 0
  ##      # - 1 -> 6 in col 1
  when (NimMajor, NimMinor, NimPatch) > (1, 6, 0):
    ## Use `Atomic` counter. If compiled with `-d:openmp` otherwise the code breaks!
    var count: Atomic[int]
    count.store(0)
    let mask = map_inline(arg):
      block:
        let cond = x != 0.T
        if cond:
          atomicInc count
        cond

    result = newTensor[int]([arg.shape.len, count.load])
  else:
    let mask = map_inline(arg): # generate the mask
      x != 0.T
    var count = 0 # and count non zero elements (avoid openmp issues)
    for x in mask:
      if x:
        inc count
    result = newTensor[int]([arg.shape.len, count])

  var ax = 0 # current axis
  var k = 0 # counter for indices in one axis
  for idx, x in mask:
    if x:
      ax = 0
      for j in idx:
        result[ax, k] = j
        inc ax
      inc k

func all*[T](t: Tensor[T]): bool =
  ## Returns true if all of the items in the input tensor are true or non-zero
  ##
  ## Input:
  ##   - A tensor
  ##
  ## Returns:
  ##   - True if at least one element is not zero
  result = true
  when T is Complex:
    const zero: T = complex(0.0, 0.0)
  else:
    const zero: T = 0.T
  for it in t:
    if it == zero:
      result = false
      break

func any*[T](t: Tensor[T]): bool =
  ## Returns true if any of the items in the input tensor is true or non-zero
  ##
  ## Input:
  ##   - A tensor
  ##
  ## Returns:
  ##   - True if at least one element is not zero
  result = false
  when T is Complex:
    const zero: T = complex(0.0, 0.0)
  else:
    const zero: T = 0.T
  for it in t:
    if it != zero:
      result = true
      break
