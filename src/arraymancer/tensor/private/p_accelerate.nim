# Copyright 2026 the Arraymancer contributors
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

## Provides packed CPU float64 reductions and private macOS float32 bulk operations.
import std/[math, complex]
import ../../laser/compiler_optim_hints
from ./p_complex import convertedComplexReductionValue, normalizedComplexMean

template realReductionValue(value: untyped): untyped =
  when typeof(value) is Complex: value.re
  else: value

template convertedReductionValue*(value: untyped; U: typedesc): untyped =
  ## Integer narrowing keeps low bits; defined floating casts truncate first.
  when U is Complex:
    convertedComplexReductionValue(value, typeof(default(U).re))
  elif U is bool:
    when typeof(value) is Complex: value.re != 0 or value.im != 0
    elif typeof(value) is bool: value
    else: value != typeof(value)(0)
  elif typeof(value) is bool: U(ord(value))
  elif U is SomeInteger:
    when typeof(realReductionValue(value)) is SomeFloat:
      when U is SomeUnsignedInt and sizeof(U) == 8: U(realReductionValue(value))
      else: cast[U](int64(realReductionValue(value)))
    elif typeof(value) is SomeUnsignedInt: cast[U](uint64(value))
    else: cast[U](int64(value))
  else: U(realReductionValue(value))

template convertedIntegerWord*(value: untyped; Word: typedesc): untyped =
  ## Signed conversion precedes the unsigned carry, so negative floats keep bits.
  when typeof(realReductionValue(value)) is SomeFloat:
    when typeof(realReductionValue(value)) is float32 and sizeof(Word) <= 4:
      cast[Word](int32(realReductionValue(value)))
    else: cast[Word](int64(realReductionValue(value)))
  else: convertedReductionValue(value, Word)

proc reduceConvertedIntegerSumSpan*[T, U](input: ptr UncheckedArray[T], base, count: uint;
                                          dtype: typedesc[U]): U =
  ## Unsigned narrow carries let the compiler vectorize modulo-width sums.
  when T is float64 or sizeof(U) == 8:
    type Word = uint64
  elif T is float32 or sizeof(U) == 4:
    type Word = uint32
  elif sizeof(U) == 2:
    type Word = uint16
  else:
    type Word = uint8
  var word: Word
  for index in 0'u ..< count:
    word += convertedIntegerWord(input[base + index], Word)
  cast[U](word)

proc reducePackedFloatTruth*[T: SomeFloat](input: ptr UncheckedArray[T],
                                          base, count: uint): bool =
  ## Magnitude bits distinguish both zeros from every nonzero, including NaNs.
  when T is float32:
    type Word = uint32
  else:
    type Word = uint64
  const magnitude = high(Word) shr 1
  let head = min(4'u, count)
  for index in 0'u ..< head:
    if (cast[Word](input[base + index]) and magnitude) == 0:
      return false
  var anyZero: Word
  for index in head ..< count:
    anyZero = anyZero or Word(ord((cast[Word](input[base + index]) and magnitude) == 0))
  anyZero == 0

template accumulateComplexScan*(acc, value: untyped; multiply: static[bool]) =
  ## Widen before arithmetic, not after a rounded complex32 multiplication.
  when multiply:
    let rr = acc.re * float64(value.re)
    let ii = acc.im * float64(value.im)
    let ri = acc.re * float64(value.im)
    let ir = acc.im * float64(value.re)
    acc.re = rr - ii
    acc.im = ri + ir
  else:
    acc.re += float64(value.re)
    acc.im += float64(value.im)

proc scanPackedComplex*[T; F: SomeFloat](output: ptr Complex[F]; input: ptr T;
    outer, axisLength, inner: uint; multiply, reverse: static[bool]) =
  ## Interleave independent scans while keeping each ordered complex128 carry.
  withCompilerOptimHints()
  let destination {.restrict.} = cast[ptr UncheckedArray[Complex[F]]](output)
  let source = cast[ptr UncheckedArray[T]](input)
  template coordinate(step: uint): uint =
    when reverse: axisLength - 1'u - step
    else: step
  let groups = outer * inner
  var group = 0'u
  while group + 4'u <= groups:
    var bases: array[4, uint]
    var carries: array[4, Complex[float64]]
    for lane in 0 ..< 4:
      let scan = group + uint(lane)
      bases[lane] = (scan div inner) * axisLength * inner + scan mod inner
      when multiply: carries[lane].re = 1.0
    for step in 0'u ..< axisLength:
      let offset = coordinate(step) * inner
      for lane in 0 ..< 4:
        let index = bases[lane] + offset
        let value = convertedComplexReductionValue(source[index], F)
        accumulateComplexScan(carries[lane], value, multiply)
        destination[index] = Complex[F](re: F(carries[lane].re), im: F(carries[lane].im))
    group += 4'u
  for tail in group ..< groups:
    let base = (tail div inner) * axisLength * inner + tail mod inner
    var carry: Complex[float64]
    when multiply: carry.re = 1.0
    for step in 0'u ..< axisLength:
      let index = base + coordinate(step) * inner
      let value = convertedComplexReductionValue(source[index], F)
      accumulateComplexScan(carry, value, multiply)
      destination[index] = Complex[F](re: F(carry.re), im: F(carry.im))

template multiplyProductValue*(target, value: untyped) =
  ## Keep component products rounded before combining them, without scan widening.
  when typeof(target) is Complex:
    let right = value
    let rr = target.re * right.re
    let ii = target.im * right.im
    let ri = target.re * right.im
    let ir = target.im * right.re
    target.re = rr - ii
    target.im = ri + ir
  else:
    target *= value

template complexTruthBits(input, index: untyped): uint64 =
  block:
    let words = cast[ptr UncheckedArray[uint64]](input)
    when sizeof(typeof(input[0])) == 8:
      words[index] and 0x7fffffff7fffffff'u64
    else:
      (words[2'u * index] or words[2'u * index + 1'u]) and 0x7fffffffffffffff'u64

proc reducePackedComplexTruth*[T: Complex](input: ptr UncheckedArray[T];
                                           base, count: uint): bool =
  ## Either signed zero is false; NaNs and infinities remain true components.
  var anyZero = 0'u64
  for index in 0'u ..< count:
    anyZero = anyZero or uint64(ord(complexTruthBits(input, base + index) == 0))
  anyZero == 0

proc reduceComplexTruthColumns*[T: Complex](output: ptr bool; input: ptr T;
    outer, length, inner: uint) =
  ## Stream rows into the final boolean carries so truth tests can vectorize.
  withCompilerOptimHints()
  let source = cast[ptr UncheckedArray[T]](input)
  let destination {.restrict.} = cast[ptr UncheckedArray[bool]](output)
  for group in 0'u ..< outer:
    let base = group * length * inner
    let target = group * inner
    for column in 0'u ..< inner: destination[target + column] = true
    for row in 0'u ..< length:
      let offset = base + row * inner
      for column in 0'u ..< inner:
        let truth = uint8(ord(complexTruthBits(source, offset + column) != 0))
        destination[target + column] =
          (uint8(ord(destination[target + column])) and truth) != 0

when defined(macosx) and defined(arm64):
  proc productPackedComplex32Span(input: ptr Complex[float32]; count: uint): Complex[float32]
  proc productPackedComplexAxis[T, U: Complex](output: ptr U; input: ptr T;
                                              outer, length, inner: uint)

proc productComplexSpan*[T; U: Complex; I: SomeInteger](input: ptr UncheckedArray[T];
    base: I; count: uint; stride: I; dtype: typedesc[U]): U {.inline.} =
  ## Four native vectors fold before the scalar tail, with no conversion buffer.
  when defined(macosx) and defined(arm64) and T is Complex[float32] and U is Complex[float32]:
    if stride == 1:
      return productPackedComplex32Span(cast[ptr Complex[float32]](unsafeAddr input[base]), count)
  const vectorBytes = when defined(arm64): 16 else: 32
  const width = vectorBytes div sizeof(U)
  const lanes = 4 * width
  result.re = 1
  when T is bool:
    for index in 0'u ..< count:
      if not input[base + I(index) * stride]: return default(U)
  else:
    var index = 0'u
    let bulk = count - count mod uint(lanes)
    if bulk != 0:
      var partials {.noinit.}: array[lanes, U]
      for lane in 0'u ..< uint(lanes):
        partials[lane] = convertedReductionValue(input[base + I(lane) * stride], U)
      index = uint(lanes)
      while index < bulk:
        for lane in 0'u ..< uint(lanes):
          multiplyProductValue(partials[lane],
            convertedReductionValue(input[base + I(index + lane) * stride], U))
        index += uint(lanes)
      for lane in 0'u ..< uint(width):
        multiplyProductValue(partials[lane], partials[lane + uint(width)])
        multiplyProductValue(partials[lane + uint(2 * width)], partials[lane + uint(3 * width)])
        multiplyProductValue(partials[lane], partials[lane + uint(2 * width)])
      var folded = partials[0]
      for lane in 1'u ..< uint(width):
        multiplyProductValue(folded, partials[lane])
      multiplyProductValue(result, folded)
    while index < count:
      multiplyProductValue(result,
        convertedReductionValue(input[base + I(index) * stride], U))
      inc index

proc productComplexAxis*[T; U: Complex](output: ptr U; input: ptr T;
    outer, length, inner: uint) =
  ## Nonempty packed columns keep independent chains and a right-hand identity.
  when defined(macosx) and defined(arm64) and
      ((T is Complex[float32] and U is Complex[float32]) or T is Complex[float64]):
    productPackedComplexAxis(output, input, outer, length, inner)
  else:
    withCompilerOptimHints()
    let source = cast[ptr UncheckedArray[T]](input)
    let destination {.restrict.} = cast[ptr UncheckedArray[U]](output)
    const vectorBytes = when defined(arm64): 16 else: 32
    const lanes = 4 * vectorBytes div sizeof(U)
    var identity: U
    identity.re = 1
    for group in 0'u ..< outer:
      let base = group * length * inner
      let target = group * inner
      if inner == 1:
        destination[target] = productComplexSpan(source, base, length, 1'u, U)
      else:
        when T is bool:
          for column in 0'u ..< inner:
            var value = identity
            for row in 0'u ..< length:
              if not source[base + row * inner + column]:
                value = default(U)
                break
            destination[target + column] = value
        else:
          var column = 0'u
          while column + uint(lanes) <= inner:
            var partials {.noinit.}: array[lanes, U]
            for lane in 0'u ..< uint(lanes):
              partials[lane] = convertedReductionValue(source[base + column + lane], U)
            for row in 1'u ..< length:
              let offset = base + row * inner + column
              for lane in 0'u ..< uint(lanes):
                multiplyProductValue(partials[lane], convertedReductionValue(source[offset + lane], U))
            for lane in 0'u ..< uint(lanes):
              multiplyProductValue(partials[lane], identity)
              destination[target + column + lane] = partials[lane]
            column += uint(lanes)
          while column < inner:
            var value = identity
            for row in 0'u ..< length:
              multiplyProductValue(value,
                convertedReductionValue(source[base + row * inner + column], U))
            destination[target + column] = value
            inc column

template finishCascadedReduction*(value: untyped; count: int;
                                  average: static[bool]): untyped =
  ## Means retain complex reciprocal rounding; sums keep the zero identity.
  when average:
    when typeof(value) is Complex: normalizedComplexMean(value, count)
    else: value / typeof(value)(count)
  else:
    value

template addCascadedValue(target, value: untyped) =
  when typeof(target) is Complex:
    let part = value
    target.re += part.re
    target.im += part.im
  else:
    target += value

template cascadeReductionRows[T, U; I: SomeInteger; N: static[int]](
    input: ptr UncheckedArray[T]; base, stride, laneStride: I; count: uint;
    sums: var array[N, U]) =
  ## Caller expansion exposes unit strides without relying on C inlining.
  block cascadeRows:
    let data = input
    let first = base
    let rowStride = stride
    let columnStride = laneStride
    let rows = count
    sums = default(typeof(sums))
    # One cascade block has no higher carry to merge.
    if rows <= 16'u:
      for position in 0'u ..< rows:
        let offset = I(position) * rowStride
        for lane in 0'u ..< uint(N):
          addCascadedValue(sums[lane],
            convertedReductionValue(data[first + offset + I(lane) * columnStride], U))
      break cascadeRows
    var levels: array[3, array[N, U]]
    var bits = 0
    var remaining = rows - 1'u
    while remaining != 0:
      inc bits
      remaining = remaining shr 1
    let power = max(4, bits div 4)
    let step = 1'u shl power
    let mask = step - 1'u
    let masks = [mask shl power, mask shl (2 * power), mask shl (3 * power)]
    var position = 0'u
    while position + step <= rows:
      for _ in 0'u ..< step:
        let offset = I(position) * rowStride
        for lane in 0'u ..< uint(N):
          addCascadedValue(sums[lane],
            convertedReductionValue(data[first + offset + I(lane) * columnStride], U))
        inc position
      for level in 0'u ..< 3'u:
        for lane in 0'u ..< uint(N):
          if level == 0:
            addCascadedValue(levels[0][lane], sums[lane])
            sums[lane] = default(U)
          else:
            addCascadedValue(levels[level][lane], levels[level - 1'u][lane])
            levels[level - 1'u][lane] = default(U)
        if (position and masks[level]) != 0: break
    while position < rows:
      let offset = I(position) * rowStride
      for lane in 0'u ..< uint(N):
        addCascadedValue(sums[lane],
          convertedReductionValue(data[first + offset + I(lane) * columnStride], U))
      inc position
    for level in 0'u ..< 3'u:
      for lane in 0'u ..< uint(N):
        addCascadedValue(sums[lane], levels[level][lane])

proc sumCascadedSpan*[T; U: SomeFloat or Complex; I: SomeInteger](input: ptr UncheckedArray[T];
    base: I; count: uint; stride: I; dtype: typedesc[U];
    vectorized: static[bool] = true): U {.inline.} =
  ## CPU row reductions fold scalar tails before their independent partials.
  const lanes = when vectorized: 128 div sizeof(U) else: 4
  const tailLanes = when vectorized: 32 div sizeof(U) else: 1
  let bulk = count - count mod uint(lanes)
  var partials {.noinit.}: array[lanes, U]
  if bulk != 0:
    cascadeReductionRows(input, base, I(lanes) * stride, stride,
      count div uint(lanes), partials)
  else:
    partials = default(typeof(partials))
  let tail = count - count mod uint(tailLanes)
  for index in bulk ..< tail:
    let lane = (index - bulk) mod uint(tailLanes)
    addCascadedValue(partials[lane],
      convertedReductionValue(input[base + I(index) * stride], U))
  for lane in countup(uint(tailLanes), uint(lanes - 1), tailLanes):
    for component in 0'u ..< uint(tailLanes):
      addCascadedValue(partials[component], partials[lane + component])
  for index in tail ..< count:
    addCascadedValue(result, convertedReductionValue(input[base + I(index) * stride], U))
  for lane in 0'u ..< uint(tailLanes):
    addCascadedValue(result, partials[lane])

proc reduceCascadedAxis*[T; U: SomeFloat or Complex](output: ptr U; input: ptr T;
    outer, length, inner: uint; average: static[bool]) =
  ## A bounded column tile needs no conversion tensor or output-sized scratch.
  withCompilerOptimHints()
  let source = cast[ptr UncheckedArray[T]](input)
  let destination {.restrict.} = cast[ptr UncheckedArray[U]](output)
  const lanes = 128 div sizeof(U)
  for group in 0'u ..< outer:
    let base = group * length * inner
    let target = group * inner
    if inner == 1:
      destination[target] = finishCascadedReduction(
        sumCascadedSpan(source, base, length, 1'u, U), int(length), average)
    else:
      var column = 0'u
      while column + uint(lanes) <= inner:
        var partials {.noinit.}: array[lanes, U]
        cascadeReductionRows(source, base + column, inner, 1'u, length, partials)
        for lane in 0'u ..< uint(lanes):
          destination[target + column + uint(lane)] =
            finishCascadedReduction(partials[lane], int(length), average)
        column += uint(lanes)
      while column < inner:
        destination[target + column] = finishCascadedReduction(
          sumCascadedSpan(source, base + column, length, inner, U, false),
          int(length), average)
        inc column

when defined(macosx):
  {.passL: "-framework Accelerate".}

  proc vDSP_sve*(input: ptr float32; stride: clong; output: ptr float32; count: culong)
    {.importc, header: "<Accelerate/Accelerate.h>".}

  proc vvexpf*(output, input: ptr float32; count: ptr cint)
    {.importc, header: "<Accelerate/Accelerate.h>".}
  proc vvlogf*(output, input: ptr float32; count: ptr cint)
    {.importc, header: "<Accelerate/Accelerate.h>".}

  when defined(arm64):
    when defined(clang):
      # Outlining splits these hot loops into calls for each converted SIMD load.
      {.localPassC: "-mno-outline".}
    type Float32x4 {.importc: "float32x4_t", bycopy, header: "<arm_neon.h>".} = object
      raw: array[4, float32]
    type Float32x4x2 {.importc: "float32x4x2_t", bycopy, header: "<arm_neon.h>".} = object
      val: array[2, Float32x4]
    proc vld2q_f32(input: ptr float32): Float32x4x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vst2q_f32(output: ptr float32; value: Float32x4x2)
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vsubq_f32(a, b: Float32x4): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vld1q_f32(input: ptr float32): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vsqrtq_f32(input: Float32x4): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vst1q_f32(output: ptr float32; value: Float32x4)
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vdupq_n_f32(value: float32): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vaddq_f32(a, b: Float32x4): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vmulq_f32(a, b: Float32x4): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vdivq_f32(a, b: Float32x4): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vgetq_lane_f32(value: Float32x4; lane: cint): float32
      {.importc, nodecl, header: "<arm_neon.h>".}
    type Float32x2 {.importc: "float32x2_t", bycopy, header: "<arm_neon.h>".} = object
      raw: array[2, float32]
    type Float64x2 {.importc: "float64x2_t", bycopy, header: "<arm_neon.h>".} = object
      raw: array[2, float64]
    type Float64x2x2 {.importc: "float64x2x2_t", bycopy, header: "<arm_neon.h>".} = object
      val: array[2, Float64x2]
    proc vld2q_f64(input: ptr float64): Float64x2x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vst2q_f64(output: ptr float64; value: Float64x2x2)
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vsubq_f64(a, b: Float64x2): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vld1_f32(input: ptr float32): Float32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vdup_n_f32(value: float32): Float32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vset_lane_f32(value: float32; vector: Float32x2; lane: cint): Float32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vget_lane_f32(value: Float32x2; lane: cint): float32
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vget_low_f32(value: Float32x4): Float32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vget_high_f32(value: Float32x4): Float32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcombine_f32(low, high: Float32x2): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcvt_f64_f32(value: Float32x2): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcvt_f32_f64(value: Float64x2): Float32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vdupq_n_f64(value: float64): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vaddq_f64(a, b: Float64x2): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vld1q_f64(input: ptr float64): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vst1q_f64(output: ptr float64; value: Float64x2)
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vsetq_lane_f64(value: float64; vector: Float64x2; lane: cint): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vgetq_lane_f64(value: Float64x2; lane: cint): float64
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vdivq_f64(a, b: Float64x2): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vmulq_f64(a, b: Float64x2): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    type UInt32x4 {.importc: "uint32x4_t", bycopy, header: "<arm_neon.h>".} = object
      raw: array[4, uint32]
    proc vceqq_f32(a, b: Float32x4): UInt32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vmaxvq_u32(value: UInt32x4): uint32
      {.importc, nodecl, header: "<arm_neon.h>".}

    type
      Int8x8 {.importc: "int8x8_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[8, int8]
      Int16x4 {.importc: "int16x4_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[4, int16]
      Int16x8 {.importc: "int16x8_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[8, int16]
      Int32x4 {.importc: "int32x4_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[4, int32]
    proc vld1_s8(input: ptr int8): Int8x8
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vdupq_n_s16(value: int16): Int16x8
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vaddq_s16(a, b: Int16x8): Int16x8
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vmovl_s8(value: Int8x8): Int16x8
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vget_low_s16(value: Int16x8): Int16x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vget_high_s16(value: Int16x8): Int16x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vmovl_s16(value: Int16x4): Int32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vld1q_s32(input: ptr int32): Int32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vld1q_u32(input: ptr uint32): UInt32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcvtq_f32_s32(value: Int32x4): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcvtq_f32_u32(value: UInt32x4): Float32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vaddvq_f32(value: Float32x4): float32
      {.importc, nodecl, header: "<arm_neon.h>".}

    type
      UInt8x8 {.importc: "uint8x8_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[8, uint8]
      UInt16x4 {.importc: "uint16x4_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[4, uint16]
      UInt16x8 {.importc: "uint16x8_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[8, uint16]
      Int32x2 {.importc: "int32x2_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[2, int32]
      UInt32x2 {.importc: "uint32x2_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[2, uint32]
      Int64x2 {.importc: "int64x2_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[2, int64]
      UInt64x2 {.importc: "uint64x2_t", bycopy, header: "<arm_neon.h>".} = object
        raw: array[2, uint64]
    proc vcreate_s8(value: uint64): Int8x8
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcreate_u8(value: uint64): UInt8x8
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcreate_s16(value: uint64): Int16x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcreate_u16(value: uint64): UInt16x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vmovl_u8(value: UInt8x8): UInt16x8
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vget_low_u16(value: UInt16x8): UInt16x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vmovl_u16(value: UInt16x4): UInt32x4
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vget_low_s32(value: Int32x4): Int32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vget_low_u32(value: UInt32x4): UInt32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vld1_s32(input: ptr int32): Int32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vld1_u32(input: ptr uint32): UInt32x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vmovl_s32(value: Int32x2): Int64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vmovl_u32(value: UInt32x2): UInt64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vld1q_s64(input: ptr int64): Int64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vld1q_u64(input: ptr uint64): UInt64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcvtq_f64_s64(value: Int64x2): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vcvtq_f64_u64(value: UInt64x2): Float64x2
      {.importc, nodecl, header: "<arm_neon.h>".}

    proc vld1_u8(input: ptr uint8): UInt8x8
      {.importc, nodecl, header: "<arm_neon.h>".}
    proc vget_high_u16(value: UInt16x8): UInt16x4
      {.importc, nodecl, header: "<arm_neon.h>".}

    template multiplyComplexProductVectors(target, value: untyped) =
      let right = value
      when typeof(target) is Float32x4x2:
        let rr = vmulq_f32(target.val[0], right.val[0])
        let ii = vmulq_f32(target.val[1], right.val[1])
        let ri = vmulq_f32(target.val[0], right.val[1])
        let ir = vmulq_f32(target.val[1], right.val[0])
        target.val[0] = vsubq_f32(rr, ii)
        target.val[1] = vaddq_f32(ri, ir)
      else:
        let rr = vmulq_f64(target.val[0], right.val[0])
        let ii = vmulq_f64(target.val[1], right.val[1])
        let ri = vmulq_f64(target.val[0], right.val[1])
        let ir = vmulq_f64(target.val[1], right.val[0])
        target.val[0] = vsubq_f64(rr, ii)
        target.val[1] = vaddq_f64(ri, ir)

    proc productPackedComplex32Span(input: ptr Complex[float32];
                                    count: uint): Complex[float32] =
      ## Deinterleaved vectors keep all eight logical partials independent.
      let source = cast[ptr UncheckedArray[Complex[float32]]](input)
      result.re = 1
      var index = 0'u
      let bulk = count and not 7'u
      if bulk != 0:
        var a = vld2q_f32(cast[ptr float32](addr source[0]))
        var b = vld2q_f32(cast[ptr float32](addr source[4]))
        index = 8
        while index < bulk:
          multiplyComplexProductVectors(a, vld2q_f32(cast[ptr float32](addr source[index])))
          multiplyComplexProductVectors(b, vld2q_f32(cast[ptr float32](addr source[index + 4])))
          index += 8
        var partials {.noinit.}: array[8, Complex[float32]]
        vst2q_f32(cast[ptr float32](addr partials[0]), a)
        vst2q_f32(cast[ptr float32](addr partials[4]), b)
        for lane in 0 ..< 2:
          multiplyProductValue(partials[lane], partials[lane + 2])
          multiplyProductValue(partials[lane + 4], partials[lane + 6])
          multiplyProductValue(partials[lane], partials[lane + 4])
        var folded = partials[0]
        multiplyProductValue(folded, partials[1])
        multiplyProductValue(result, folded)
      while index < count:
        multiplyProductValue(result, source[index])
        inc index

    proc loadComplexProductVectors[T, U: Complex](source: ptr UncheckedArray[T];
        index: uint; dtype: typedesc[U]): auto {.inline.} =
      when U is Complex[float32]:
        when T is Complex[float32]:
          vld2q_f32(cast[ptr float32](addr source[index]))
        else:
          let low = vld2q_f64(cast[ptr float64](addr source[index]))
          let high = vld2q_f64(cast[ptr float64](addr source[index + 2]))
          var converted: Float32x4x2
          converted.val[0] = vcombine_f32(vcvt_f32_f64(low.val[0]), vcvt_f32_f64(high.val[0]))
          converted.val[1] = vcombine_f32(vcvt_f32_f64(low.val[1]), vcvt_f32_f64(high.val[1]))
          converted
      else:
        vld2q_f64(cast[ptr float64](addr source[index]))

    proc productPackedComplexAxis[T, U: Complex](output: ptr U; input: ptr T;
                                                outer, length, inner: uint) =
      ## Deinterleaved columns share only loads, never arithmetic or output storage.
      withCompilerOptimHints()
      let source = cast[ptr UncheckedArray[T]](input)
      let destination {.restrict.} = cast[ptr UncheckedArray[U]](output)
      when U is Complex[float32]:
        const width = 4'u
        var identity: Float32x4x2
        identity.val[0] = vdupq_n_f32(1)
        identity.val[1] = vdupq_n_f32(0)
        template store(index: uint; value: untyped) =
          vst2q_f32(cast[ptr float32](addr destination[index]), value)
      else:
        const width = 2'u
        var identity: Float64x2x2
        identity.val[0] = vdupq_n_f64(1)
        identity.val[1] = vdupq_n_f64(0)
        template store(index: uint; value: untyped) =
          vst2q_f64(cast[ptr float64](addr destination[index]), value)
      for group in 0'u ..< outer:
        let base = group * length * inner
        let target = group * inner
        if inner == 1:
          destination[target] = productComplexSpan(source, base, length, 1'u, U)
        else:
          var column = 0'u
          while column + 2'u * width <= inner:
            var a = loadComplexProductVectors(source, base + column, U)
            var b = loadComplexProductVectors(source, base + column + width, U)
            for row in 1'u ..< length:
              let offset = base + row * inner + column
              multiplyComplexProductVectors(a, loadComplexProductVectors(source, offset, U))
              multiplyComplexProductVectors(b, loadComplexProductVectors(source, offset + width, U))
            multiplyComplexProductVectors(a, identity)
            multiplyComplexProductVectors(b, identity)
            store(target + column, a)
            store(target + column + width, b)
            column += 2'u * width
          while column < inner:
            var value: U
            value.re = 1
            for row in 0'u ..< length:
              multiplyProductValue(value, convertedReductionValue(source[base + row * inner + column], U))
            destination[target + column] = value
            inc column

    proc loadSmallIntegerLanes[T](input: ptr UncheckedArray[T], index: uint;
                                    count: static[int]): auto {.inline.} =
      ## Exact-width copies become unaligned loads without aliasing or overreads.
      var bits: uint64
      copyMem(addr bits, unsafeAddr input[index], count * sizeof(T))
      when T is int8:
        vmovl_s16(vget_low_s16(vmovl_s8(vcreate_s8(bits))))
      elif T is int16:
        vmovl_s16(vcreate_s16(bits))
      elif T is uint16:
        vmovl_u16(vcreate_u16(bits))
      else:
        vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(bits))))

    # Both cast helpers fill every lane; runtime result-zeroing would dominate loads.
    proc loadConvertedFloat32x4[T](input: ptr UncheckedArray[T], index: uint): Float32x4 {.inline, noinit.} =
      ## Unaligned loads cast each lane before any output-dtype arithmetic.
      when T is float32:
        vld1q_f32(unsafeAddr input[index])
      elif T is int32:
        vcvtq_f32_s32(vld1q_s32(unsafeAddr input[index]))
      elif T is uint32:
        vcvtq_f32_u32(vld1q_u32(unsafeAddr input[index]))
      elif T is int8 or T is int16:
        vcvtq_f32_s32(loadSmallIntegerLanes(input, index, 4))
      elif T is bool or T is uint8 or T is uint16:
        vcvtq_f32_u32(loadSmallIntegerLanes(input, index, 4))
      elif T is float64:
        vcombine_f32(vcvt_f32_f64(vld1q_f64(unsafeAddr input[index])),
                     vcvt_f32_f64(vld1q_f64(unsafeAddr input[index + 2])))
      else:
        let low = vset_lane_f32(convertedReductionValue(input[index + 1], float32),
          vdup_n_f32(convertedReductionValue(input[index], float32)), 1)
        let high = vset_lane_f32(convertedReductionValue(input[index + 3], float32),
          vdup_n_f32(convertedReductionValue(input[index + 2], float32)), 1)
        vcombine_f32(low, high)

    proc loadConvertedByteFloat32Pair[T: int8 or uint8](input: ptr UncheckedArray[T],
        index: uint; first, second: var Float32x4) {.inline.} =
      ## Eight live bytes let LLVM widen both halves without artificial lane dependencies.
      when T is int8:
        let wide = vmovl_s8(vld1_s8(unsafeAddr input[index]))
        first = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide)))
        second = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide)))
      else:
        let wide = vmovl_u8(vld1_u8(unsafeAddr input[index]))
        first = vcvtq_f32_u32(vmovl_u16(vget_low_u16(wide)))
        second = vcvtq_f32_u32(vmovl_u16(vget_high_u16(wide)))

    proc loadConvertedFloat64x2[T](input: ptr UncheckedArray[T], index: uint): Float64x2 {.inline, noinit.} =
      ## Integer widening retains numeric casts without a scalar lane pack.
      when T is float64:
        vld1q_f64(unsafeAddr input[index])
      elif T is float32:
        vcvt_f64_f32(vld1_f32(unsafeAddr input[index]))
      elif T is Complex:
        vsetq_lane_f64(convertedReductionValue(input[index + 1], float64),
          vdupq_n_f64(convertedReductionValue(input[index], float64)), 1)
      elif T is int8 or T is int16:
        vcvtq_f64_s64(vmovl_s32(vget_low_s32(loadSmallIntegerLanes(input, index, 2))))
      elif T is bool or T is uint8 or T is uint16:
        vcvtq_f64_u64(vmovl_u32(vget_low_u32(loadSmallIntegerLanes(input, index, 2))))
      elif T is int32:
        vcvtq_f64_s64(vmovl_s32(vld1_s32(unsafeAddr input[index])))
      elif T is uint32:
        vcvtq_f64_u64(vmovl_u32(vld1_u32(unsafeAddr input[index])))
      elif T is SomeSignedInt:
        vcvtq_f64_s64(vld1q_s64(cast[ptr int64](unsafeAddr input[index])))
      else:
        vcvtq_f64_u64(vld1q_u64(cast[ptr uint64](unsafeAddr input[index])))

    proc reduceConvertedProductColumns*[T, U](output: ptr U; input: ptr T;
                                               outer, axisLength, inner: uint) =
      ## Parallelize independent columns, never reassociate a column's products.
      withCompilerOptimHints()
      let destination {.restrict.} = cast[ptr UncheckedArray[U]](output)
      let source = cast[ptr UncheckedArray[T]](input)
      when U is float32:
        const width = 4'u
        template identity: untyped = vdupq_n_f32(1'f32)
        template load(index: uint): untyped = loadConvertedFloat32x4(source, index)
        template multiply(a, b: untyped): untyped = vmulq_f32(a, b)
        template store(index: uint; value: untyped) = vst1q_f32(addr destination[index], value)
      else:
        const width = 2'u
        template identity: untyped = vdupq_n_f64(1.0)
        template load(index: uint): untyped = loadConvertedFloat64x2(source, index)
        template multiply(a, b: untyped): untyped = vmulq_f64(a, b)
        template store(index: uint; value: untyped) = vst1q_f64(addr destination[index], value)
      for group in 0'u ..< outer:
        let inputBase = group * axisLength * inner
        let outputBase = group * inner
        var column = 0'u
        while column + 4 * width <= inner:
          var a = identity()
          var b = a
          var c = a
          var d = a
          for row in 0'u ..< axisLength:
            let base = inputBase + row * inner + column
            when U is float32 and (T is int8 or T is uint8):
              var
                first {.noinit.}: Float32x4
                second {.noinit.}: Float32x4
                third {.noinit.}: Float32x4
                fourth {.noinit.}: Float32x4
              loadConvertedByteFloat32Pair(source, base, first, second)
              loadConvertedByteFloat32Pair(source, base + 8, third, fourth)
              a = multiply(a, first)
              b = multiply(b, second)
              c = multiply(c, third)
              d = multiply(d, fourth)
            else:
              a = multiply(a, load(base))
              b = multiply(b, load(base + width))
              c = multiply(c, load(base + 2 * width))
              d = multiply(d, load(base + 3 * width))
          store(outputBase + column, a)
          store(outputBase + column + width, b)
          store(outputBase + column + 2 * width, c)
          store(outputBase + column + 3 * width, d)
          column += 4 * width
        while column + width <= inner:
          var value = identity()
          for row in 0'u ..< axisLength:
            value = multiply(value, load(inputBase + row * inner + column))
          store(outputBase + column, value)
          column += width
        for tail in column ..< inner:
          var value = U(1)
          for row in 0'u ..< axisLength:
            value *= convertedReductionValue(source[inputBase + row * inner + tail], U)
          destination[outputBase + tail] = value

    proc sumConvertedFloat32Block*[T](input: ptr UncheckedArray[T],
                                       base, count: uint): float32 {.inline.} =
      ## Convert before the SIMD sum; callers bound each block to 1024 values.
      var a = vdupq_n_f32(0)
      var b = a
      var c = a
      var d = a
      let bulk = count and not 15'u
      var index = 0'u
      while index < bulk:
        a = vaddq_f32(a, loadConvertedFloat32x4(input, base + index))
        b = vaddq_f32(b, loadConvertedFloat32x4(input, base + index + 4))
        c = vaddq_f32(c, loadConvertedFloat32x4(input, base + index + 8))
        d = vaddq_f32(d, loadConvertedFloat32x4(input, base + index + 12))
        index += 16
      var packed = vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d))
      while index + 4 <= count:
        packed = vaddq_f32(packed, loadConvertedFloat32x4(input, base + index))
        index += 4
      result = vaddvq_f32(packed)
      while index < count:
        result += float32(input[base + index])
        inc index

    proc reduceConvertedFloat32Columns*[T](output: ptr float32,
        input: ptr UncheckedArray[T], outer, axisLength, inner: uint;
        average: static[bool]) =
      ## Byte inputs and their 128-term sums are exact in both int16 and float32.
      withCompilerOptimHints()
      let destination {.restrict.} = cast[ptr UncheckedArray[float32]](output)
      when average:
        let divisor = vdupq_n_f32(float32(axisLength))
      let columns = inner and not 7'u
      for batch in 0'u ..< outer:
        let base = batch * axisLength * inner
        let outputBase = batch * inner
        var column = 0'u
        while column < columns:
          var value = vdupq_n_s16(0)
          for row in 0'u ..< axisLength:
            let source = cast[ptr int8](unsafeAddr input[base + row * inner + column])
            value = vaddq_s16(value, vmovl_s8(vld1_s8(source)))
          var low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(value)))
          var high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(value)))
          when average:
            low = vdivq_f32(low, divisor)
            high = vdivq_f32(high, divisor)
          vst1q_f32(addr destination[outputBase + column], low)
          vst1q_f32(addr destination[outputBase + column + 4], high)
          column += 8
        for tail in column ..< inner:
          var value = 0'f32
          for row in 0'u ..< axisLength:
            when T is bool: value += float32(ord(input[base + row * inner + tail]))
            else: value += float32(input[base + row * inner + tail])
          when average: value /= float32(axisLength)
          destination[outputBase + tail] = value

    proc scanPackedFloat32*[T](output: ptr float32; input: ptr T;
                              outer, axisLength, inner: uint;
                           multiply, reverse: static[bool]) =
      ## Scan disjoint packed spans, preserving sequential float64 accumulation.
      ## SIMD lanes are separate scans, never reassociated prefixes of one scan.
      withCompilerOptimHints()
      let destination {.restrict.} = cast[ptr UncheckedArray[float32]](output)
      let source = cast[ptr UncheckedArray[T]](input)
      template inputValue(index: uint): float32 =
        when T is bool: float32(ord(source[index]))
        elif T is Complex: float32(source[index].re)
        else: float32(source[index])
      const identity = when multiply: 1.0 else: 0.0
      template combine(a, b: Float64x2): Float64x2 =
        when multiply: vmulq_f64(a, b)
        else: vaddq_f64(a, b)
      template coordinate(step: uint): uint =
        when reverse: axisLength - 1 - step
        else: step
      if inner == 1:
        var group = 0'u
        while group + 4 <= outer:
          let base0 = group * axisLength
          let base1 = base0 + axisLength
          let base2 = base1 + axisLength
          let base3 = base2 + axisLength
          var low = vdupq_n_f64(identity)
          var high = low
          for step in 0'u ..< axisLength:
            let index = coordinate(step)
            let a = vset_lane_f32(inputValue(base1 + index),
                                 vdup_n_f32(inputValue(base0 + index)), 1)
            let b = vset_lane_f32(inputValue(base3 + index),
                                 vdup_n_f32(inputValue(base2 + index)), 1)
            low = combine(low, vcvt_f64_f32(a))
            high = combine(high, vcvt_f64_f32(b))
            let packedLow = vcvt_f32_f64(low)
            let packedHigh = vcvt_f32_f64(high)
            destination[base0 + index] = vget_lane_f32(packedLow, 0)
            destination[base1 + index] = vget_lane_f32(packedLow, 1)
            destination[base2 + index] = vget_lane_f32(packedHigh, 0)
            destination[base3 + index] = vget_lane_f32(packedHigh, 1)
          group += 4
        for tail in group ..< outer:
          let base = tail * axisLength
          var accumulated = identity
          for step in 0'u ..< axisLength:
            let index = base + coordinate(step)
            when multiply: accumulated *= float64(inputValue(index))
            else: accumulated += float64(inputValue(index))
            destination[index] = float32(accumulated)
        return
      for group in 0'u ..< outer:
        let base = group * axisLength * inner
        var column = 0'u
        while column + 4 <= inner:
          var low = vdupq_n_f64(identity)
          var high = low
          for step in 0'u ..< axisLength:
            let index = base + coordinate(step) * inner + column
            when T is float32:
              let values = vld1q_f32(unsafeAddr source[index])
              let a = vget_low_f32(values)
              let b = vget_high_f32(values)
            elif T is int32 or T is uint32 or T is float64:
              let values = loadConvertedFloat32x4(source, index)
              let a = vget_low_f32(values)
              let b = vget_high_f32(values)
            else:
              let a = vset_lane_f32(inputValue(index + 1),
                                   vdup_n_f32(inputValue(index)), 1)
              let b = vset_lane_f32(inputValue(index + 3),
                                   vdup_n_f32(inputValue(index + 2)), 1)
            low = combine(low, vcvt_f64_f32(a))
            high = combine(high, vcvt_f64_f32(b))
            vst1q_f32(addr destination[index],
                     vcombine_f32(vcvt_f32_f64(low), vcvt_f32_f64(high)))
          column += 4
        for tail in column ..< inner:
          var accumulated = identity
          for step in 0'u ..< axisLength:
            let index = base + coordinate(step) * inner + tail
            when multiply: accumulated *= float64(inputValue(index))
            else: accumulated += float64(inputValue(index))
            destination[index] = float32(accumulated)

    proc scanPackedFloat64*[T](output: ptr float64; input: ptr T;
                              outer, axisLength, inner: uint;
                           multiply, reverse: static[bool]) =
      ## Vectorize independent scans while retaining each scan's ordered carry.
      withCompilerOptimHints()
      let destination {.restrict.} = cast[ptr UncheckedArray[float64]](output)
      let source = cast[ptr UncheckedArray[T]](input)
      template inputValue(index: uint): float64 =
        when T is bool: float64(ord(source[index]))
        elif T is Complex: float64(source[index].re)
        else: float64(source[index])
      const identity = when multiply: 1.0 else: 0.0
      template combine(a, b: Float64x2): Float64x2 =
        when multiply: vmulq_f64(a, b)
        else: vaddq_f64(a, b)
      template coordinate(step: uint): uint =
        when reverse: axisLength - 1'u - step
        else: step

      if inner == 1:
        var group = 0'u
        when defined(arm64):
          while group + 2 <= outer:
            let base0 = group * axisLength
            let base1 = base0 + axisLength
            var accumulated = vdupq_n_f64(identity)
            for step in 0'u ..< axisLength:
              let index = coordinate(step)
              var values = vdupq_n_f64(inputValue(base0 + index))
              values = vsetq_lane_f64(inputValue(base1 + index), values, 1)
              accumulated = combine(accumulated, values)
              destination[base0 + index] = vgetq_lane_f64(accumulated, 0)
              destination[base1 + index] = vgetq_lane_f64(accumulated, 1)
            group += 2
        for tail in group ..< outer:
          let base = tail * axisLength
          var accumulated = identity
          for step in 0'u ..< axisLength:
            let index = base + coordinate(step)
            when multiply: accumulated *= inputValue(index)
            else: accumulated += inputValue(index)
            destination[index] = accumulated
        return

      for group in 0'u ..< outer:
        let base = group * axisLength * inner
        var column = 0'u
        when defined(arm64):
          while column + 2 <= inner:
            var accumulated = vdupq_n_f64(identity)
            for step in 0'u ..< axisLength:
              let index = base + coordinate(step) * inner + column
              when T is float64:
                let values = vld1q_f64(unsafeAddr source[index])
              elif T is float32:
                let values = vcvt_f64_f32(vld1_f32(unsafeAddr source[index]))
              else:
                var values = vdupq_n_f64(inputValue(index))
                values = vsetq_lane_f64(inputValue(index + 1), values, 1)
              accumulated = combine(accumulated, values)
              vst1q_f64(addr destination[index], accumulated)
            column += 2
        for tail in column ..< inner:
          var accumulated = identity
          for step in 0'u ..< axisLength:
            let index = base + coordinate(step) * inner + tail
            when multiply: accumulated *= inputValue(index)
            else: accumulated += inputValue(index)
            destination[index] = accumulated

    proc cumprodGradientPackedColumnsFloat32*(
        output, input, prefixes, seed: ptr float32;
        outer, axisLength, inner: uint): bool =
      ## Vectorize independent zero-free columns, not terms within a reverse sum.
      ## Return false before writing when a zero needs the generic first-zero masks.
      withCompilerOptimHints()
      let destination {.restrict.} = cast[ptr UncheckedArray[float32]](output)
      let source = cast[ptr UncheckedArray[float32]](input)
      let cached = cast[ptr UncheckedArray[float32]](prefixes)
      let upstream = cast[ptr UncheckedArray[float32]](seed)
      let count = outer * axisLength * inner
      let zero = vdupq_n_f32(0'f32)
      var cursor = 0'u
      while cursor + 4 <= count:
        if vmaxvq_u32(vceqq_f32(vld1q_f32(unsafeAddr source[cursor]), zero)) != 0:
          return false
        cursor += 4
      for tail in cursor ..< count:
        if source[tail] == 0'f32: return false
      for group in 0'u ..< outer:
        let base = group * axisLength * inner
        var column = 0'u
        while column + 4 <= inner:
          var low = vdupq_n_f64(0.0)
          var high = low
          var remaining = axisLength
          while remaining > 0:
            dec remaining
            let index = base + remaining * inner + column
            let weight = vmulq_f32(vld1q_f32(unsafeAddr cached[index]),
                                   vld1q_f32(unsafeAddr upstream[index]))
            low = vaddq_f64(low, vcvt_f64_f32(vget_low_f32(weight)))
            high = vaddq_f64(high, vcvt_f64_f32(vget_high_f32(weight)))
            let rounded = vcombine_f32(vcvt_f32_f64(low), vcvt_f32_f64(high))
            vst1q_f32(addr destination[index],
                      vdivq_f32(rounded, vld1q_f32(unsafeAddr source[index])))
          column += 4
        for tail in column ..< inner:
          var accumulated = 0.0
          var remaining = axisLength
          while remaining > 0:
            dec remaining
            let index = base + remaining * inner + tail
            let weight = cached[index] * upstream[index]
            accumulated += float64(weight)
            destination[index] = float32(accumulated) / source[index]
      return true
    proc productGradientPackedColumnsFloat64*(
        output, input, products, seed: ptr float64;
        outer, axisLength, inner: uint): bool =
      ## Stream independent column derivatives; zeros retain the generic masks.
      withCompilerOptimHints()
      let destination {.restrict.} = cast[ptr UncheckedArray[float64]](output)
      let source = cast[ptr UncheckedArray[float64]](input)
      let cached = cast[ptr UncheckedArray[float64]](products)
      let upstream = cast[ptr UncheckedArray[float64]](seed)
      for index in 0'u ..< outer * axisLength * inner:
        if source[index] == 0.0: return false
      for group in 0'u ..< outer:
        let inputBase = group * axisLength * inner
        let reducedBase = group * inner
        for row in 0'u ..< axisLength:
          let base = inputBase + row * inner
          var column = 0'u
          while column + 2 <= inner:
            let weighted = vmulq_f64(vld1q_f64(unsafeAddr cached[reducedBase + column]),
                                     vld1q_f64(unsafeAddr upstream[reducedBase + column]))
            vst1q_f64(addr destination[base + column],
                      vdivq_f64(weighted, vld1q_f64(unsafeAddr source[base + column])))
            column += 2
          for tail in column ..< inner:
            let weighted = cached[reducedBase + tail] * upstream[reducedBase + tail]
            destination[base + tail] = weighted / source[base + tail]
      true


  proc reduceFloat64Span[T](input: ptr T; count: uint;
                           multiply: static[bool]): float64 =
    ## Independent partial carries shorten the dependency chain for one span.
    let source = cast[ptr UncheckedArray[T]](input)
    const identity = when multiply: 1.0 else: 0.0
    var cursor = 0'u
    when defined(arm64):
      template combine(a, b: Float64x2): Float64x2 =
        when multiply: vmulq_f64(a, b)
        else: vaddq_f64(a, b)
      var a = vdupq_n_f64(identity)
      var b = a
      var c = a
      var d = a
      while cursor + 8 <= count:
        a = combine(a, loadConvertedFloat64x2(source, cursor))
        b = combine(b, loadConvertedFloat64x2(source, cursor + 2))
        c = combine(c, loadConvertedFloat64x2(source, cursor + 4))
        d = combine(d, loadConvertedFloat64x2(source, cursor + 6))
        cursor += 8
      let lanes = combine(combine(a, b), combine(c, d))
      when multiply:
        result = vgetq_lane_f64(lanes, 0) * vgetq_lane_f64(lanes, 1)
      else:
        result = vgetq_lane_f64(lanes, 0) + vgetq_lane_f64(lanes, 1)
    else:
      var a = identity
      var b = identity
      var c = identity
      var d = identity
      while cursor + 4 <= count:
        when multiply:
          a *= convertedReductionValue(source[cursor], float64)
          b *= convertedReductionValue(source[cursor + 1], float64)
          c *= convertedReductionValue(source[cursor + 2], float64)
          d *= convertedReductionValue(source[cursor + 3], float64)
        else:
          a += convertedReductionValue(source[cursor], float64)
          b += convertedReductionValue(source[cursor + 1], float64)
          c += convertedReductionValue(source[cursor + 2], float64)
          d += convertedReductionValue(source[cursor + 3], float64)
        cursor += 4
      when multiply: result = ((a * b) * c) * d
      else: result = ((a + b) + c) + d
    while cursor < count:
      when multiply: result *= convertedReductionValue(source[cursor], float64)
      else: result += convertedReductionValue(source[cursor], float64)
      inc cursor

  proc reducePackedFloat64*(output, input: ptr float64; count: uint;
                           multiply: static[bool]) =
    ## Callers establish a nonempty C-contiguous span and valid storage bounds.
    let destination = cast[ptr UncheckedArray[float64]](output)
    destination[0] = reduceFloat64Span(input, count, multiply)

  proc reducePackedFloat64Axis*(output, input: ptr float64;
                               outer, axisLength, inner: uint;
                               multiply, average: static[bool]) =
    ## SIMD lanes cover independent axis reductions, never terms of one output.
    static: doAssert not (multiply and average)
    withCompilerOptimHints()
    let destination {.restrict.} = cast[ptr UncheckedArray[float64]](output)
    let source = cast[ptr UncheckedArray[float64]](input)
    if inner == 1:
      for group in 0'u ..< outer:
        let value = reduceFloat64Span(unsafeAddr source[group * axisLength],
                                      axisLength, multiply)
        when average: destination[group] = value / float64(axisLength)
        else: destination[group] = value
      return
    when defined(arm64) and multiply:
      reduceConvertedProductColumns(output, input, outer, axisLength, inner)
    else:
      const identity = when multiply: 1.0 else: 0.0
      for group in 0'u ..< outer:
        let inputBase = group * axisLength * inner
        let outputBase = group * inner
        var column = 0'u
        when defined(arm64):
          template combine(a, b: Float64x2): Float64x2 = vaddq_f64(a, b)
          # Adjacent output pairs share each row visit without regrouping terms.
          while column + 4 <= inner:
            var a0 = vdupq_n_f64(identity)
            var b0 = a0
            var c0 = a0
            var d0 = a0
            var a1 = a0
            var b1 = a0
            var c1 = a0
            var d1 = a0
            var row = 0'u
            while row + 4 <= axisLength:
              let base = inputBase + row * inner + column
              a0 = vaddq_f64(a0, vld1q_f64(unsafeAddr source[base]))
              a1 = vaddq_f64(a1, vld1q_f64(unsafeAddr source[base + 2]))
              b0 = vaddq_f64(b0, vld1q_f64(unsafeAddr source[base + inner]))
              b1 = vaddq_f64(b1, vld1q_f64(unsafeAddr source[base + inner + 2]))
              c0 = vaddq_f64(c0, vld1q_f64(unsafeAddr source[base + 2 * inner]))
              c1 = vaddq_f64(c1, vld1q_f64(unsafeAddr source[base + 2 * inner + 2]))
              d0 = vaddq_f64(d0, vld1q_f64(unsafeAddr source[base + 3 * inner]))
              d1 = vaddq_f64(d1, vld1q_f64(unsafeAddr source[base + 3 * inner + 2]))
              row += 4
            var value0 = vaddq_f64(vaddq_f64(a0, b0), vaddq_f64(c0, d0))
            var value1 = vaddq_f64(vaddq_f64(a1, b1), vaddq_f64(c1, d1))
            for tail in row ..< axisLength:
              let base = inputBase + tail * inner + column
              value0 = vaddq_f64(value0, vld1q_f64(unsafeAddr source[base]))
              value1 = vaddq_f64(value1, vld1q_f64(unsafeAddr source[base + 2]))
            when average:
              let divisor = vdupq_n_f64(float64(axisLength))
              value0 = vdivq_f64(value0, divisor)
              value1 = vdivq_f64(value1, divisor)
            vst1q_f64(addr destination[outputBase + column], value0)
            vst1q_f64(addr destination[outputBase + column + 2], value1)
            column += 4
          while column + 2 <= inner:
            var a = vdupq_n_f64(identity)
            var b = a
            var c = a
            var d = a
            var row = 0'u
            while row + 4 <= axisLength:
              a = combine(a, vld1q_f64(unsafeAddr source[inputBase + row * inner + column]))
              b = combine(b, vld1q_f64(unsafeAddr source[inputBase + (row + 1) * inner + column]))
              c = combine(c, vld1q_f64(unsafeAddr source[inputBase + (row + 2) * inner + column]))
              d = combine(d, vld1q_f64(unsafeAddr source[inputBase + (row + 3) * inner + column]))
              row += 4
            var value = combine(combine(a, b), combine(c, d))
            for tail in row ..< axisLength:
              value = combine(value,
                              vld1q_f64(unsafeAddr source[inputBase + tail * inner + column]))
            when average:
              value = vdivq_f64(value, vdupq_n_f64(float64(axisLength)))
            vst1q_f64(addr destination[outputBase + column], value)
            column += 2
        for tail in column ..< inner:
          var value = identity
          for row in 0'u ..< axisLength:
            when multiply: value *= source[inputBase + row * inner + tail]
            else: value += source[inputBase + row * inner + tail]
          when average: value /= float64(axisLength)
          destination[outputBase + tail] = value

  proc reduceFloat32Span[T](input: ptr T; count: int;
                           multiply: static[bool]): float32 =
    ## Independent accumulators hide arithmetic latency without a temporary copy.
    let source = cast[ptr UncheckedArray[T]](input)
    let length = uint(count)
    const identity = when multiply: 1'f32 else: 0'f32
    var cursor = 0'u
    result = identity
    when defined(arm64):
      template combine(a, b: Float32x4): Float32x4 =
        when multiply: vmulq_f32(a, b)
        else: vaddq_f32(a, b)
      var a = vdupq_n_f32(identity)
      var b = a
      var c = a
      var d = a
      while cursor + 16 <= length:
        when T is int8 or T is uint8:
          var
            first {.noinit.}: Float32x4
            second {.noinit.}: Float32x4
            third {.noinit.}: Float32x4
            fourth {.noinit.}: Float32x4
          loadConvertedByteFloat32Pair(source, cursor, first, second)
          loadConvertedByteFloat32Pair(source, cursor + 8, third, fourth)
          a = combine(a, first)
          b = combine(b, second)
          c = combine(c, third)
          d = combine(d, fourth)
        else:
          a = combine(a, loadConvertedFloat32x4(source, cursor))
          b = combine(b, loadConvertedFloat32x4(source, cursor + 4))
          c = combine(c, loadConvertedFloat32x4(source, cursor + 8))
          d = combine(d, loadConvertedFloat32x4(source, cursor + 12))
        cursor += 16
      let lanes = combine(combine(a, b), combine(c, d))
      when multiply:
        result = ((vgetq_lane_f32(lanes, 0) * vgetq_lane_f32(lanes, 1)) *
                  vgetq_lane_f32(lanes, 2)) * vgetq_lane_f32(lanes, 3)
      else:
        result = (vgetq_lane_f32(lanes, 0) + vgetq_lane_f32(lanes, 1)) +
                 (vgetq_lane_f32(lanes, 2) + vgetq_lane_f32(lanes, 3))
    for index in cursor ..< length:
      when multiply: result *= convertedReductionValue(source[index], float32)
      else: result += convertedReductionValue(source[index], float32)

  proc reduceConvertedProductSpan*[T, U](input: ptr T; count: uint;
                                         dtype: typedesc[U]): U =
    ## Reuse native product grouping with casts before output-dtype arithmetic.
    when U is float32: reduceFloat32Span(input, count.int, true)
    else: reduceFloat64Span(input, count, true)

  proc reducePackedFloat32*(output, input: ptr float32;
                           outer, axisLength, inner: int;
                           multiply, average: static[bool]) =
    ## Write disjoint packed axis outputs, fusing means into the reduction.
    ## Callers establish nonempty input and row-major layout before entering.
    static: doAssert not (multiply and average)
    withCompilerOptimHints()
    let destination {.restrict.} = cast[ptr UncheckedArray[float32]](output)
    let source = cast[ptr UncheckedArray[float32]](input)
    # Validated storage bounds every unsigned offset; no per-element overflow guards.
    let groups = uint(outer)
    let length = uint(axisLength)
    let width = uint(inner)
    if inner == 1:
      for group in 0'u ..< groups:
        let value = reduceFloat32Span(unsafeAddr source[group * length],
                                     axisLength, multiply)
        when average: destination[group] = value / float32(axisLength)
        else: destination[group] = value
      return
    when defined(arm64) and multiply:
      reduceConvertedProductColumns(output, input, groups, length, width)
    else:
      const identity = when multiply: 1'f32 else: 0'f32
      for group in 0'u ..< groups:
        let inputBase = group * length * width
        let outputBase = group * width
        var column = 0'u
        when defined(arm64):
          template combine(a, b: Float32x4): Float32x4 = vaddq_f32(a, b)
          while column + 4 <= width:
            var a = vdupq_n_f32(identity)
            var b = a
            var c = a
            var d = a
            var row = 0'u
            while row + 4 <= length:
              a = combine(a, vld1q_f32(unsafeAddr source[inputBase + row * width + column]))
              b = combine(b, vld1q_f32(unsafeAddr source[inputBase + (row + 1) * width + column]))
              c = combine(c, vld1q_f32(unsafeAddr source[inputBase + (row + 2) * width + column]))
              d = combine(d, vld1q_f32(unsafeAddr source[inputBase + (row + 3) * width + column]))
              row += 4
            var value = combine(combine(a, b), combine(c, d))
            for tail in row ..< length:
              value = combine(value, vld1q_f32(unsafeAddr source[inputBase + tail * width + column]))
            when average: value = vdivq_f32(value, vdupq_n_f32(float32(axisLength)))
            vst1q_f32(addr destination[outputBase + column], value)
            column += 4
        for tail in column ..< width:
          var value = identity
          for row in 0'u ..< length:
            when multiply: value *= source[inputBase + row * width + tail]
            else: value += source[inputBase + row * width + tail]
          when average: value /= float32(axisLength)
          destination[outputBase + tail] = value

  proc sqrtFloat32*(output, input: ptr float32; count: int) =
    ## Write nonnegative-count, disjoint packed spans using hardware square roots.
    ## A separate translation unit keeps fallback fragments out of this hot loop.
    withCompilerOptimHints()
    let destination {.restrict.} = cast[ptr UncheckedArray[float32]](output)
    let source = cast[ptr UncheckedArray[float32]](input)
    var cursor = 0
    when defined(arm64):
      let vectorCount = count and not 3
      while cursor < vectorCount:
        let values = vld1q_f32(unsafeAddr source[cursor])
        vst1q_f32(addr destination[cursor], vsqrtq_f32(values))
        cursor += 4
    for index in cursor ..< count:
      destination[index] = math.sqrt(source[index])
when not defined(macosx):
  proc reducePackedFloat64*(output, input: ptr float64; count: uint;
                           multiply: static[bool]) =
    ## Reduce a validated packed span without temporary storage.
    let destination = cast[ptr UncheckedArray[float64]](output)
    let source = cast[ptr UncheckedArray[float64]](input)
    const identity = when multiply: 1.0 else: 0.0
    var a = identity
    var b = identity
    var c = identity
    var d = identity
    var cursor = 0'u
    while cursor + 4 <= count:
      when multiply:
        a *= source[cursor]
        b *= source[cursor + 1]
        c *= source[cursor + 2]
        d *= source[cursor + 3]
      else:
        a += source[cursor]
        b += source[cursor + 1]
        c += source[cursor + 2]
        d += source[cursor + 3]
      cursor += 4
    var value: float64
    when multiply: value = ((a * b) * c) * d
    else: value = ((a + b) + c) + d
    while cursor < count:
      when multiply: value *= source[cursor]
      else: value += source[cursor]
      inc cursor
    destination[0] = value

  proc reducePackedFloat64Axis*(output, input: ptr float64;
                               outer, axisLength, inner: uint;
                               multiply, average: static[bool]) =
    ## Reduce packed axes with independent scalar carries on non-Accelerate CPUs.
    static: doAssert not (multiply and average)
    withCompilerOptimHints()
    let destination {.restrict.} = cast[ptr UncheckedArray[float64]](output)
    let source = cast[ptr UncheckedArray[float64]](input)
    const identity = when multiply: 1.0 else: 0.0
    if inner == 1:
      for group in 0'u ..< outer:
        reducePackedFloat64(addr destination[group],
                            unsafeAddr source[group * axisLength],
                            axisLength, multiply)
        when average: destination[group] /= float64(axisLength)
      return
    for group in 0'u ..< outer:
      let inputBase = group * axisLength * inner
      let outputBase = group * inner
      for column in 0'u ..< inner:
        var value = identity
        for row in 0'u ..< axisLength:
          when multiply: value *= source[inputBase + row * inner + column]
          else: value += source[inputBase + row * inner + column]
        when average: value /= float64(axisLength)
        destination[outputBase + column] = value
