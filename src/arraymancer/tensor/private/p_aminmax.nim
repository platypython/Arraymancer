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

## Provides fused CPU minimum/maximum folds without input materialization.
import std/complex
import ../data_structure, ../init_cpu
import ./p_accelerate, ./p_extrema

when defined(macosx) and defined(arm64) and defined(clang):
  {.localPassC: "-mno-outline".}

when defined(macosx) and defined(arm64):
  {.emit: """
  #include <arm_neon.h>
  #include <stddef.h>
  static inline void am_aminmax_whole_f32(const float *source, size_t count,
                                         float *minimum, float *maximum) {
    float32x4_t low = vld1q_f32(source), high = low;
    size_t position = 4;
    for (; position + 4 <= count; position += 4) {
      float32x4_t value = vld1q_f32(source + position);
      low = vminq_f32(low, value);
      high = vmaxq_f32(high, value);
    }
    if (position != count) {
      float lows[4], highs[4];
      vst1q_f32(lows, low);
      vst1q_f32(highs, high);
      for (size_t lane = 0; position < count; ++lane, ++position) {
        float32x4_t value = vdupq_n_f32(source[position]);
        lows[lane] = vgetq_lane_f32(vminq_f32(vdupq_n_f32(lows[lane]), value), 0);
        highs[lane] = vgetq_lane_f32(vmaxq_f32(vdupq_n_f32(highs[lane]), value), 0);
      }
      low = vld1q_f32(lows);
      high = vld1q_f32(highs);
    }
    low = vminq_f32(low, vextq_f32(low, low, 2));
    high = vmaxq_f32(high, vextq_f32(high, high, 2));
    low = vminq_f32(low, vrev64q_f32(low));
    high = vmaxq_f32(high, vrev64q_f32(high));
    *minimum = vgetq_lane_f32(low, 0);
    *maximum = vgetq_lane_f32(high, 0);
  }
  static inline void am_aminmax_whole_f64(const double *source, size_t count,
                                         double *minimum, double *maximum) {
    float64x2_t low = vld1q_f64(source), high = low;
    size_t position = 2;
    for (; position + 2 <= count; position += 2) {
      float64x2_t value = vld1q_f64(source + position);
      low = vminq_f64(low, value);
      high = vmaxq_f64(high, value);
    }
    if (position != count) {
      float64x2_t value = vdupq_n_f64(source[position]);
      float64x2_t lowTail = vminq_f64(low, value);
      float64x2_t highTail = vmaxq_f64(high, value);
      low = vsetq_lane_f64(vgetq_lane_f64(lowTail, 0), low, 0);
      high = vsetq_lane_f64(vgetq_lane_f64(highTail, 0), high, 0);
    }
    *minimum = vgetq_lane_f64(vminq_f64(low, vextq_f64(low, low, 1)), 0);
    *maximum = vgetq_lane_f64(vmaxq_f64(high, vextq_f64(high, high, 1)), 0);
  }
  #define AM_AMINMAX_AXIS(S, U, T, WORD, V, MASK, WIDTH) \
  static inline void am_aminmax_combine_##S(V *low, V *high, V value) { \
    MASK ordered = vceqq_##S(*low, *low); \
    MASK replaceLow = vbicq_##U(ordered, vcgeq_##S(value, *low)); \
    MASK replaceHigh = vbicq_##U(ordered, vcleq_##S(value, *high)); \
    *low = vbslq_##S(replaceLow, value, *low); \
    *high = vbslq_##S(replaceHigh, value, *high); \
  } \
  static inline void am_aminmax_span_##S(const T *source, size_t count, \
                                        T *minimum, T *maximum) { \
    T firstZero = 0; \
    int seenZero = 0; \
    T lowScalar, highScalar; \
    size_t position = 0; \
    if (count >= WIDTH) { \
      V low = vld1q_##S(source), high = low; \
      do { \
        V value = vld1q_##S(source + position); \
        MASK ordered = vceqq_##S(value, value); \
        if (vaddvq_##U(ordered) != (WORD)(0 - WIDTH)) { \
          T lanes[WIDTH]; \
          vst1q_##S(lanes, value); \
          for (size_t lane = 0; lane < WIDTH; ++lane) { \
            if (lanes[lane] != lanes[lane]) { \
              *minimum = lanes[lane]; *maximum = lanes[lane]; return; \
            } \
          } \
        } \
        if (!seenZero && vaddvq_##U(vceqq_##S(value, vdupq_n_##S(0))) != 0) { \
          T lanes[WIDTH]; \
          vst1q_##S(lanes, value); \
          for (size_t lane = 0; lane < WIDTH; ++lane) { \
            if (lanes[lane] == 0) { firstZero = lanes[lane]; seenZero = 1; break; } \
          } \
        } \
        low = vminq_##S(low, value); \
        high = vmaxq_##S(high, value); \
        position += WIDTH; \
      } while (position + WIDTH <= count); \
      lowScalar = vminvq_##S(low); \
      highScalar = vmaxvq_##S(high); \
    } else { \
      lowScalar = highScalar = source[0]; \
      if (lowScalar != lowScalar) { *minimum = lowScalar; *maximum = lowScalar; return; } \
      if (lowScalar == 0) { firstZero = lowScalar; seenZero = 1; } \
      position = 1; \
    } \
    for (; position < count; ++position) { \
      T value = source[position]; \
      if (value != value) { *minimum = value; *maximum = value; return; } \
      if (!seenZero && value == 0) { firstZero = value; seenZero = 1; } \
      if (value < lowScalar) lowScalar = value; \
      if (value > highScalar) highScalar = value; \
    } \
    *minimum = lowScalar == 0 ? firstZero : lowScalar; \
    *maximum = highScalar == 0 ? firstZero : highScalar; \
  } \
  static inline void am_aminmax_axis_##S(const T *source, T *minimum, T *maximum, \
                                        size_t outer, size_t length, size_t inner) { \
    for (size_t block = 0; block < outer; ++block) { \
      const T *input = source + block * length * inner; \
      T *lowOutput = minimum + block * inner; \
      T *highOutput = maximum + block * inner; \
      if (inner == 1) { \
        am_aminmax_span_##S(input, length, lowOutput, highOutput); \
        continue; \
      } \
      size_t column = 0; \
      for (; column + 8 * WIDTH <= inner; column += 8 * WIDTH) { \
        V lows[8], highs[8]; \
        for (size_t vector = 0; vector < 8; ++vector) { \
          lows[vector] = vld1q_##S(input + column + vector * WIDTH); \
          highs[vector] = lows[vector]; \
        } \
        for (size_t step = 1; step < length; ++step) { \
          for (size_t vector = 0; vector < 8; ++vector) { \
            V value = vld1q_##S(input + step * inner + column + vector * WIDTH); \
            am_aminmax_combine_##S(&lows[vector], &highs[vector], value); \
          } \
        } \
        for (size_t vector = 0; vector < 8; ++vector) { \
          vst1q_##S(lowOutput + column + vector * WIDTH, lows[vector]); \
          vst1q_##S(highOutput + column + vector * WIDTH, highs[vector]); \
        } \
      } \
      for (; column + WIDTH <= inner; column += WIDTH) { \
        V low = vld1q_##S(input + column), high = low; \
        for (size_t step = 1; step < length; ++step) { \
          V value = vld1q_##S(input + step * inner + column); \
          am_aminmax_combine_##S(&low, &high, value); \
        } \
        vst1q_##S(lowOutput + column, low); \
        vst1q_##S(highOutput + column, high); \
      } \
      for (; column < inner; ++column) { \
        T low = input[column], high = low; \
        if (low == low) { \
          for (size_t step = 1; step < length; ++step) { \
            T value = input[step * inner + column]; \
            if (!(value >= low)) { \
              low = value; \
              if (value != value) { high = value; break; } \
            } else if (!(value <= high)) high = value; \
          } \
        } \
        lowOutput[column] = low; highOutput[column] = high; \
      } \
    } \
  }
  AM_AMINMAX_AXIS(f32, u32, float, uint32_t, float32x4_t, uint32x4_t, 4)
  AM_AMINMAX_AXIS(f64, u64, double, uint64_t, float64x2_t, uint64x2_t, 2)
  #undef AM_AMINMAX_AXIS
  """.}
  proc aminmaxNativeWhole(source: ptr float32; count: csize_t;
      minimum, maximum: ptr float32) {.importc: "am_aminmax_whole_f32", nodecl.}
  proc aminmaxNativeWhole(source: ptr float64; count: csize_t;
      minimum, maximum: ptr float64) {.importc: "am_aminmax_whole_f64", nodecl.}
  proc aminmaxNativeAxis(source, minimum, maximum: ptr float32;
      outer, length, inner: csize_t) {.importc: "am_aminmax_axis_f32", nodecl.}
  proc aminmaxNativeAxis(source, minimum, maximum: ptr float64;
      outer, length, inner: csize_t) {.importc: "am_aminmax_axis_f64", nodecl.}

proc aminmaxSpan[T; I: SomeInteger](source: ptr UncheckedArray[T]; base: I;
    count: uint; stride: I): tuple[min, max: T] {.inline.} =
  ## Dimension reductions keep the first equal value and first NaN bits.
  result.min = source[base]
  result.max = result.min
  when T is SomeFloat:
    if result.min != result.min: return
  when T is bool:
    for index in 1'u ..< count:
      if source[base + I(index) * stride] != result.min: return (false, true)
  else:
    for index in 1'u ..< count:
      let value = source[base + I(index) * stride]
      when T is SomeFloat:
        if not (value >= result.min):
          result.min = value
          if value != value: return (value, value)
        elif not (value <= result.max):
          result.max = value
      else:
        if value < result.min: result.min = value
        if value > result.max: result.max = value

proc aminmaxWhole[T](arg: Tensor[T]; packed: static[bool]): tuple[min, max: T] =
  ## Whole floats use one native-width chain in logical row-major order.
  let source = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
  let count = arg.size
  var offset = 0
  when not packed:
    var coordinates: Metadata
    coordinates.len = arg.rank
  template nextValue(): T =
    block:
      let value = source[offset]
      when packed:
        inc offset
      else:
        for dimension in countdown(arg.rank - 1, 0):
          inc coordinates[dimension]
          offset += arg.strides[dimension]
          if coordinates[dimension] < arg.shape[dimension]: break
          coordinates[dimension] = 0
          offset -= arg.shape[dimension] * arg.strides[dimension]
      value
  when T is SomeFloat:
    const vectorBytes = when defined(arm64): 16 else: 32
    const width = vectorBytes div sizeof(T)
    var minima, maxima {.noinit.}: array[width, T]
    let active = min(count, width)
    for lane in 0 ..< active:
      let value = nextValue()
      minima[lane] = value
      maxima[lane] = value
    var position = active
    while position + width <= count:
      for lane in 0 ..< width:
        let value = nextValue()
        minima[lane] = extremaVectorValue(minima[lane], value, false)
        maxima[lane] = extremaVectorValue(maxima[lane], value, true)
      position += width
    for lane in 0 ..< count - position:
      let value = nextValue()
      minima[lane] = extremaVectorValue(minima[lane], value, false)
      maxima[lane] = extremaVectorValue(maxima[lane], value, true)
    result = (minima[0], maxima[0])
    when sizeof(T) == 4:
      if active == width:
        # Shuffle horizontal folds visit lanes in bit-reversed priority order.
        for index in 1 ..< width:
          var bits = index
          var lane = 0
          var remaining = width
          while remaining > 1:
            lane = (lane shl 1) or (bits and 1)
            bits = bits shr 1
            remaining = remaining shr 1
          result.min = extremaVectorValue(result.min, minima[lane], false)
          result.max = extremaVectorValue(result.max, maxima[lane], true)
        return
    for lane in 1 ..< active:
      result.min = extremaVectorValue(result.min, minima[lane], false)
      result.max = extremaVectorValue(result.max, maxima[lane], true)
  else:
    let first = nextValue()
    result = (first, first)
    for index in 1 ..< count:
      let value = nextValue()
      result.min = extremaScalar(result.min, value, false)
      result.max = extremaScalar(result.max, value, true)
      when T is bool:
        if not result.min and result.max: return

proc aminmaxPacked[T](source, minima, maxima: ptr UncheckedArray[T];
    outer, length, inner: uint) =
  ## Packed positions stay within validated storage; unsigned math avoids checks.
  when defined(macosx) and defined(arm64) and T is SomeFloat:
    aminmaxNativeAxis(cast[ptr T](source), cast[ptr T](minima),
      cast[ptr T](maxima), csize_t(outer), csize_t(length), csize_t(inner))
  else:
    for blockIndex in 0'u ..< outer:
      let base = blockIndex * length * inner
      if inner == 1:
        let pair = aminmaxSpan(source, base, length, 1'u)
        minima[blockIndex] = pair.min
        maxima[blockIndex] = pair.max
      else:
        when T is bool:
          for column in 0'u ..< inner:
            let pair = aminmaxSpan(source, base + column, length, inner)
            minima[blockIndex * inner + column] = pair.min
            maxima[blockIndex * inner + column] = pair.max
        else:
          var column = 0'u
          while column < inner:
            const tile = 32
            let active = min(uint(tile), inner - column)
            var low, high {.noinit.}: array[tile, T]
            for lane in 0'u ..< active:
              let value = source[base + column + lane]
              low[lane] = value
              high[lane] = value
            for step in 1'u ..< length:
              for lane in 0'u ..< active:
                let value = source[base + step * inner + column + lane]
                when T is SomeFloat:
                  let replaceMinimum = uint8(low[lane] == low[lane]) and
                    uint8(not (value >= low[lane]))
                  let replaceMaximum = uint8(high[lane] == high[lane]) and
                    uint8(not (value <= high[lane]))
                  if replaceMinimum != 0: low[lane] = value
                  if replaceMaximum != 0: high[lane] = value
                else:
                  low[lane] = extremaScalar(low[lane], value, false)
                  high[lane] = extremaScalar(high[lane], value, true)
            for lane in 0'u ..< active:
              minima[blockIndex * inner + column + lane] = low[lane]
              maxima[blockIndex * inner + column + lane] = high[lane]
            column += active

proc reduceAminmax*[T: SomeNumber or bool or Complex](arg: Tensor[T]; axis: int;
    whole: static[bool]; keepdim: bool): tuple[min, max: Tensor[T]] =
  ## Validate before allocating the two independent final output tensors.
  when T is Complex:
    raise newException(ValueError, "aminmax unsupported complex dtype")
  elif whole and T is SomeUnsignedInt and sizeof(T) > 1:
    if arg.size == 0:
      raise newException(ValueError, "aminmax cannot reduce an empty tensor")
    raise newException(ValueError, "aminmax unsupported CPU dtype")
  else:
    var dimension = 0
    var length = arg.size
    when whole:
      if length == 0:
        raise newException(ValueError, "aminmax cannot reduce an empty tensor")
    else:
      dimension = normalizedExtremaAxis(arg.rank, axis)
      length = if arg.rank == 0: 1 else: arg.shape[dimension]
      if length == 0:
        raise newException(IndexDefect, "aminmax empty reduced axis")
    when T is SomeUnsignedInt and sizeof(T) > 1:
      if arg.rank != 0:
        raise newException(ValueError, "aminmax unsupported CPU dtype")
    var outputShape: Metadata
    var outputCount = 1
    for index in 0 ..< arg.rank:
      let selected = whole or index == dimension
      if selected:
        if keepdim:
          inc outputShape.len
          outputShape[outputShape.len - 1] = 1
      else:
        inc outputShape.len
        outputShape[outputShape.len - 1] = arg.shape[index]
        outputCount *= arg.shape[index]
    result.min = newTensorUninit[T](outputShape)
    result.max = newTensorUninit[T](outputShape)
    if outputCount == 0: return
    let source = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
    let minima = cast[ptr UncheckedArray[T]](result.min.get_offset_ptr)
    let maxima = cast[ptr UncheckedArray[T]](result.max.get_offset_ptr)
    if arg.rank == 0:
      minima[0] = source[0]
      maxima[0] = source[0]
      return
    when whole:
      when defined(macosx) and defined(arm64) and T is SomeFloat:
        if arg.is_C_contiguous and length >= 16 div sizeof(T):
          aminmaxNativeWhole(cast[ptr T](source), csize_t(length),
            cast[ptr T](minima), cast[ptr T](maxima))
          return
      let pair = if arg.is_C_contiguous: aminmaxWhole(arg, true)
        else: aminmaxWhole(arg, false)
      minima[0] = pair.min
      maxima[0] = pair.max
    else:
      if arg.is_C_contiguous:
        var outer = 1'u
        var inner = 1'u
        for index in 0 ..< dimension: outer *= uint(arg.shape[index])
        for index in dimension + 1 ..< arg.rank: inner *= uint(arg.shape[index])
        aminmaxPacked(source, minima, maxima, outer, uint(length), inner)
      else:
        var coordinates: Metadata
        coordinates.len = arg.rank
        var offset = 0
        for outputIndex in 0 ..< outputCount:
          let pair = aminmaxSpan(source, offset, uint(length), arg.strides[dimension])
          minima[outputIndex] = pair.min
          maxima[outputIndex] = pair.max
          for index in countdown(arg.rank - 1, 0):
            if index != dimension:
              inc coordinates[index]
              offset += arg.strides[index]
              if coordinates[index] < arg.shape[index]: break
              coordinates[index] = 0
              offset -= arg.shape[index] * arg.strides[index]
