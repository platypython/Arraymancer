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
import std/math
import ../../laser/compiler_optim_hints
when defined(macosx):
  {.passL: "-framework Accelerate".}

  proc vDSP_sve*(input: ptr float32; stride: clong; output: ptr float32; count: culong)
    {.importc, header: "<Accelerate/Accelerate.h>".}

  proc vvexpf*(output, input: ptr float32; count: ptr cint)
    {.importc, header: "<Accelerate/Accelerate.h>".}
  proc vvlogf*(output, input: ptr float32; count: ptr cint)
    {.importc, header: "<Accelerate/Accelerate.h>".}

  when defined(arm64):
    type Float32x4 {.importc: "float32x4_t", bycopy, header: "<arm_neon.h>".} = object
      raw: array[4, float32]
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

    proc scanPackedFloat32*(output, input: ptr float32;
                           outer, axisLength, inner: uint;
                           multiply, reverse: static[bool]) =
      ## Scan disjoint packed spans, preserving sequential float64 accumulation.
      ## SIMD lanes are separate scans, never reassociated prefixes of one scan.
      withCompilerOptimHints()
      let destination {.restrict.} = cast[ptr UncheckedArray[float32]](output)
      let source = cast[ptr UncheckedArray[float32]](input)
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
            let a = vset_lane_f32(source[base1 + index],
                                 vdup_n_f32(source[base0 + index]), 1)
            let b = vset_lane_f32(source[base3 + index],
                                 vdup_n_f32(source[base2 + index]), 1)
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
            when multiply: accumulated *= float64(source[index])
            else: accumulated += float64(source[index])
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
            let values = vld1q_f32(unsafeAddr source[index])
            low = combine(low, vcvt_f64_f32(vget_low_f32(values)))
            high = combine(high, vcvt_f64_f32(vget_high_f32(values)))
            vst1q_f32(addr destination[index],
                     vcombine_f32(vcvt_f32_f64(low), vcvt_f32_f64(high)))
          column += 4
        for tail in column ..< inner:
          var accumulated = identity
          for step in 0'u ..< axisLength:
            let index = base + coordinate(step) * inner + tail
            when multiply: accumulated *= float64(source[index])
            else: accumulated += float64(source[index])
            destination[index] = float32(accumulated)

    proc scanPackedFloat64*(output, input: ptr float64;
                           outer, axisLength, inner: uint;
                           multiply, reverse: static[bool]) =
      ## Vectorize independent scans while retaining each scan's ordered carry.
      withCompilerOptimHints()
      let destination {.restrict.} = cast[ptr UncheckedArray[float64]](output)
      let source = cast[ptr UncheckedArray[float64]](input)
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
              var values = vdupq_n_f64(source[base0 + index])
              values = vsetq_lane_f64(source[base1 + index], values, 1)
              accumulated = combine(accumulated, values)
              destination[base0 + index] = vgetq_lane_f64(accumulated, 0)
              destination[base1 + index] = vgetq_lane_f64(accumulated, 1)
            group += 2
        for tail in group ..< outer:
          let base = tail * axisLength
          var accumulated = identity
          for step in 0'u ..< axisLength:
            let index = base + coordinate(step)
            when multiply: accumulated *= source[index]
            else: accumulated += source[index]
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
              accumulated = combine(accumulated,
                                    vld1q_f64(unsafeAddr source[index]))
              vst1q_f64(addr destination[index], accumulated)
            column += 2
        for tail in column ..< inner:
          var accumulated = identity
          for step in 0'u ..< axisLength:
            let index = base + coordinate(step) * inner + tail
            when multiply: accumulated *= source[index]
            else: accumulated += source[index]
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


  proc reduceFloat64Span(input: ptr float64; count: uint;
                        multiply: static[bool]): float64 =
    ## Independent partial carries shorten the dependency chain for one span.
    let source = cast[ptr UncheckedArray[float64]](input)
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
        a = combine(a, vld1q_f64(unsafeAddr source[cursor]))
        b = combine(b, vld1q_f64(unsafeAddr source[cursor + 2]))
        c = combine(c, vld1q_f64(unsafeAddr source[cursor + 4]))
        d = combine(d, vld1q_f64(unsafeAddr source[cursor + 6]))
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
      when multiply: result = ((a * b) * c) * d
      else: result = ((a + b) + c) + d
    while cursor < count:
      when multiply: result *= source[cursor]
      else: result += source[cursor]
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
    const identity = when multiply: 1.0 else: 0.0
    if inner == 1:
      for group in 0'u ..< outer:
        let value = reduceFloat64Span(unsafeAddr source[group * axisLength],
                                      axisLength, multiply)
        when average: destination[group] = value / float64(axisLength)
        else: destination[group] = value
      return
    for group in 0'u ..< outer:
      let inputBase = group * axisLength * inner
      let outputBase = group * inner
      var column = 0'u
      when defined(arm64):
        template combine(a, b: Float64x2): Float64x2 =
          when multiply: vmulq_f64(a, b)
          else: vaddq_f64(a, b)
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

  proc reduceFloat32Span(input: ptr float32; count: int;
                        multiply: static[bool]): float32 =
    ## Independent accumulators hide arithmetic latency without a temporary copy.
    let source = cast[ptr UncheckedArray[float32]](input)
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
        a = combine(a, vld1q_f32(unsafeAddr source[cursor]))
        b = combine(b, vld1q_f32(unsafeAddr source[cursor + 4]))
        c = combine(c, vld1q_f32(unsafeAddr source[cursor + 8]))
        d = combine(d, vld1q_f32(unsafeAddr source[cursor + 12]))
        cursor += 16
      let lanes = combine(combine(a, b), combine(c, d))
      when multiply:
        result = (vgetq_lane_f32(lanes, 0) * vgetq_lane_f32(lanes, 1)) *
                 (vgetq_lane_f32(lanes, 2) * vgetq_lane_f32(lanes, 3))
      else:
        result = (vgetq_lane_f32(lanes, 0) + vgetq_lane_f32(lanes, 1)) +
                 (vgetq_lane_f32(lanes, 2) + vgetq_lane_f32(lanes, 3))
    for index in cursor ..< length:
      when multiply: result *= source[index]
      else: result += source[index]

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
    const identity = when multiply: 1'f32 else: 0'f32
    for group in 0'u ..< groups:
      let inputBase = group * length * width
      let outputBase = group * width
      var column = 0'u
      when defined(arm64):
        template combine(a, b: Float32x4): Float32x4 =
          when multiply: vmulq_f32(a, b)
          else: vaddq_f32(a, b)
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
