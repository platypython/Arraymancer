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

## Provides private macOS float32 bulk operations.
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
    proc vmulq_f64(a, b: Float64x2): Float64x2
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
