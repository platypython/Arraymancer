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
