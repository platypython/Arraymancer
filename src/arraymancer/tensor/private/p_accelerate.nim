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
