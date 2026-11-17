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

## Provides allocation-free planning for shared C-order inner spans.
import ../data_structure

type InnerSpan*[N: static int] = object
  length*, outerRank*: int
  steps*: array[N, int]

func innerSpan*[N: static int](shape: Metadata; strides: array[N, Metadata]): InnerSpan[N] =
  ## Collapse trailing axes only when every operand advances by zero or one.
  result.length = 1
  result.outerRank = shape.len
  for axis in countdown(shape.len - 1, 0):
    if shape[axis] == 1:
      result.outerRank = axis
      continue
    var compatible = shape[axis] > 0
    for operand in 0 ..< N:
      if result.length == 1:
        if strides[operand][axis] notin {0, 1}: compatible = false
      elif strides[operand][axis] != result.steps[operand] * result.length:
        compatible = false
    if not compatible: break
    if result.length == 1:
      for operand in 0 ..< N:
        result.steps[operand] = strides[operand][axis]
    result.length *= shape[axis]
    result.outerRank = axis

func spanOffsets*[N: static int](shape: Metadata; strides: array[N, Metadata];
    origins: array[N, int]; outerRank, span: int): array[N, int] {.inline.} =
  ## Decode outer coordinates once, not once per output element.
  result = origins
  var remaining = span
  for axis in countdown(outerRank - 1, 0):
    let coordinate = remaining mod shape[axis]
    remaining = remaining div shape[axis]
    for operand in 0 ..< N:
      result[operand] += coordinate * strides[operand][axis]

func disjointOwnedStorage*[T, U: KnownSupportsCopyMem](a: Tensor[T], b: Tensor[U]): bool =
  ## Foreign buffers may overlap even when their storage wrappers differ.
  a.size > 0 and b.size > 0 and a.storage.isMemOwner and b.storage.isMemOwner and
    cast[pointer](a.storage.raw_buffer) != cast[pointer](b.storage.raw_buffer)

template withSpanStep*(step: int; stride: untyped; body: untyped) =
  ## Specialize the inner loop so the compiler sees unit-stride SIMD loads.
  if step == 0:
    const stride = 0
    body
  else:
    const stride = 1
    body
