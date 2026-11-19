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

## Provides ordered logarithmic scans with wide, unrounded prefix carries.
import std/[complex, math]
import ../data_structure, ../init_cpu
import ./p_extrema, ./p_logsumexp

when defined(macosx) and defined(arm64) and defined(clang):
  {.localPassC: "-mno-outline".}

proc logarithmicOnePlus(value: float64): float64
  {.importc: "log1p", header: "<math.h>".}

proc logarithmicAdd(x, y: float64): float64 {.inline.} =
  ## Equal infinities bypass the otherwise indeterminate subtraction.
  let minimum = if y != y: y elif y < x: y else: x
  let maximum = if y != y: y elif x < y: y else: x
  if minimum != maximum or abs(minimum) != Inf:
    logarithmicOnePlus(exp(minimum - maximum)) + maximum
  else: x

proc logarithmicOnePlus(z: Complex[float64]): Complex[float64] {.inline.} =
  ## Apple's pinned branch avoids cancellation without wrapping the carried phase.
  when defined(macosx):
    let angle = arctan2(z.im, z.re + 1.0)
    if abs(z) < 0.5:
      let remainder = z.re * (2.0 + z.re) + z.im * z.im
      let real = if remainder == 0.0: z.re else: 0.5 * logarithmicOnePlus(remainder)
      return Complex[float64](re: real, im: angle)
    Complex[float64](re: ln(hypot(z.re + 1.0, z.im)), im: angle)
  else:
    let one = Complex[float64](re: 1.0, im: 0.0)
    let incremented = z + one
    if incremented == one: return z
    let logarithm = complex.ln(incremented)
    if incremented - one == z: return logarithm
    logarithm * (z / (incremented - one))

proc logarithmicAdd(x, y: Complex[float64]): Complex[float64] {.inline.} =
  ## NaNs infect both components; equal real infinities have explicit phase rules.
  if x.re != x.re or x.im != x.im or y.re != y.re or y.im != y.im:
    return Complex[float64](re: NaN, im: NaN)
  let minimum = if x.re < y.re: x else: y
  let maximum = if x.re < y.re: y else: x
  if abs(minimum.re) == Inf and minimum.re == maximum.re:
    if minimum.re < 0: return minimum
    return complex.ln(exponentialComplex(minimum) + exponentialComplex(maximum))
  logarithmicOnePlus(exponentialComplex(minimum - maximum)) + maximum

proc logarithmicScanStep*[T, A](accumulated: var A; value: T): T {.inline.} =
  ## Store a rounded observation while keeping the full-width recurrence alive.
  when T is Complex:
    type F = typeof(default(T).re)
    accumulated = logarithmicAdd(
      Complex[float64](re: float64(value.re), im: float64(value.im)), accumulated)
    T(re: F(accumulated.re), im: F(accumulated.im))
  else:
    accumulated = logarithmicAdd(float64(value), accumulated)
    T(accumulated)

proc logarithmicScanSpan[T: SomeFloat or Complex; I: SomeInteger](
    destination, source: ptr UncheckedArray[T]; outputBase, inputBase: I;
    count: uint; outputStride, inputStride: I) =
  ## The carry is wider than float32/complex64 outputs, never their rounded prefix.
  when T is Complex:
    var accumulated = Complex[float64](re: -Inf, im: 0.0)
  else:
    var accumulated = -Inf
  for index in 0'u ..< count:
    destination[outputBase + I(index) * outputStride] =
      logarithmicScanStep(accumulated, source[inputBase + I(index) * inputStride])

proc logarithmicScanPacked[T: SomeFloat or Complex](
    destination, source: ptr UncheckedArray[T]; inputOffset, groups, length,
    inner: uint) =
  ## Independent chains hide transcendental latency and coalesce column reads.
  if groups == 1:
    logarithmicScanSpan(destination, source, 0'u, inputOffset, length, 1'u, 1'u)
    return
  const Lanes = 32'u
  var groupStart = 0'u
  while groupStart < groups:
    let width = min(Lanes, groups - groupStart)
    var bases {.noinit.}: array[Lanes, uint]
    when T is Complex:
      var accumulated {.noinit.}: array[Lanes, Complex[float64]]
    else:
      var accumulated {.noinit.}: array[Lanes, float64]
    for lane in 0'u ..< width:
      let group = groupStart + lane
      bases[lane] = (group div inner) * length * inner + group mod inner
      when T is Complex:
        accumulated[lane] = Complex[float64](re: -Inf, im: 0.0)
      else:
        accumulated[lane] = -Inf
    for index in 0'u ..< length:
      for lane in 0'u ..< width:
        let position = bases[lane] + index * inner
        destination[position] =
          logarithmicScanStep(accumulated[lane], source[inputOffset + position])
    groupStart += width

proc scanLogcumsumexp*[T: SomeNumber or bool or Complex](arg: Tensor[T];
    axis: int): Tensor[T] =
  ## Validate before dtype dispatch; only independent output storage is allocated.
  let dimension = normalizedExtremaAxis(arg.rank, axis)
  when T is SomeInteger or T is bool:
    raise newException(ValueError, "logcumsumexp supports floating and complex CPU inputs")
  else:
    result = newTensorUninit[T](arg.shape)
    if result.size == 0: return
    let source = arg.storage.raw_buffer
    let destination = result.storage.raw_buffer
    if arg.rank == 0:
      destination[0] = source[arg.offset]
      return
    let length = uint(arg.shape[dimension])
    var outer = 1'u
    var inner = 1'u
    for index in 0 ..< dimension: outer *= uint(arg.shape[index])
    for index in dimension + 1 ..< arg.rank: inner *= uint(arg.shape[index])
    let groups = outer * inner
    if arg.is_C_contiguous:
      logarithmicScanPacked(destination, source, uint(arg.offset), groups, length, inner)
      return
    var coordinates: Metadata
    coordinates.len = arg.rank
    var inputBase = arg.offset
    for group in 0'u ..< groups:
      let base = (group div inner) * length * inner + group mod inner
      logarithmicScanSpan(destination, source, int(base), inputBase, length,
        int(inner), arg.strides[dimension])
      if group + 1 < groups:
        for index in countdown(arg.rank - 1, 0):
          if index == dimension: continue
          inc coordinates[index]
          inputBase += arg.strides[index]
          if coordinates[index] < arg.shape[index]: break
          coordinates[index] = 0
          inputBase -= arg.shape[index] * arg.strides[index]
