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

## Provides rounded-forward logarithmic VJPs with one final gradient allocation.
import std/[complex, math]
import ../../tensor/[data_structure, init_cpu]
import ../../tensor/private/p_logsumexp
when defined(macosx):
  import ../../tensor/private/p_accelerate

when defined(macosx) and defined(arm64) and defined(clang):
  {.localPassC: "-mno-outline".}

proc cpuLogsumexpGradient*[T: SomeFloat or Complex](input, normalizer,
    seed: Tensor[T]; reduced: Metadata): Tensor[T] =
  ## Rounded subtraction, exponential and multiplication match the forward dtype.
  ## The gate supplies validated axes and an unchanged packed forward result.
  result = newTensorUninit[T](input.shape)
  if result.size == 0: return
  when T is Complex:
    type F = typeof(default(T).re)
  else:
    type F = T
  const Tile = 256'u
  var shifted {.noinit.}, terms {.noinit.}: array[Tile, F]
  var weights {.noinit.}: array[Tile, T]
  when T is Complex:
    var phases {.noinit.}, sines {.noinit.}, cosines {.noinit.}: array[Tile, F]
    var cachedNormalizer {.noinit.}: T
    var cachedNormalizerOffset = high(uint)
  let source = input.storage.raw_buffer
  let chosen = normalizer.storage.raw_buffer
  let incoming = seed.storage.raw_buffer
  let destination = result.storage.raw_buffer

  template prepare(index, inputOffset, normalizerOffset, seedOffset: untyped) =
    block:
      when T is Complex:
        let reducedOffset = uint(normalizerOffset)
        if reducedOffset != cachedNormalizerOffset:
          # Tensor subtraction applies complex alpha=1 before subtracting.
          # Its zero cross-products turn infinite components into NaNs.
          cachedNormalizer = T(re: F(1), im: F(0)) * chosen[normalizerOffset]
          cachedNormalizerOffset = reducedOffset
        let difference = source[inputOffset] - cachedNormalizer
        shifted[index] = difference.re
        phases[index] = difference.im
      else:
        shifted[index] = source[inputOffset] - chosen[normalizerOffset]
      weights[index] = incoming[seedOffset]

  template finish(first, count: untyped) =
    block:
      when defined(macosx):
        var countC = cint(count)
        when F is float32: vvexpf(addr terms[0], addr shifted[0], addr countC)
        else: vvexp(addr terms[0], addr shifted[0], addr countC)
        when T is Complex:
          when F is float32:
            vvsincosf(addr sines[0], addr cosines[0], addr phases[0], addr countC)
          else:
            vvsincos(addr sines[0], addr cosines[0], addr phases[0], addr countC)
      for index in 0'u ..< count:
        when T is Complex:
          let exponential =
            when defined(macosx):
              complexExponentialParts(shifted[index], phases[index], terms[index],
                sines[index], cosines[index])
            else:
              exponentialComplex(Complex[F](re: shifted[index], im: phases[index]))
          destination[first + index] = weights[index] * exponential.conjugate
        else:
          let exponential =
            when defined(macosx): terms[index]
            else: exp(shifted[index])
          destination[first + index] = weights[index] * exponential

  var firstAxis = int(input.rank)
  var lastAxis = -1
  var selectedCount = 0
  for axis in 0 ..< input.rank:
    if reduced[axis] != 0:
      firstAxis = min(firstAxis, axis)
      lastAxis = axis
      inc selectedCount
  if input.is_C_contiguous and seed.is_C_contiguous and
      selectedCount == lastAxis - firstAxis + 1:
    var outer = 1'u
    var length = 1'u
    var inner = 1'u
    for axis in 0 ..< firstAxis: outer *= uint(input.shape[axis])
    for axis in firstAxis .. lastAxis: length *= uint(input.shape[axis])
    for axis in lastAxis + 1 ..< input.rank: inner *= uint(input.shape[axis])
    let inputOffset = uint(input.offset)
    let normalizerOffset = uint(normalizer.offset)
    let seedOffset = uint(seed.offset)
    let blockLength = length * inner
    for blockIndex in 0'u ..< outer:
      let base = blockIndex * blockLength
      let reducedBase = blockIndex * inner
      var first = 0'u
      var column = 0'u
      while first < blockLength:
        let count = min(Tile, blockLength - first)
        for index in 0'u ..< count:
          prepare(index, inputOffset + base + first + index,
            normalizerOffset + reducedBase + column,
            seedOffset + reducedBase + column)
          inc column
          if column == inner: column = 0
        finish(base + first, count)
        first += count
    return

  # Virtual unsqueezing preserves signed and broadcast seed strides without views.
  var chosenSteps, seedSteps, coordinates: Metadata
  chosenSteps.len = input.rank
  seedSteps.len = input.rank
  coordinates.len = input.rank
  var retainedAxis = 0
  for axis in 0 ..< input.rank:
    if reduced[axis] == 0:
      let outputAxis = if normalizer.rank == input.rank: axis else: retainedAxis
      chosenSteps[axis] = normalizer.strides[outputAxis]
      seedSteps[axis] = seed.strides[outputAxis]
      inc retainedAxis
  var inputOffset = input.offset
  var normalizerOffset = normalizer.offset
  var seedOffset = seed.offset
  var first = 0'u
  let total = uint(input.size)
  while first < total:
    let count = min(Tile, total - first)
    for index in 0'u ..< count:
      prepare(index, inputOffset, normalizerOffset, seedOffset)
      for axis in countdown(input.rank - 1, 0):
        inc coordinates[axis]
        inputOffset += input.strides[axis]
        normalizerOffset += chosenSteps[axis]
        seedOffset += seedSteps[axis]
        if coordinates[axis] < input.shape[axis]: break
        coordinates[axis] = 0
        inputOffset -= input.shape[axis] * input.strides[axis]
        normalizerOffset -= input.shape[axis] * chosenSteps[axis]
        seedOffset -= input.shape[axis] * seedSteps[axis]
    finish(first, count)
    first += count
