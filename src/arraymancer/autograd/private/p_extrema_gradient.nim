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

## Computes first-order extrema VJPs without materializing masks or counts.

import ../../tensor/[data_structure, init_cpu],
       ../../laser/compiler_optim_hints

func strideMagnitude(stride: int): uint =
  ## Unsigned magnitude also handles the smallest signed stride.
  if stride < 0: 0'u - cast[uint](stride)
  else: uint(stride)

proc packedExtremaGradient[T: SomeFloat](output, input, extrema, seed: ptr T;
    outer, length, inner: uint) =
  ## Column tiles keep counts local while both data passes remain vectorizable.
  withCompilerOptimHints()
  let source = cast[ptr UncheckedArray[T]](input)
  let destination {.restrict.} = cast[ptr UncheckedArray[T]](output)
  let chosen = cast[ptr UncheckedArray[T]](extrema)
  let upstream = cast[ptr UncheckedArray[T]](seed)
  if inner == 1:
    for group in 0'u ..< outer:
      let base = group * length
      let target = chosen[group]
      var matches = 0'u
      for index in 0'u ..< length:
        matches += uint(ord(source[base + index] == target))
      let weight = upstream[group] / T(matches)
      for index in 0'u ..< length:
        destination[base + index] = weight * T(ord(source[base + index] == target))
  else:
    const tile = 32
    for group in 0'u ..< outer:
      let base = group * length * inner
      var column = 0'u
      while column < inner:
        let columns = min(uint(tile), inner - column)
        let targetBase = group * inner + column
        var counts {.noinit.}: array[tile, uint]
        var weights {.noinit.}: array[tile, T]
        for lane in 0'u ..< columns: counts[lane] = 0
        for row in 0'u ..< length:
          let offset = base + row * inner + column
          for lane in 0'u ..< columns:
            counts[lane] += uint(ord(source[offset + lane] == chosen[targetBase + lane]))
        for lane in 0'u ..< columns:
          weights[lane] = upstream[targetBase + lane] / T(counts[lane])
        for row in 0'u ..< length:
          let offset = base + row * inner + column
          for lane in 0'u ..< columns:
            # Multiplying the mask preserves -0 and the inactive Inf * 0 NaNs.
            destination[offset + lane] = weights[lane] *
              T(ord(source[offset + lane] == chosen[targetBase + lane]))
        column += columns

proc stridedSeedValue[T](seed: Tensor[T], index: int): T =
  ## Match logical output order when a cached gradient is not row-major.
  var offset = seed.offset
  var remaining = index
  for axis in countdown(seed.rank - 1, 0):
    offset += (remaining mod seed.shape[axis]) * seed.strides[axis]
    remaining = remaining div seed.shape[axis]
  seed.storage.raw_buffer[offset]

proc cpuExtremaGradient*[T: SomeFloat](input, extrema, seed: Tensor[T];
    reduced: Metadata): Tensor[T] =
  ## The gate supplies validated axes and its unchanged packed forward result.
  ## Every input position receives (seed / tie count) * its numeric match mask.
  result = newTensorUninit[T](input.shape)
  if result.size == 0: return
  let source = input.storage.raw_buffer
  let destination = result.storage.raw_buffer
  let chosenValues = extrema.storage.raw_buffer
  let upstreamValues = seed.storage.raw_buffer
  if input.rank == 0:
    let matched = T(ord(source[input.offset] == chosenValues[extrema.offset]))
    destination[0] = (upstreamValues[seed.offset] / matched) * matched
    return

  var first = input.rank
  var last = -1
  var selectedCount = 0
  for axis in 0 ..< input.rank:
    if reduced[axis] != 0:
      first = min(first, axis)
      last = axis
      inc selectedCount
  let packedSeed = seed.is_C_contiguous
  if input.is_C_contiguous and packedSeed and selectedCount == last - first + 1:
    var outer = 1'u
    var length = 1'u
    var inner = 1'u
    for axis in 0 ..< first: outer *= uint(input.shape[axis])
    for axis in first .. last: length *= uint(input.shape[axis])
    for axis in last + 1 ..< input.rank: inner *= uint(input.shape[axis])
    packedExtremaGradient(addr destination[0], unsafeAddr source[input.offset],
      unsafeAddr chosenValues[extrema.offset], unsafeAddr upstreamValues[seed.offset],
      outer, length, inner)
    return

  var selected, retained: Metadata
  for axis in 0 ..< input.rank:
    if reduced[axis] != 0:
      selected.add axis
    else:
      retained.add axis
  for position in 1 ..< selected.len:
    let axis = selected[position]
    let magnitude = strideMagnitude(input.strides[axis])
    var insertion = position
    while insertion > 0:
      let previous = selected[insertion - 1]
      let previousMagnitude = strideMagnitude(input.strides[previous])
      if previousMagnitude < magnitude or
          (previousMagnitude == magnitude and result.strides[previous] <= result.strides[axis]):
        break
      selected[insertion] = previous
      dec insertion
    selected[insertion] = axis

  let leafAxis = selected[0]
  let inputStride = input.strides[leafAxis]
  let outputStride = result.strides[leafAxis]
  var leafLength = input.shape[leafAxis]
  var leafDimensions = 1
  while leafDimensions < selected.len:
    let axis = selected[leafDimensions]
    if input.strides[axis] != inputStride * leafLength or
        result.strides[axis] != outputStride * leafLength:
      break
    leafLength *= input.shape[axis]
    inc leafDimensions
  let remainingDimensions = selected.len - leafDimensions
  var spans = 1
  for index in leafDimensions ..< selected.len: spans *= input.shape[selected[index]]
  var inputBase = input.offset
  var outputBase = 0
  var groupCoordinates {.noinit.}: Metadata
  groupCoordinates.len = retained.len
  for index in 0 ..< retained.len: groupCoordinates[index] = 0

  template walkSpans(operation: untyped) =
    if remainingDimensions == 0:
      operation(inputBase, outputBase)
    elif remainingDimensions == 1:
      let axis = selected[leafDimensions]
      let inputStep = input.strides[axis]
      let outputStep = result.strides[axis]
      for row in 0 ..< input.shape[axis]:
        operation(inputBase + row * inputStep, outputBase + row * outputStep)
    else:
      var coordinates {.noinit.}: Metadata
      coordinates.len = remainingDimensions
      for index in 0 ..< remainingDimensions: coordinates[index] = 0
      var sourceBase = inputBase
      var gradientBase = outputBase
      for span in 0 ..< spans:
        operation(sourceBase, gradientBase)
        if span + 1 < spans:
          for index in 0 ..< remainingDimensions:
            let axis = selected[index + leafDimensions]
            if coordinates[index] < input.shape[axis] - 1:
              inc coordinates[index]
              sourceBase += input.strides[axis]
              gradientBase += result.strides[axis]
              break
            coordinates[index] = 0
            sourceBase -= (input.shape[axis] - 1) * input.strides[axis]
            gradientBase -= (input.shape[axis] - 1) * result.strides[axis]

  for group in 0 ..< extrema.size:
    let target = chosenValues[extrema.offset + group]
    var matches = 0'u
    template countSpan(sourceStart, gradientStart: untyped) =
      block:
        if inputStride == 0:
          matches += uint(leafLength) * uint(ord(source[sourceStart] == target))
        elif inputStride == 1:
          let values = cast[ptr UncheckedArray[T]](unsafeAddr source[sourceStart])
          for index in 0'u ..< uint(leafLength):
            matches += uint(ord(values[index] == target))
        else:
          for index in 0 ..< leafLength:
            matches += uint(ord(source[sourceStart + index * inputStride] == target))
    walkSpans(countSpan)
    let incoming = if packedSeed: upstreamValues[seed.offset + group]
                   else: stridedSeedValue(seed, group)
    let weight = incoming / T(matches)
    template writeSpan(sourceStart, gradientStart: untyped) =
      block:
        let values = cast[ptr UncheckedArray[T]](addr destination[gradientStart])
        let step = uint(outputStride)
        if inputStride == 0:
          let masked = weight * T(ord(source[sourceStart] == target))
          for index in 0'u ..< uint(leafLength): values[index * step] = masked
        elif inputStride == 1:
          let inputs = cast[ptr UncheckedArray[T]](unsafeAddr source[sourceStart])
          for index in 0'u ..< uint(leafLength):
            values[index * step] = weight * T(ord(inputs[index] == target))
        else:
          for index in 0 ..< leafLength:
            values[uint(index) * step] = weight *
              T(ord(source[sourceStart + index * inputStride] == target))
    walkSpans(writeSpan)
    if group + 1 < extrema.size:
      for index in countdown(retained.len - 1, 0):
        let axis = retained[index]
        if groupCoordinates[index] < input.shape[axis] - 1:
          inc groupCoordinates[index]
          inputBase += input.strides[axis]
          outputBase += result.strides[axis]
          break
        groupCoordinates[index] = 0
        inputBase -= (input.shape[axis] - 1) * input.strides[axis]
        outputBase -= (input.shape[axis] - 1) * result.strides[axis]

proc cpuPairedExtremaGradient*[T: SomeFloat](indices: Tensor[int64], seed: Tensor[T];
    inputShape: Metadata; axis: int): Tensor[T] =
  ## Scatter only the saved first index; inactive positions remain positive zero.
  result = newTensor[T](inputShape)
  if result.size == 0: return
  let destination = result.storage.raw_buffer
  let upstream = seed.storage.raw_buffer
  if inputShape.len == 0:
    destination[0] = upstream[seed.offset]
    return
  var inner = 1'u
  for dimension in axis + 1 ..< inputShape.len:
    inner *= uint(inputShape[dimension])
  let length = uint(inputShape[axis])
  let chosen = indices.storage.raw_buffer
  let outer = uint(indices.size) div inner
  if seed.is_C_contiguous:
    for blockIndex in 0'u ..< outer:
      let base = blockIndex * length * inner
      let seedBase = blockIndex * inner
      for column in 0'u ..< inner:
        let group = seedBase + column
        let target = base + uint(chosen[indices.offset + int(group)]) * inner + column
        destination[target] = upstream[seed.offset + int(group)]
  else:
    for blockIndex in 0'u ..< outer:
      let base = blockIndex * length * inner
      let seedBase = blockIndex * inner
      for column in 0'u ..< inner:
        let group = seedBase + column
        let target = base + uint(chosen[indices.offset + int(group)]) * inner + column
        destination[target] = stridedSeedValue(seed, int(group))
