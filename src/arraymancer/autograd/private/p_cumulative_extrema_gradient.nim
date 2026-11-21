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

## Provides component-width, ordered scatter-add VJPs for cumulative extrema.
import ../../tensor/[data_structure, init_cpu]

func flattenedSeedStride(shape, strides: Metadata, firstDimension,
    lastDimension: int, flatStride: var int): bool {.inline.} =
  ## Whether a row-major flattened range has one signed or zero stride.
  flatStride = 0
  if firstDimension > lastDimension:
    return true
  var trailing = 1
  var foundStride = false
  for dimension in countdown(lastDimension, firstDimension):
    if shape[dimension] > 1:
      if not foundStride:
        if strides[dimension] mod trailing != 0:
          return false
        flatStride = strides[dimension] div trailing
        foundStride = true
      elif strides[dimension] != flatStride * trailing:
        return false
    trailing *= shape[dimension]
  true

proc advanceSeedOffset(coordinates: var Metadata, shape, strides: Metadata,
    firstDimension, lastDimension: int, offset: var int) {.inline.} =
  ## Advance one row-major coordinate, updating its signed-stride offset.
  for dimension in countdown(lastDimension, firstDimension):
    inc coordinates[dimension]
    offset += strides[dimension]
    if coordinates[dimension] < shape[dimension]:
      break
    coordinates[dimension] = 0
    offset -= shape[dimension] * strides[dimension]

proc accumulateExtremaLane[T: SomeFloat](chosen: ptr UncheckedArray[int64],
    upstream, destination: ptr UncheckedArray[T], indicesOffset, outputBase,
    length, inner, column, seedOffset, axisStride: int) {.inline.} =
  ## Each lane's saved prefix indices are monotone, so each target is one run.
  var selected = uint(chosen[indicesOffset + outputBase + column])
  var accumulated = T(0)
  if axisStride == 0:
    let seedValue = upstream[seedOffset]
    for position in 0 ..< length:
      let flatIndex = outputBase + position * inner + column
      let nextSelected = uint(chosen[indicesOffset + flatIndex])
      if nextSelected != selected:
        destination[outputBase + int(selected) * inner + column] = accumulated
        selected = nextSelected
        accumulated = T(0)
      accumulated += seedValue
  else:
    for position in 0 ..< length:
      let flatIndex = outputBase + position * inner + column
      let nextSelected = uint(chosen[indicesOffset + flatIndex])
      if nextSelected != selected:
        destination[outputBase + int(selected) * inner + column] = accumulated
        selected = nextSelected
        accumulated = T(0)
      accumulated += upstream[seedOffset + position * axisStride]
  destination[outputBase + int(selected) * inner + column] = accumulated

proc cpuCumulativeExtremaGradient*[T: SomeFloat](indices: Tensor[int64],
    seed: Tensor[T]; inputShape: Metadata; axis: int): Tensor[T] =
  ## Saved indices are dense, input-shaped forward outputs; only seeds may stride.
  ## Every contribution adds to +0 in axis order, including scalar -0 seeds.
  result = newTensor[T](inputShape)
  if result.size == 0: return
  let destination = result.storage.raw_buffer
  let chosen = indices.storage.raw_buffer
  let upstream = seed.storage.raw_buffer
  if inputShape.len == 0:
    destination[0] += upstream[seed.offset]
    return
  var inner = 1
  for dimension in axis + 1 ..< inputShape.len:
    inner *= inputShape[dimension]
  let length = inputShape[axis]
  let outer = result.size div (length * inner)
  if seed.is_C_contiguous and inner == 1:
    # Prefix indices never decrease, so each destination is one contiguous run.
    # Keep its ordered T-width sum in a register instead of reloading a scatter.
    for blockIndex in 0 ..< outer:
      let base = blockIndex * length
      var selected = uint(chosen[indices.offset + base])
      var accumulated = T(0)
      for position in 0 ..< length:
        let flatIndex = base + position
        let nextSelected = uint(chosen[indices.offset + flatIndex])
        if nextSelected != selected:
          destination[base + int(selected)] = accumulated
          selected = nextSelected
          accumulated = T(0)
        accumulated += upstream[seed.offset + flatIndex]
      destination[base + int(selected)] = accumulated
    return

  var outerStride, innerStride: int
  let outerIsAffine = flattenedSeedStride(seed.shape, seed.strides, 0,
    axis - 1, outerStride)
  let innerIsAffine = flattenedSeedStride(seed.shape, seed.strides, axis + 1,
    inputShape.len - 1, innerStride)
  var outerCoordinates: Metadata
  var innerCoordinates: Metadata
  outerCoordinates.len = inputShape.len
  innerCoordinates.len = inputShape.len
  if not outerIsAffine:
    for dimension in 0 ..< axis:
      outerCoordinates[dimension] = 0
  if not innerIsAffine:
    for dimension in axis + 1 ..< inputShape.len:
      innerCoordinates[dimension] = 0
  var outerOffset = 0
  for blockIndex in 0 ..< outer:
    let outputBase = blockIndex * length * inner
    let blockSeedOffset =
      if outerIsAffine: seed.offset + blockIndex * outerStride
      else: seed.offset + outerOffset
    var innerOffset = 0
    for column in 0 ..< inner:
      let laneOffset =
        if innerIsAffine: column * innerStride
        else: innerOffset
      accumulateExtremaLane(chosen, upstream, destination, indices.offset,
        outputBase, length, inner, column, blockSeedOffset + laneOffset,
        seed.strides[axis])
      if not innerIsAffine and column + 1 < inner:
        advanceSeedOffset(innerCoordinates, inputShape, seed.strides,
          axis + 1, inputShape.len - 1, innerOffset)
    if not outerIsAffine and blockIndex + 1 < outer:
      advanceSeedOffset(outerCoordinates, inputShape, seed.strides, 0,
        axis - 1, outerOffset)
