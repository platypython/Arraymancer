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

## Provides shape planning and cast-free CPU extrema over tensor views.
import std/complex
import ../data_structure, ../init_cpu
import ./p_accelerate

when defined(macosx) and defined(arm64) and defined(clang):
  # Inlined strided spans must not become per-lane outlined helper calls.
  {.localPassC: "-mno-outline".}

proc normalizedExtremaAxis*(rank, axis: int): int {.inline.} =
  ## Scalars expose one virtual axis, without gaining an output dimension.
  let count = max(rank, 1)
  if axis < -count or axis >= count:
    raise newException(IndexDefect, "extrema axis out of range")
  if axis < 0: axis + count
  else: axis

proc reduceExtrema*[T: SomeNumber or bool or Complex](arg: Tensor[T];
    axes: openArray[int]; keepdim: bool; maximum: static[bool]): Tensor[T] =
  ## Plan on the stack and allocate only the final, independent output.
  let inputSize = arg.size
  if inputSize == 0:
    if axes.len == 0:
      raise newException(ValueError, "extrema empty reduction requires dimensions")
    # Empty-axis checks precede duplicate checks, in the caller's axis order.
    for axis in axes:
      let dimension = normalizedExtremaAxis(arg.rank, axis)
      if arg.shape[dimension] == 0:
        raise newException(IndexDefect, "extrema empty reduced axis")
  var selected: Metadata
  selected.len = max(arg.rank, 1)
  if axes.len == 0:
    for dimension in 0 ..< arg.rank: selected[dimension] = 1
  else:
    for axis in axes:
      let dimension = normalizedExtremaAxis(arg.rank, axis)
      if selected[dimension] != 0:
        raise newException(ValueError, "extrema duplicate reduction axis")
      selected[dimension] = 1
  var outputShape: Metadata
  var outputCount = 1
  var reducedCount = 1
  var first = arg.rank
  var last = -1
  var selectedCount = 0
  for dimension in 0 ..< arg.rank:
    if selected[dimension] != 0:
      first = min(first, dimension)
      last = dimension
      inc selectedCount
      reducedCount *= arg.shape[dimension]
      if keepdim:
        inc outputShape.len
        outputShape[outputShape.len - 1] = 1
    else:
      inc outputShape.len
      outputShape[outputShape.len - 1] = arg.shape[dimension]
      outputCount *= arg.shape[dimension]
  when T is Complex or (T is SomeUnsignedInt and sizeof(T) > 1):
    if outputCount != 0:
      raise newException(ValueError, "extrema unsupported CPU dtype")
    result = newTensorUninit[T](outputShape)
  else:
    result = newTensorUninit[T](outputShape)
    if outputCount == 0: return
    let source = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
    let destination = cast[ptr UncheckedArray[T]](result.get_offset_ptr)
    if arg.rank == 0:
      destination[0] = source[0]
      return
    if outputCount == 1 and (arg.is_C_contiguous or arg.is_F_contiguous):
      destination[0] = extremaSpan(source, 0, uint(inputSize), 1, maximum)
      return
    if selectedCount == last - first + 1 and arg.is_C_contiguous:
      var outer = 1'u
      var inner = 1'u
      for dimension in 0 ..< first: outer *= uint(arg.shape[dimension])
      for dimension in last + 1 ..< arg.rank: inner *= uint(arg.shape[dimension])
      extremaPackedAxis(cast[ptr T](destination), cast[ptr T](source),
        outer, uint(reducedCount), inner, maximum)
      return
    var reducedAxes: Metadata
    # TensorIterator consumes the smallest-stride reduction dimension first.
    for dimension in 0 ..< arg.rank:
      if selected[dimension] != 0:
        var position = reducedAxes.len
        inc reducedAxes.len
        while position > 0 and
            abs(arg.strides[reducedAxes[position - 1]]) > abs(arg.strides[dimension]):
          reducedAxes[position] = reducedAxes[position - 1]
          dec position
        reducedAxes[position] = dimension
    let leaf = reducedAxes[0]
    let leafStride = arg.strides[leaf]
    var leafLength = arg.shape[leaf]
    var leafDimensions = 1
    # Merge physically adjacent selected dimensions, including Fortran views.
    while leafDimensions < reducedAxes.len and
        arg.strides[reducedAxes[leafDimensions]] == leafStride * leafLength:
      leafLength *= arg.shape[reducedAxes[leafDimensions]]
      inc leafDimensions
    let remainingDimensions = reducedAxes.len - leafDimensions
    let chunks = reducedCount div leafLength
    var outputCoordinates: Metadata
    outputCoordinates.len = arg.rank
    var reducedCoordinates {.noinit.}: Metadata
    if remainingDimensions >= 2:
      reducedCoordinates = default(Metadata)
      reducedCoordinates.len = arg.rank
    var outputOffset = 0
    for outputIndex in 0 ..< outputCount:
      var value: T
      if remainingDimensions == 0:
        value = extremaSpan(source, outputOffset, uint(leafLength), leafStride, maximum)
      elif remainingDimensions == 1:
        let outerAxis = reducedAxes[leafDimensions]
        value = extremaTwoDimSpan(source, outputOffset, leafLength,
          arg.shape[outerAxis], leafStride, arg.strides[outerAxis], maximum)
      else:
        value = extremaIdentity(T, maximum)
        var reducedOffset = 0
        for chunk in 0 ..< chunks:
          value = extremaScalar(value, extremaSpan(source, outputOffset + reducedOffset,
            uint(leafLength), leafStride, maximum), maximum)
          when T is bool:
            if value == maximum:
              for index in leafDimensions ..< reducedAxes.len:
                reducedCoordinates[reducedAxes[index]] = 0
              break
          for index in leafDimensions ..< reducedAxes.len:
            let dimension = reducedAxes[index]
            inc reducedCoordinates[dimension]
            reducedOffset += arg.strides[dimension]
            if reducedCoordinates[dimension] < arg.shape[dimension]: break
            reducedCoordinates[dimension] = 0
            reducedOffset -= arg.shape[dimension] * arg.strides[dimension]
      destination[outputIndex] = value
      if outputIndex + 1 < outputCount:
        for dimension in countdown(arg.rank - 1, 0):
          if selected[dimension] == 0:
            inc outputCoordinates[dimension]
            outputOffset += arg.strides[dimension]
            if outputCoordinates[dimension] < arg.shape[dimension]: break
            outputCoordinates[dimension] = 0
            outputOffset -= arg.shape[dimension] * arg.strides[dimension]
