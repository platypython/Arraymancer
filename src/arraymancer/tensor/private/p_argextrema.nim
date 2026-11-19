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

## Provides first-index CPU extrema with optional independent paired values.
import std/complex
import ../data_structure, ../init_cpu
import ./p_accelerate, ./p_extrema
import ../../laser/compiler_optim_hints

when defined(macosx) and defined(arm64) and defined(clang):
  {.localPassC: "-mno-outline".}

proc replacesExtremum[T](best, candidate: T; maximum: static[bool]): bool {.inline.} =
  ## Equal values and later NaNs cannot replace the first logical occurrence.
  when T is SomeFloat:
    if best != best: return false
    if candidate != candidate: return true
  when maximum: candidate > best
  else: candidate < best

proc argContiguousSpan[T; I: SomeInteger](source: ptr UncheckedArray[T];
    base: I; length: uint; maximum: static[bool]): int64 {.inline.} =
  ## Vector-fold the value, then locate its first occurrence without scratch storage.
  when T is bool:
    for index in 0'u ..< length:
      if source[base + I(index)] == maximum: return int64(index)
    return 0
  else:
    let best = extremaSpan(source, base, length, I(1), maximum)
    when T is SomeFloat:
      if best != best:
        for index in 0'u ..< length - 1'u:
          let candidate = source[base + I(index)]
          if candidate != candidate: return int64(index)
        return int64(length - 1'u)
    for index in 0'u ..< length - 1'u:
      if source[base + I(index)] == best: return int64(index)
    # The folded value occurs in this nonempty span, so the last index must match.
    result = int64(length - 1'u)

proc argStridedSpan[T](source: ptr UncheckedArray[T]; base, length, stride: int;
    maximum: static[bool]; paired: static[bool] = false;
    selected: ptr T = nil): int64 {.inline.} =
  ## Keep logical index order and retain the selected source bits for paired outputs.
  if stride == 0:
    when paired: selected[] = source[base]
    return 0
  if stride == 1:
    result = argContiguousSpan(source, base, uint(length), maximum)
    when paired: selected[] = source[base + int(result)]
    return
  when T is bool:
    var offset = base
    for index in 0 ..< length:
      if source[offset] == maximum:
        when paired: selected[] = maximum
        return int64(index)
      offset += stride
    when paired: selected[] = not maximum
    return 0
  else:
    var best = source[base]
    var offset = base + stride
    for index in 1 ..< length:
      let candidate = source[offset]
      if replacesExtremum(best, candidate, maximum):
        best = candidate
        result = int64(index)
      offset += stride
    when paired: selected[] = best

proc argPackedAxis[T](output: ptr int64; input: ptr T;
    outer, length, inner: uint; maximum: static[bool];
    paired: static[bool] = false; values: ptr T = nil) =
  ## Column tiles retain first-source values and allocate no intermediate tensors.
  withCompilerOptimHints()
  let source = cast[ptr UncheckedArray[T]](input)
  let destination {.restrict.} = cast[ptr UncheckedArray[int64]](output)
  when paired:
    let valueDestination {.restrict.} = cast[ptr UncheckedArray[T]](values)
  for group in 0'u ..< outer:
    let base = group * length * inner
    let target = group * inner
    if inner == 1:
      destination[target] = argContiguousSpan(source, base, length, maximum)
      when paired:
        valueDestination[target] = source[base + uint(destination[target])]
    else:
      when T is bool:
        for column in 0'u ..< inner:
          when paired:
            destination[target + column] = argStridedSpan(source,
              int(base + column), int(length), int(inner), maximum, true,
              addr valueDestination[target + column])
          else:
            destination[target + column] = argStridedSpan(source,
              int(base + column), int(length), int(inner), maximum)
      else:
        when sizeof(T) == 8:
          type LaneIndex = uint64
        else:
          type LaneIndex = uint32
        if uint64(length) > uint64(high(LaneIndex)):
          for column in 0'u ..< inner:
            when paired:
              destination[target + column] = argStridedSpan(source,
                int(base + column), int(length), int(inner), maximum, true,
                addr valueDestination[target + column])
            else:
              destination[target + column] = argStridedSpan(source,
                int(base + column), int(length), int(inner), maximum)
          continue
        template reduceColumns(lanes: static[int]) =
          block:
            var best {.noinit.}: array[lanes, T]
            var bestIndices: array[lanes, LaneIndex]
            for lane in 0 ..< lanes:
              best[lane] = source[base + column + uint(lane)]
            for row in 1'u ..< length:
              let rowBase = base + row * inner + column
              for lane in 0 ..< lanes:
                let candidate = source[rowBase + uint(lane)]
                when T is SomeFloat:
                  # Unordered candidates win once; equal values retain source bits.
                  when maximum:
                    let replace = uint8(best[lane] == best[lane]) and
                      uint8(not (candidate <= best[lane]))
                  else:
                    let replace = uint8(best[lane] == best[lane]) and
                      uint8(not (candidate >= best[lane]))
                  best[lane] = if replace != 0: candidate else: best[lane]
                  bestIndices[lane] = if replace != 0: LaneIndex(row) else: bestIndices[lane]
                else:
                  if replacesExtremum(best[lane], candidate, maximum):
                    best[lane] = candidate
                    bestIndices[lane] = LaneIndex(row)
            for lane in 0 ..< lanes:
              destination[target + column + uint(lane)] = int64(bestIndices[lane])
              when paired:
                valueDestination[target + column + uint(lane)] = best[lane]
        var column = 0'u
        while column + 32'u <= inner:
          reduceColumns(32)
          column += 32'u
        while column < inner:
          reduceColumns(1)
          inc column

proc argLogicalWhole[T](source: ptr UncheckedArray[T]; shape, strides: Metadata;
                        length: int; maximum: static[bool]): int64 =
  ## Flatten logical coordinates, not physical storage, for transposes and broadcasts.
  var coordinates: Metadata
  coordinates.len = shape.len
  var offset = 0
  var best = source[0]
  for index in 1 ..< length:
    var dimension = shape.len - 1
    while dimension >= 0:
      inc coordinates[dimension]
      if coordinates[dimension] < shape[dimension]:
        offset += strides[dimension]
        break
      offset -= (shape[dimension] - 1) * strides[dimension]
      coordinates[dimension] = 0
      dec dimension
    let candidate = source[offset]
    if replacesExtremum(best, candidate, maximum):
      best = candidate
      result = int64(index)

proc argLogicalAxis[T](output: ptr UncheckedArray[int64];
    source: ptr UncheckedArray[T]; shape, strides: Metadata;
    axis, length, outputCount: int; maximum: static[bool];
    paired: static[bool] = false; values: ptr T = nil) =
  ## Carry retained coordinates once per output; paired values reuse the selected bits.
  var retainedShape, retainedStrides, coordinates: Metadata
  for dimension in 0 ..< shape.len:
    if dimension != axis:
      retainedShape.add shape[dimension]
      retainedStrides.add strides[dimension]
  coordinates.len = retainedShape.len
  var base = 0
  when paired:
    let valueDestination = cast[ptr UncheckedArray[T]](values)
  for group in 0 ..< outputCount:
    when paired:
      output[group] = argStridedSpan(source, base, length, strides[axis],
        maximum, true, addr valueDestination[group])
    else:
      output[group] = argStridedSpan(source, base, length, strides[axis], maximum)
    var dimension = retainedShape.len - 1
    while dimension >= 0:
      inc coordinates[dimension]
      if coordinates[dimension] < retainedShape[dimension]:
        base += retainedStrides[dimension]
        break
      base -= (retainedShape[dimension] - 1) * retainedStrides[dimension]
      coordinates[dimension] = 0
      dec dimension

proc reduceArgExtrema*[T: SomeNumber or bool or Complex](arg: Tensor[T];
    axis: int; whole: static[bool]; keepdim: bool;
    maximum: static[bool]): Tensor[int64] =
  ## Allocate only independent int64 indices after dtype and dimension validation.
  when T is bool or T is Complex:
    raise newException(ValueError, "indexed extrema unsupported CPU dtype")
  else:
    let inputSize = arg.size
    var dimension = 0
    var length = inputSize
    when whole:
      if inputSize == 0:
        raise newException(IndexDefect, "indexed extrema empty reduction requires an axis")
    else:
      dimension = normalizedExtremaAxis(arg.rank, axis)
      length = if arg.rank == 0: 1 else: arg.shape[dimension]
      if length == 0:
        raise newException(IndexDefect, "indexed extrema empty reduced axis")

    var outputShape: Metadata
    var outputCount = 1
    for current in 0 ..< arg.rank:
      when whole:
        if keepdim: outputShape.add 1
      else:
        if current == dimension:
          if keepdim: outputShape.add 1
        else:
          outputShape.add arg.shape[current]
          outputCount *= arg.shape[current]

    when T is SomeUnsignedInt and sizeof(T) > 1:
      # Dispatch is skipped for positive-rank explicit singleton axes and empty outputs.
      when whole:
        raise newException(ValueError, "indexed extrema unsupported CPU dtype")
      else:
        if outputCount != 0 and (arg.rank == 0 or length != 1):
          raise newException(ValueError, "indexed extrema unsupported CPU dtype")
        result = newTensorUninit[int64](outputShape)
        if outputCount == 0: return
        let destination = cast[ptr UncheckedArray[int64]](result.get_offset_ptr)
        for index in 0 ..< outputCount: destination[index] = 0
    else:
      result = newTensorUninit[int64](outputShape)
      if outputCount == 0: return
      let destination = cast[ptr UncheckedArray[int64]](result.get_offset_ptr)
      if inputSize == 1 or length == 1:
        for index in 0 ..< outputCount: destination[index] = 0
        return
      let source = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
      when whole:
        if arg.is_C_contiguous:
          destination[0] = argContiguousSpan(source, 0'u, uint(inputSize), maximum)
        else:
          destination[0] = argLogicalWhole(source, arg.shape, arg.strides, inputSize, maximum)
      else:
        if arg.strides[dimension] == 0:
          for index in 0 ..< outputCount: destination[index] = 0
        elif arg.is_C_contiguous:
          var outer = 1'u
          var inner = 1'u
          for current in 0 ..< dimension: outer *= uint(arg.shape[current])
          for current in dimension + 1 ..< arg.rank: inner *= uint(arg.shape[current])
          argPackedAxis(cast[ptr int64](destination), cast[ptr T](source),
            outer, uint(length), inner, maximum)
        else:
          argLogicalAxis(destination, source, arg.shape, arg.strides,
            dimension, length, outputCount, maximum)

proc reducePairedExtrema*[T: SomeNumber or bool or Complex](arg: Tensor[T];
    axis: int; keepdim: bool;
    maximum: static[bool]): tuple[values: Tensor[T], indices: Tensor[int64]] =
  ## Validate before allocation, then fill only the two independent final outputs.
  let dimension = normalizedExtremaAxis(arg.rank, axis)
  let length = if arg.rank == 0: 1 else: arg.shape[dimension]
  if length == 0:
    raise newException(IndexDefect, "paired extrema empty reduced axis")
  when T is Complex:
    raise newException(ValueError, "paired extrema unsupported CPU dtype")
  else:
    var outputShape: Metadata
    var outputCount = 1
    for current in 0 ..< arg.rank:
      if current == dimension:
        if keepdim: outputShape.add 1
      else:
        outputShape.add arg.shape[current]
        outputCount *= arg.shape[current]
    when T is SomeUnsignedInt and sizeof(T) > 1:
      # Paired reducers skip dispatch for scalars and empties, not singleton axes.
      if arg.rank > 0 and outputCount != 0:
        raise newException(ValueError, "paired extrema unsupported CPU dtype")
    result.values = newTensorUninit[T](outputShape)
    result.indices = newTensorUninit[int64](outputShape)
    if outputCount == 0: return
    let source = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
    let values = cast[ptr UncheckedArray[T]](result.values.get_offset_ptr)
    let indices = cast[ptr UncheckedArray[int64]](result.indices.get_offset_ptr)
    if arg.rank == 0:
      values[0] = source[0]
      indices[0] = 0
      return
    when not (T is SomeUnsignedInt and sizeof(T) > 1):
      if arg.is_C_contiguous:
        var outer = 1'u
        var inner = 1'u
        for current in 0 ..< dimension: outer *= uint(arg.shape[current])
        for current in dimension + 1 ..< arg.rank: inner *= uint(arg.shape[current])
        argPackedAxis(cast[ptr int64](indices), cast[ptr T](source),
          outer, uint(length), inner, maximum, true, cast[ptr T](values))
      else:
        argLogicalAxis(indices, source, arg.shape, arg.strides,
          dimension, length, outputCount, maximum, true, cast[ptr T](values))
