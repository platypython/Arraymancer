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

## Provides independent cumulative extrema and latest-source int64 indices.
import std/complex
import ../data_structure, ../init_cpu
import ./p_extrema
import ../../laser/compiler_optim_hints

when defined(macosx) and defined(arm64) and defined(clang):
  {.localPassC: "-mno-outline".}

proc cumulativeExtremaStep[T](value: T; index: int64; carried: var T;
    selected: var int64; maximum: static[bool]) {.inline.} =
  ## Equality selects the new source bits; every new NaN replaces the old one.
  when T is bool:
    # Keep the Boolean carry on a one-cycle AND/OR dependency, not a select.
    when maximum:
      let replaces = value or not carried
      carried = carried or value
    else:
      let replaces = not value or carried
      carried = carried and value
    if replaces: selected = index
  else:
    let replaces = when maximum: value >= carried else: value <= carried
    when T is SomeFloat:
      if replaces or value != value:
        carried = value
        selected = index
    else:
      # Eager selection exposes independent lanes to the C vectorizer.
      carried = if replaces: value else: carried
      selected = if replaces: index else: selected

func cumulativeExtremaKey[U: uint32 or uint64](bits: U; maximum: static[bool]): U {.inline.} =
  ## Both zeros meet; every NaN dominates finite keys while keeping its source bits.
  const Sign = U(1) shl (sizeof(U) * 8 - 1)
  const Infinity = when U is uint32: U(0x7f800000'u32)
    else: U(0x7ff0000000000000'u64)
  if (bits and not Sign) > Infinity:
    when maximum: high(U)
    else: U(0)
  elif (bits and Sign) != 0:
    U(0) - bits
  else:
    bits xor Sign

proc cumulativeExtremaTwoSpans[T: SomeSignedInt or uint8](
    values: ptr UncheckedArray[T]; indices: ptr UncheckedArray[int64];
    source: ptr UncheckedArray[T];
    outputBase, inputFirst, inputSecond, length, inputStride: uint;
    maximum: static[bool]) {.inline.} =
  ## Independent chains overlap comparisons without strided output scatter.
  type Carry = when sizeof(T) <= 4: int32 else: int64
  var first = Carry(source[inputFirst])
  var second = Carry(source[inputSecond])
  var firstIndex = 0'i64
  var secondIndex = 0'i64
  values[outputBase] = cast[T](first)
  values[outputBase + length] = cast[T](second)
  indices[outputBase] = 0
  indices[outputBase + length] = 0
  for index in 1'u ..< length:
    cumulativeExtremaStep(Carry(source[inputFirst + index * inputStride]),
      cast[int64](index), first, firstIndex, maximum)
    cumulativeExtremaStep(Carry(source[inputSecond + index * inputStride]),
      cast[int64](index), second, secondIndex, maximum)
    values[outputBase + index] = cast[T](first)
    values[outputBase + length + index] = cast[T](second)
    indices[outputBase + index] = firstIndex
    indices[outputBase + length + index] = secondIndex

proc cumulativeExtremaBoundedSpan[T: SomeSignedInt or uint8](
    values: ptr UncheckedArray[T]; indices: ptr UncheckedArray[int64];
    source: ptr UncheckedArray[T]; outputBase, inputBase, length: uint;
    maximum: static[bool]) {.noinline.} =
  ## A vectorizable bound reduction removes value dependencies after its first occurrence.
  var bound = source[inputBase]
  for index in 1'u ..< length:
    let value = source[inputBase + index]
    when maximum:
      if value > bound: bound = value
    else:
      if value < bound: bound = value
  var carried = source[inputBase]
  var selected = 0'i64
  values[outputBase] = carried
  indices[outputBase] = 0
  var index = 1'u
  if carried != bound:
    while index < length:
      cumulativeExtremaStep(source[inputBase + index],
        cast[int64](index), carried, selected, maximum)
      values[outputBase + index] = carried
      indices[outputBase + index] = selected
      inc index
      if carried == bound: break
    if index == length: return
  for position in index ..< length:
    values[outputBase + position] = bound
  for position in index ..< length:
    if source[inputBase + position] == bound:
      selected = cast[int64](position)
    indices[outputBase + position] = selected

proc cumulativeExtremaSpan[T](values: ptr UncheckedArray[T];
    indices: ptr UncheckedArray[int64]; source: ptr UncheckedArray[T];
    outputBase, inputBase, length, inputStride: uint;
    maximum: static[bool]) {.inline.} =
  ## Valid physical offsets stay in-bounds even when the unsigned step wraps a negative stride.
  when T is SomeSignedInt or (T is uint8 and maximum):
    if length >= 256 and inputStride == 1:
      cumulativeExtremaBoundedSpan(values, indices, source,
        outputBase, inputBase, length, maximum)
      return
  var carried = source[inputBase]
  var selected = 0'i64
  values[outputBase] = carried
  indices[outputBase] = selected
  if inputStride == 0:
    for index in 1'u ..< length:
      values[outputBase + index] = carried
      indices[outputBase + index] = cast[int64](index)
    return
  when T is SomeUnsignedInt and not maximum:
    # Zero absorbs unsigned minima; only later zero indices still change.
    var index = 1'u
    if carried != T(0):
      while index < length:
        cumulativeExtremaStep(source[inputBase + index * inputStride],
          cast[int64](index), carried, selected, maximum)
        values[outputBase + index] = carried
        indices[outputBase + index] = selected
        inc index
        if carried == T(0): break
      if index == length: return
    for position in index ..< length:
      values[outputBase + position] = T(0)
    for position in index ..< length:
      if source[inputBase + position * inputStride] == T(0):
        selected = cast[int64](position)
      indices[outputBase + position] = selected
    return
  elif T is bool:
    # After the absorbing value, only the latest matching index still changes.
    var index = 1'u
    if carried != maximum:
      while index < length:
        let value = source[inputBase + index * inputStride]
        values[outputBase + index] = value
        indices[outputBase + index] = cast[int64](index)
        inc index
        if value == maximum:
          selected = cast[int64](index - 1)
          break
      if index == length: return
    for position in index ..< length:
      values[outputBase + position] = maximum
    for position in index ..< length:
      if source[inputBase + position * inputStride] == maximum:
        selected = cast[int64](position)
      indices[outputBase + position] = selected
  elif T is SomeFloat:
    type Bits = when T is float32: uint32 else: uint64
    var carriedBits = cast[Bits](carried)
    var carriedKey = cumulativeExtremaKey(carriedBits, maximum)
    for index in 1'u ..< length:
      let bits = cast[Bits](source[inputBase + index * inputStride])
      let key = cumulativeExtremaKey(bits, maximum)
      let replaces = when maximum: key >= carriedKey else: key <= carriedKey
      if replaces:
        carriedBits = bits
        carriedKey = key
        selected = cast[int64](index)
      values[outputBase + index] = cast[T](carriedBits)
      indices[outputBase + index] = selected
  else:
    # Keep narrow integer carries in full registers between comparisons.
    type Carry = when sizeof(T) <= 4: int32 else: int64
    var wideCarry = Carry(carried)
    when sizeof(T) <= 4:
      for index in 1'u ..< length:
        cumulativeExtremaStep(Carry(source[inputBase + index * inputStride]),
          cast[int64](index), wideCarry, selected, maximum)
        values[outputBase + index] = cast[T](wideCarry)
        indices[outputBase + index] = selected
    else:
      var index = 1'u
      # Independent two-value prefixes halve the carry dependency across pairs.
      while index + 1 < length:
        let left = Carry(source[inputBase + index * inputStride])
        let right = Carry(source[inputBase + (index + 1) * inputStride])
        var pairValue = left
        var pairIndex = cast[int64](index)
        cumulativeExtremaStep(right, cast[int64](index + 1),
          pairValue, pairIndex, maximum)
        let replacesLeft = when maximum: left >= wideCarry else: left <= wideCarry
        values[outputBase + index] = cast[T](if replacesLeft: left else: wideCarry)
        indices[outputBase + index] = if replacesLeft: cast[int64](index) else: selected
        cumulativeExtremaStep(pairValue, pairIndex, wideCarry, selected, maximum)
        values[outputBase + index + 1] = cast[T](wideCarry)
        indices[outputBase + index + 1] = selected
        index += 2
      if index < length:
        cumulativeExtremaStep(Carry(source[inputBase + index * inputStride]),
          cast[int64](index), wideCarry, selected, maximum)
        values[outputBase + index] = cast[T](wideCarry)
        indices[outputBase + index] = selected

proc cumulativeExtremaPacked[T](values: ptr UncheckedArray[T];
    indices: ptr UncheckedArray[int64]; source: ptr UncheckedArray[T];
    inputOffset, groups, length, inner: uint; maximum: static[bool]) =
  ## Dense columns keep contiguous lane addresses visible to the C vectorizer.
  if inner == 1:
    var group = 0'u
    when T is SomeSignedInt or T is uint8:
      while group + 1 < groups:
        let base = group * length
        cumulativeExtremaTwoSpans(values, indices, source, base,
          inputOffset + base, inputOffset + base + length, length, 1'u, maximum)
        group += 2
    while group < groups:
      let base = group * length
      cumulativeExtremaSpan(values, indices, source, base,
        inputOffset + base, length, 1'u, maximum)
      inc group
    return
  const Lanes = 32'u
  withCompilerOptimHints()
  # Allocating entry points own disjoint value/index buffers and never mutate input.
  let inputData {.restrict.} = source
  let valueOut {.restrict.} = values
  let indexOut {.restrict.} = indices
  var carriedStorage {.noinit.}: array[Lanes, T]
  var selectedStorage {.noinit.}: array[Lanes, int64]
  # Width is capped at Lanes; checked lane accesses inhibit vectorization.
  let carried {.restrict.} = cast[ptr UncheckedArray[T]](addr carriedStorage)
  let selected {.restrict.} = cast[ptr UncheckedArray[int64]](addr selectedStorage)
  for outer in 0'u ..< groups div inner:
    var column = 0'u
    while column < inner:
      let width = min(Lanes, inner - column)
      let base = outer * length * inner + column
      for lane in 0'u ..< width:
        carried[lane] = inputData[inputOffset + base + lane]
        selected[lane] = 0
        valueOut[base + lane] = carried[lane]
        indexOut[base + lane] = 0
      for index in 1'u ..< length:
        let position = base + index * inner
        for lane in 0'u ..< width:
          cumulativeExtremaStep(inputData[inputOffset + position + lane],
            cast[int64](index), carried[lane], selected[lane], maximum)
          valueOut[position + lane] = carried[lane]
          indexOut[position + lane] = selected[lane]
      column += width

proc scanCumulativeExtrema*[T: SomeNumber or bool or Complex](arg: Tensor[T];
    axis: int; maximum: static[bool]): tuple[values: Tensor[T], indices: Tensor[int64]] =
  ## Scalar and empty dispatch precedes dtype rejection, but never axis validation.
  let dimension = normalizedExtremaAxis(arg.rank, axis)
  if arg.rank == 0 or arg.size == 0:
    result.values = newTensorUninit[T](arg.shape)
    result.indices = newTensorUninit[int64](arg.shape)
    if arg.rank == 0:
      result.values.storage.raw_buffer[0] = arg.storage.raw_buffer[arg.offset]
      result.indices.storage.raw_buffer[0] = 0
    return
  when T is Complex or (T is SomeUnsignedInt and sizeof(T) > 1):
    raise newException(ValueError, "cumulative extrema unsupported CPU dtype")
  else:
    result.values = newTensorUninit[T](arg.shape)
    result.indices = newTensorUninit[int64](arg.shape)
    let source = arg.storage.raw_buffer
    let values = result.values.storage.raw_buffer
    let indices = result.indices.storage.raw_buffer
    let length = uint(arg.shape[dimension])
    var inner = 1'u
    for index in dimension + 1 ..< arg.rank: inner *= uint(arg.shape[index])
    let groups = uint(arg.size) div length
    let packed = arg.is_C_contiguous
    if packed and length == 1:
      copyMem(values, addr source[arg.offset], arg.size * sizeof(T))
      zeroMem(indices, arg.size * sizeof(int64))
      return
    if packed:
      cumulativeExtremaPacked(values, indices, source, uint(arg.offset),
        groups, length, inner, maximum)
      return

    var coordinates: Metadata
    coordinates.len = arg.rank
    var inputBase = arg.offset
    let inputStride = arg.strides[dimension]
    let inputStep = cast[uint](inputStride)
    if inner == 1:
      template advanceBase() =
        for coordinateAxis in countdown(arg.rank - 1, 0):
          if coordinateAxis == dimension: continue
          inc coordinates[coordinateAxis]
          inputBase += arg.strides[coordinateAxis]
          if coordinates[coordinateAxis] < arg.shape[coordinateAxis]: break
          coordinates[coordinateAxis] = 0
          inputBase -= arg.shape[coordinateAxis] * arg.strides[coordinateAxis]
      var group = 0'u
      when T is SomeSignedInt or T is uint8:
        if inputStride < -1 or inputStride > 1:
          while group + 1 < groups:
            let firstBase = uint(inputBase)
            advanceBase()
            cumulativeExtremaTwoSpans(values, indices, source, group * length,
              firstBase, uint(inputBase), length, inputStep, maximum)
            group += 2
            if group < groups: advanceBase()
      while group < groups:
        cumulativeExtremaSpan(values, indices, source, group * length,
          uint(inputBase), length, inputStep, maximum)
        inc group
        if group < groups: advanceBase()
      return
    const Lanes = 32'u
    var inputBases {.noinit.}: array[Lanes, uint]
    var outputBases {.noinit.}: array[Lanes, uint]
    var carried {.noinit.}: array[Lanes, T]
    var selected {.noinit.}: array[Lanes, int64]
    var groupStart = 0'u
    while groupStart < groups:
      let width = min(Lanes, groups - groupStart)
      for lane in 0'u ..< width:
        let group = groupStart + lane
        let base = (group div inner) * length * inner + group mod inner
        outputBases[lane] = base
        inputBases[lane] = uint(inputBase)
        if group + 1 < groups:
          for index in countdown(arg.rank - 1, 0):
            if index == dimension: continue
            inc coordinates[index]
            inputBase += arg.strides[index]
            if coordinates[index] < arg.shape[index]: break
            coordinates[index] = 0
            inputBase -= arg.shape[index] * arg.strides[index]
        carried[lane] = source[inputBases[lane]]
        selected[lane] = 0
        values[base] = carried[lane]
        indices[base] = 0
      if inputStride == 0:
        # Broadcast prefixes always select their latest logically distinct index.
        for index in 1'u ..< length:
          for lane in 0'u ..< width:
            let position = outputBases[lane] + index * inner
            values[position] = carried[lane]
            indices[position] = cast[int64](index)
      else:
        for index in 1'u ..< length:
          for lane in 0'u ..< width:
            let position = outputBases[lane] + index * inner
            cumulativeExtremaStep(source[inputBases[lane] + index * inputStep],
              cast[int64](index), carried[lane], selected[lane], maximum)
            values[position] = carried[lane]
            indices[position] = selected[lane]
      groupStart += width
