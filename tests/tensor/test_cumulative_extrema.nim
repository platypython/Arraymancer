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

## Exercises latest-prefix source bits, independent outputs and exceptional dispatch.
import ../../src/arraymancer
import std/[unittest, complex, strutils]

proc releaseExternalOutput(memalloc: pointer) {.noconv, gcsafe, raises: [].} =
  ## Lifetime ownership does not imply a known writable capacity.
  deallocShared(memalloc)

proc checkNumericPrefixes[T: SomeNumber]() =
  ## Latest ties differ from the first-tie ordinary indexed extrema contract.
  let source = [[T(3), T(1), T(1), T(2)], [T(2), T(3), T(3), T(1)]].toTensor
  let minima: tuple[values: Tensor[T], indices: Tensor[int64]] = source.cummin(1)
  check minima.values == [[T(3), T(1), T(1), T(1)], [T(2), T(2), T(2), T(1)]].toTensor
  check minima.indices == [[0'i64, 1, 2, 2], [0'i64, 0, 0, 3]].toTensor
  let maxima = source.cummax(-1)
  check maxima.values == [[T(3), T(3), T(3), T(3)], [T(2), T(3), T(3), T(3)]].toTensor
  check maxima.indices == [[0'i64, 0, 0, 0], [0'i64, 1, 2, 2]].toTensor
  let columns = source.cummax()
  check columns.values == [[T(3), T(1), T(1), T(2)], [T(3), T(3), T(3), T(2)]].toTensor
  check columns.indices == [[0'i64, 0, 0, 0], [0'i64, 1, 1, 0]].toTensor

  var outputValues = newTensorUninit[T](source.shape)
  var outputIndices = newTensorUninit[int64](source.shape)
  source.cummin(1, outputValues, outputIndices)
  check outputValues == minima.values
  check outputIndices == minima.indices
  source.cummax(-1, outputValues, outputIndices)
  check outputValues == maxima.values
  check outputIndices == maxima.indices
  let singleton = source.reshape(1, 8).cummin(0)
  check singleton.values == source.reshape(1, 8)
  check singleton.indices == newTensor[int64](1, 8)
  when T is SomeInteger:
    let limits = [low(T), high(T), T(0), low(T), high(T)].toTensor
    let boundaryMin = limits.cummin()
    let boundaryMax = limits.cummax()
    check boundaryMin.values == newTensorWith[T]([5], low(T))
    when T is SomeUnsignedInt:
      check boundaryMin.indices == [0'i64, 0, 2, 3, 3].toTensor
      let positive = [T(7), T(3), T(3), T(9), T(2), T(2)].toTensor.cummin()
      check positive.values == [T(7), T(3), T(3), T(3), T(2), T(2)].toTensor
      check positive.indices == [0'i64, 1, 2, 2, 4, 5].toTensor
    else:
      check boundaryMin.indices == [0'i64, 0, 0, 3, 3].toTensor
    check boundaryMax.values == [low(T), high(T), high(T), high(T), high(T)].toTensor
    check boundaryMax.indices == [0'i64, 1, 1, 1, 4].toTensor
    const floor = when T is SomeSignedInt: -8 else: 0
    var long = newTensorUninit[T](257)
    for index in 0 ..< long.size:
      long[index] = T(index mod 17 + floor)
    let longMin = long.cummin()
    let longMax = long.cummax()
    let reverseMin = long[256..0|-1].cummin()
    let reverseMax = long[256..0|-1].cummax()
    for index in 0 ..< long.size:
      check longMin.values[index] == T(floor)
      check longMin.indices[index] == int64(index - index mod 17)
      check longMax.values[index] == T(min(index, 16) + floor)
      let selected = if index < 16: index else: index - (index - 16) mod 17
      check longMax.indices[index] == int64(selected)
      check reverseMin.values[index] == T(floor + (if index == 0: 1 else: 0))
      let minIndex = if index == 0: 0 else: index - (index - 1) mod 17
      check reverseMin.indices[index] == int64(minIndex)
      check reverseMax.values[index] == T(floor + (if index < 2: 1 else: 16))
      let maxIndex = if index < 2: 0 else: index - (index - 2) mod 17
      check reverseMax.indices[index] == int64(maxIndex)
    var late = newTensorWith[T]([257], T(1))
    late[256] = low(T)
    let lateMin = late.cummin()
    late[256] = high(T)
    let lateMax = late.cummax()
    for index in 0 ..< late.size:
      check lateMin.values[index] == (if index == 256: low(T) else: T(1))
      check lateMax.values[index] == (if index == 256: high(T) else: T(1))
      check lateMin.indices[index] == int64(index)
      check lateMax.indices[index] == int64(index)

proc checkLatestFloatingBits[T: SomeFloat]() =
  ## A thirty-third chain exercises the last lane of packed and strided scans.
  when T is float32:
    type U = uint32
    let firstNaN = cast[T](0x7fc12345'u32)
    let laterNaN = cast[T](0x7fc54321'u32)
  else:
    type U = uint64
    let firstNaN = cast[T](0x7ff8123456789abc'u64)
    let laterNaN = cast[T](0x7ff876543210abcd'u64)
  var source = newTensor[T](4, 33)
  for column in 0 ..< 33:
    source[0, column] = T(-0.0)
    source[2, column] = T(-0.0)
  source[1, 32] = firstNaN
  source[3, 32] = laterNaN
  for maximum in [false, true]:
    let packed = if maximum: source.cummax(0) else: source.cummin(0)
    let strided = if maximum: source.transpose.cummax(1) else: source.transpose.cummin(1)
    for row in 0 ..< 4:
      for column in 0 ..< 33:
        let expected = if column == 32 and row >= 1:
          (if row == 3: laterNaN else: firstNaN)
        else: source[row, column]
        let expectedIndex = if column == 32 and row == 2: 1'i64 else: int64(row)
        check cast[U](packed.values[row, column]) == cast[U](expected)
        check cast[U](strided.values[column, row]) == cast[U](expected)
        check packed.indices[row, column] == expectedIndex
        check strided.indices[column, row] == expectedIndex
  let vector = [T(-0.0), T(0.0), T(-0.0), T(0.0)].toTensor.cummax()
  for index in 0 ..< 4:
    check cast[U](vector.values[index]) == cast[U](source[index, 0])
    check vector.indices[index] == int64(index)
  source[0, 32] = firstNaN
  let broadcast = source[0..0, _].broadcast(4, 33).cummin(0)
  for row in 0 ..< 4:
    for column in 0 ..< 33:
      check cast[U](broadcast.values[row, column]) == cast[U](source[0, column])
      check broadcast.indices[row, column] == int64(row)

proc checkIEEEPrefixes[T: SomeFloat]() =
  ## Ordered endpoints pin the winning indices independently of the scan implementation.
  when T is float32:
    type U = uint32
    let words = [0xff800000'u32, 0xff7fffff'u32, 0x80800000'u32, 0x80000001'u32,
      0x80000000'u32, 0x00000000'u32, 0x00000001'u32, 0x00800000'u32, 0x7f7fffff'u32,
      0x7f800000'u32, 0x7fc00011'u32, 0xffc00022'u32]
  else:
    type U = uint64
    let words = [0xfff0000000000000'u64, 0xffefffffffffffff'u64, 0x8010000000000000'u64,
      0x8000000000000001'u64, 0x8000000000000000'u64, 0x0000000000000000'u64,
      0x0000000000000001'u64, 0x0010000000000000'u64, 0x7fefffffffffffff'u64,
      0x7ff0000000000000'u64, 0x7ff8000000000011'u64, 0xfff8000000000022'u64]
  var source = newTensorUninit[T](words.len)
  for index, word in words: source[int(index)] = cast[T](word)
  let candidates = [source[0..9], source[9..0|-1], source]
  for variant, candidate in candidates:
    for maximum in [false, true]:
      let pair = if maximum: candidate.cummax() else: candidate.cummin()
      for index in 0 ..< candidate.size:
        let selected = if variant == 1:
          (if maximum: 0 else: index)
        elif maximum or index >= 10: index
        else: 0
        check pair.indices[index] == int64(selected)
        check cast[U](pair.values[index]) == cast[U](candidate[selected])

proc checkExceptionalPrefixes[T](value: T) =
  ## Scalar/empty shortcuts must not enable unsupported positive-rank singletons.
  let owner = [value, value, value].toTensor
  let scalar = owner[1..1].reshape(newSeq[int]())
  for axis in [-1, 0]:
    let pair = scalar.cummax(axis)
    check pair.values.shape.len == 0
    check pair.indices.shape.len == 0
    check pair.values.atContiguousIndex(0) == value
    check pair.indices.atContiguousIndex(0) == 0'i64
  for axis in [-3, -2, -1, 0, 1, 2]:
    let pair = newTensor[T](2, 0, 3).cummin(axis)
    check pair.values.shape == [2, 0, 3]
    check pair.indices.shape == [2, 0, 3]
    check pair.values.size == 0
    check pair.indices.size == 0
  var scalarValues = newTensorUninit[T]([2])
  var scalarIndices = newTensorUninit[int64]([2])
  scalar.cummax(-1, scalarValues, scalarIndices)
  check scalarValues.shape.len == 0
  check scalarIndices.shape.len == 0
  check scalarValues.atContiguousIndex(0) == value
  check scalarIndices.atContiguousIndex(0) == 0'i64

  let emptySource = newTensor[T](2, 0, 3)
  var emptyValues = newTensorUninit[T]([1])
  var emptyIndices = newTensorUninit[int64]([1])
  emptySource.cummin(1, emptyValues, emptyIndices)
  check emptyValues.shape == [2, 0, 3]
  check emptyIndices.shape == [2, 0, 3]
  check emptyValues.size == 0
  check emptyIndices.size == 0
  expect ValueError: discard [value].toTensor.cummin(0)
  expect ValueError: discard newTensorWith[T]([2, 1, 3], value).cummax(1)
  expect IndexDefect: discard scalar.cummin(-2)
  expect IndexDefect: discard scalar.cummax(1)
  expect IndexDefect: discard [value].toTensor.cummax(1)
  expect IndexDefect: discard newTensor[T](2, 0, 3).cummin(3)

suite "CPU cumulative min/max extrema":
  test "Numeric values retain dtype and every prefix keeps its latest winning index":
    checkNumericPrefixes[float32]()
    checkNumericPrefixes[float64]()
    checkNumericPrefixes[int8]()
    checkNumericPrefixes[int16]()
    checkNumericPrefixes[int32]()
    checkNumericPrefixes[int64]()
    checkNumericPrefixes[uint8]()

  test "Boolean prefixes select the latest equal truth value":
    let source = [[true, false, true], [false, true, true]].toTensor
    let maxima = source.cummax(1)
    let minima = source.cummin(1)
    check maxima.values == [[true, true, true], [false, true, true]].toTensor
    check maxima.indices == [[0'i64, 0, 2], [0'i64, 1, 2]].toTensor
    check minima.values == [[true, false, false], [false, false, false]].toTensor
    check minima.indices == [[0'i64, 1, 1], [0'i64, 0, 0]].toTensor
    var outputValues = newTensorUninit[bool](source.shape)
    var outputIndices = newTensorUninit[int64](source.shape)
    source.cummin(1, outputValues, outputIndices)
    check outputValues == minima.values
    check outputIndices == minima.indices
    source.cummax(1, outputValues, outputIndices)
    check outputValues == maxima.values
    check outputIndices == maxima.indices
    let empty = newTensor[bool](0, 3).cummax(0)
    check empty.values.shape == [0, 3]
    check empty.indices.shape == [0, 3]

  test "Float32 packed, signed-zero, NaN payload and partial-lane rules":
    checkLatestFloatingBits[float32]()

  test "Float64 packed, signed-zero, NaN payload and partial-lane rules":
    checkLatestFloatingBits[float64]()

  test "Ordered IEEE endpoints preserve latest zero and NaN source bits":
    checkIEEEPrefixes[float32]()
    checkIEEEPrefixes[float64]()

  test "Offset, signed and stepped input views keep logical prefix indices":
    let source = [[9, 1, 9], [0, 8, 8]].toTensor
    let reversed = source[1..0|-1, 2..0|-1].cummax(1)
    check reversed.values == [[8, 8, 8], [9, 9, 9]].toTensor
    check reversed.indices == [[0'i64, 1, 1], [0'i64, 0, 2]].toTensor
    let offset = source.transpose()[_, 1..^1].cummax(0)
    check offset.values == [[0], [8], [8]].toTensor
    check offset.indices == [[0'i64], [1'i64], [2'i64]].toTensor
    let wide = [[9, 1, 7, 0, 5, -1], [0, 8, 2, 8, 4, 8],
      [-2, -3, -4, -5, -6, -7]].toTensor
    let stepped = wide[_, 0..4|2].cummin(1)
    check stepped.values == [[9, 7, 5], [0, 0, 0], [-2, -4, -6]].toTensor
    check stepped.indices == [[0'i64, 1, 2], [0'i64, 0, 0], [0'i64, 1, 2]].toTensor
    let reverseStep = wide[2..0|-1, 4..0|-2].cummax(-1)
    check reverseStep.values == [[-6, -4, -2], [4, 4, 4], [5, 7, 9]].toTensor
    check reverseStep.indices == [[0'i64, 1, 2], [0'i64, 0, 0], [0'i64, 1, 2]].toTensor
    let wrapped = wide.reshape(3, 2, 3).permute(1, 2, 0).cummin(-1)
    check wrapped.values == [[[9, 0, -2], [1, 1, -3], [7, 2, -4]],
      [[0, 0, -5], [5, 4, -6], [-1, -1, -7]]].toTensor
    check wrapped.indices == [[[0'i64, 1, 2], [0'i64, 0, 2], [0'i64, 1, 2]],
      [[0'i64, 0, 2], [0'i64, 1, 2], [0'i64, 0, 2]]].toTensor

  test "Equal-width value/index outputs and repeated calls own independent storage":
    let source = [9'i64, 2, 9].toTensor
    var pair = source.cummax()
    let repeated = source.cummax()
    check pair.values.get_offset_ptr != pair.indices.get_offset_ptr
    check pair.values.get_offset_ptr != source.get_offset_ptr
    check pair.indices.get_offset_ptr != source.get_offset_ptr
    check pair.values.get_offset_ptr != repeated.values.get_offset_ptr
    check pair.indices.get_offset_ptr != repeated.indices.get_offset_ptr
    pair.values[0] = -77
    pair.indices[2] = 55
    check source == [9'i64, 2, 9].toTensor
    check repeated.values == [9'i64, 9, 9].toTensor
    check repeated.indices == [0'i64, 0, 2].toTensor
    check pair.values[2] == 9'i64
    check pair.indices[0] == 0'i64

  test "Wider unsigned values only bypass dispatch for scalar and empty inputs":
    checkExceptionalPrefixes(high(uint16))
    checkExceptionalPrefixes(high(uint32))
    checkExceptionalPrefixes(high(uint64))

  test "Complex scalar and empty values bypass dispatch without losing components":
    checkExceptionalPrefixes(Complex[float32](re: 2, im: 7))
    checkExceptionalPrefixes(Complex[float64](re: 2, im: 7))
    let scalar = newTensorWith[Complex[float32]]([], Complex[float32](
      re: cast[float32](0x7fc12345'u32), im: -0.0'f32)).cummin(-1)
    let value = scalar.values.atContiguousIndex(0)
    check cast[uint32](value.re) == 0x7fc12345'u32
    check cast[uint32](value.im) == 0x80000000'u32
    check scalar.indices.atContiguousIndex(0) == 0'i64

  test "Caller outputs preserve transposed, offset and signed layouts":
    let source = [[9'i64, 1, 7], [0, 8, 2]].toTensor
    let expectedMinValues = [[9'i64, 1, 1], [0, 0, 0]].toTensor
    let expectedMinIndices = [[0'i64, 1, 1], [0, 0, 0]].toTensor
    let expectedMaxValues = [[9'i64, 9, 9], [0, 8, 8]].toTensor
    let expectedMaxIndices = [[0'i64, 0, 0], [0, 1, 1]].toTensor

    var valueOwner = newTensorWith[int64]([3, 4], 777)
    var values = valueOwner[_, 1..2].transpose()
    var indexOwner = newTensorWith[int64]([4, 5], -99)
    var indices = indexOwner[1..2, 3..1|-1]
    let valueBase = values.storage.raw_buffer
    let indexBase = indices.storage.raw_buffer
    let valueStrides = values.strides
    let indexStrides = indices.strides
    let valueOffset = values.offset
    let indexOffset = indices.offset
    check values.shape == source.shape
    check values.strides == [1, 4]
    check indices.shape == source.shape
    check indices.strides == [5, -1]
    check valueOffset == 1
    check indexOffset == 8

    source.cummin(1, values, indices)
    check values == expectedMinValues
    check indices == expectedMinIndices
    source.cummax(1, values, indices)
    check values == expectedMaxValues
    check indices == expectedMaxIndices
    check values.storage.raw_buffer == valueBase
    check indices.storage.raw_buffer == indexBase
    check values.strides == valueStrides
    check indices.strides == indexStrides
    check values.offset == valueOffset
    check indices.offset == indexOffset
    for row in 0 ..< 3:
      check valueOwner[row, 0] == 777
      check valueOwner[row, 3] == 777
    for column in 0 ..< 5:
      check indexOwner[0, column] == -99
      check indexOwner[3, column] == -99
    for row in 1 .. 2:
      check indexOwner[row, 0] == -99
      check indexOwner[row, 4] == -99

  test "Caller outputs accept disjoint shared interleaved storage":
    let source = [[9'i64, 1, 7], [0, 8, 2]].toTensor
    let expectedValues = [[9'i64, 1, 1], [0, 0, 0]].toTensor
    let expectedIndices = [[0'i64, 1, 1], [0, 0, 0]].toTensor
    var owner = newTensorWith[int64]([2, 6], -88)
    var values = owner[_, 0..4|2]
    var indices = owner[_, 1..5|2]
    let base = owner.storage.raw_buffer
    source.cummin(1, values, indices)
    check values == expectedValues
    check indices == expectedIndices
    check values.storage.raw_buffer == base
    check indices.storage.raw_buffer == base
    check owner[0, 0] == 9
    check owner[0, 1] == 0
    check owner[0, 2] == 1
    check owner[0, 3] == 1
    check owner[0, 4] == 1
    check owner[0, 5] == 1

    let floatSource = [[2.0'f64, 1, 3], [1, 3, 3]].toTensor
    let floatExpected = [[2.0'f64, 1, 1], [1, 1, 1]].toTensor
    let indexExpected = [[0'i64, 1, 1], [0, 0, 0]].toTensor
    var crossOwner = newTensorWith[int64]([2, 6], -77)
    var floatValues = newTensorUninit[float64](floatSource.shape)
    var floatStorage: CpuStorage[float64]
    floatStorage.cpuStorageFromBuffer(
      cast[pointer](crossOwner.storage.raw_buffer), int(crossOwner.size))
    floatValues.storage = floatStorage
    floatValues.strides[0] = 6
    floatValues.strides[1] = 2
    var crossIndices = crossOwner[_, 1..5|2]
    let crossBase = crossOwner.storage.raw_buffer
    floatSource.cummin(1, floatValues, crossIndices)
    check floatValues == floatExpected
    check crossIndices == indexExpected
    check cast[pointer](floatValues.storage.raw_buffer) == cast[pointer](crossBase)
    check crossIndices.storage.raw_buffer == crossBase

  test "Exact source aliases use sequential writes and preserve scalar bits":
    let minimumValues = [[9'i64, 1, 1], [0, 0, 0]].toTensor
    let minimumIndices = [[0'i64, 1, 1], [0, 0, 0]].toTensor
    var sourceValues = [[9'i64, 1, 7], [0, 8, 2]].toTensor
    var values = sourceValues
    var indices = newTensorUninit[int64](sourceValues.shape)
    sourceValues.cummin(1, values, indices)
    check values == minimumValues
    check indices == minimumIndices
    check sourceValues == minimumValues

    let maximumValues = [[9'i64, 9, 9], [0, 8, 8]].toTensor
    let maximumIndices = [[0'i64, 0, 0], [0, 1, 1]].toTensor
    var sourceIndices = [[9'i64, 1, 7], [0, 8, 2]].toTensor
    var indexAlias = sourceIndices
    var independentValues = newTensorUninit[int64](sourceIndices.shape)
    sourceIndices.cummax(1, independentValues, indexAlias)
    check independentValues == maximumValues
    check indexAlias == maximumIndices
    check sourceIndices == maximumIndices

    let floatSource = [[2.0'f64, 1, 3], [1, 3, 3]].toTensor
    let floatExpected = [[2.0'f64, 1, 1], [1, 1, 1]].toTensor
    let indexExpected = [[0'i64, 1, 1], [0, 0, 0]].toTensor
    var byteStorage: CpuStorage[int64]
    byteStorage.cpuStorageFromBuffer(
      cast[pointer](floatSource.storage.raw_buffer),
      floatSource.storage.capacityElems)
    var byteIndices = newTensorUninit[int64](floatSource.shape)
    byteIndices.storage = byteStorage
    byteIndices.strides = floatSource.strides
    byteIndices.offset = floatSource.offset
    var floatValues = newTensorUninit[float64](floatSource.shape)
    floatSource.cummin(1, floatValues, byteIndices)
    check floatValues == floatExpected
    check byteIndices == indexExpected
    check cast[pointer](byteIndices.storage.raw_buffer) ==
      cast[pointer](floatSource.storage.raw_buffer)

    let complexSource = newTensorWith[Complex[float64]]([],
      Complex[float64](re: 1.25, im: 2.5))
    var complexValues = newTensorUninit[Complex[float64]]([0])
    var componentStorage: CpuStorage[int64]
    componentStorage.cpuStorageFromBuffer(
      cast[pointer](complexSource.storage.raw_buffer), 1)
    var componentIndex = newTensorUninit[int64]([])
    componentIndex.storage = componentStorage
    complexSource.cummin(-1, complexValues, componentIndex)
    let scalarResult = complexValues.atContiguousIndex(0)
    check scalarResult.re == 1.25
    check scalarResult.im == 2.5
    check componentIndex.atContiguousIndex(0) == 0'i64
    let sourceAfter = complexSource.atContiguousIndex(0)
    check sourceAfter.re == 0.0
    check sourceAfter.im == 2.5

  test "Caller outputs retain offsets and detach safely on growth":
    let source = [[2'i64, 1, 3], [1, 3, 3]].toTensor
    let expectedValues = [[2'i64, 1, 1], [1, 1, 1]].toTensor
    let expectedIndices = [[0'i64, 1, 1], [0, 0, 0]].toTensor

    var valueOwner = newTensorWith[int64]([12], 77)
    var values = valueOwner[3..3]
    var indexOwner = newTensorWith[int64]([12], 88)
    var indices = indexOwner[2..2]
    let valueBase = values.storage.raw_buffer
    let indexBase = indices.storage.raw_buffer
    let oldValues = values
    let oldIndices = indices
    source.cummin(1, values, indices)
    check values == expectedValues
    check indices == expectedIndices
    check values.shape == source.shape
    check values.strides == [3, 1]
    check indices.strides == [3, 1]
    check values.offset == 3
    check indices.offset == 2
    check values.storage.raw_buffer == valueBase
    check indices.storage.raw_buffer == indexBase
    check oldValues.shape == [1]
    check oldValues.offset == 3
    check oldValues[0] == expectedValues[0, 0]
    check oldIndices.shape == [1]
    check oldIndices.offset == 2
    check oldIndices[0] == expectedIndices[0, 0]

    let emptySource = newTensor[int64](2, 0, 3)
    var emptyValueOwner = newTensorWith[int64]([12], 17)
    var emptyValues = emptyValueOwner[3..3]
    var emptyIndexOwner = newTensorWith[int64]([12], 18)
    var emptyIndices = emptyIndexOwner[2..2]
    let emptyValueBase = emptyValues.storage.raw_buffer
    let emptyIndexBase = emptyIndices.storage.raw_buffer
    emptySource.cummin(1, emptyValues, emptyIndices)
    check emptyValues.shape == [2, 0, 3]
    check emptyIndices.shape == [2, 0, 3]
    check emptyValues.strides == [3, 3, 1]
    check emptyIndices.strides == [3, 3, 1]
    check emptyValues.offset == 3
    check emptyIndices.offset == 2
    check emptyValues.storage.raw_buffer == emptyValueBase
    check emptyIndices.storage.raw_buffer == emptyIndexBase
    check emptyValueOwner[3] == 17
    check emptyIndexOwner[2] == 18

    let scalar = newTensorWith[int64]([], 23)
    var scalarValueOwner = newTensorWith[int64]([12], 27)
    var scalarValues = scalarValueOwner[3..3]
    var scalarIndexOwner = newTensorWith[int64]([12], 28)
    var scalarIndices = scalarIndexOwner[2..2]
    let scalarValueBase = scalarValues.storage.raw_buffer
    let scalarIndexBase = scalarIndices.storage.raw_buffer
    scalar.cummax(-1, scalarValues, scalarIndices)
    check scalarValues.shape.len == 0
    check scalarIndices.shape.len == 0
    check scalarValues.strides.len == 0
    check scalarIndices.strides.len == 0
    check scalarValues.offset == 3
    check scalarIndices.offset == 2
    check scalarValues.storage.raw_buffer == scalarValueBase
    check scalarIndices.storage.raw_buffer == scalarIndexBase
    check scalarValueOwner[3] == 23
    check scalarIndexOwner[2] == 0

    var growthValueOwner = newTensorWith[int64]([4], 37)
    var growthValues = growthValueOwner[3..3]
    var growthIndexOwner = newTensorWith[int64]([3], 38)
    var growthIndices = growthIndexOwner[2..2]
    let oldValueBase = growthValues.storage.raw_buffer
    let oldIndexBase = growthIndices.storage.raw_buffer
    let oldValueView = growthValues
    let oldIndexView = growthIndices
    source.cummin(1, growthValues, growthIndices)
    check growthValues == expectedValues
    check growthIndices == expectedIndices
    check growthValues.offset == 3
    check growthIndices.offset == 2
    check growthValues.storage.raw_buffer != oldValueBase
    check growthIndices.storage.raw_buffer != oldIndexBase
    check oldValueView.shape == [1]
    check oldIndexView.shape == [1]
    check oldValueView.storage.raw_buffer == oldValueBase
    check oldIndexView.storage.raw_buffer == oldIndexBase
    check oldValueView[0] == 37
    check oldIndexView[0] == 38
    check growthValueOwner[3] == 37
    check growthIndexOwner[2] == 38

  test "Axis and dtype failures precede destination changes":
    let source = [[2'i64, 1, 3], [1, 3, 3]].toTensor
    var values = newTensorWith[int64]([1], 41)
    var indices = newTensorWith[int64]([1], 52)
    let valueBase = values.storage.raw_buffer
    let indexBase = indices.storage.raw_buffer
    expect IndexDefect:
      source.cummin(3, values, indices)
    check values.shape == [1]
    check indices.shape == [1]
    check values.storage.raw_buffer == valueBase
    check indices.storage.raw_buffer == indexBase
    check values[0] == 41
    check indices[0] == 52

    let unsupported = newTensorWith[uint16]([2, 3], 4'u16)
    var unsignedValues = newTensorWith[uint16]([1], 61'u16)
    var unsignedIndices = newTensorWith[int64]([1], 62)
    let unsignedValueBase = unsignedValues.storage.raw_buffer
    let unsignedIndexBase = unsignedIndices.storage.raw_buffer
    expect ValueError:
      unsupported.cummax(1, unsignedValues, unsignedIndices)
    check unsignedValues.shape == [1]
    check unsignedIndices.shape == [1]
    check unsignedValues.storage.raw_buffer == unsignedValueBase
    check unsignedIndices.storage.raw_buffer == unsignedIndexBase
    check unsignedValues[0] == 61'u16
    check unsignedIndices[0] == 62

  test "Capacity and malformed metadata failures preserve both outputs":
    let source = [[2'i64, 1, 3], [1, 3, 3]].toTensor
    var values = newTensorWith[int64]([2, 3], 41)
    var shortIndices = newTensorWith[int64]([4], 52)
    shortIndices.shape = values.shape
    shortIndices.strides = values.strides
    let valueBase = values.storage.raw_buffer
    let indexBase = shortIndices.storage.raw_buffer
    let valueBefore = [valueBase[0], valueBase[1], valueBase[2],
      valueBase[3], valueBase[4], valueBase[5]]
    let indexBefore = [indexBase[0], indexBase[1], indexBase[2], indexBase[3]]
    expect ValueError:
      source.cummin(1, values, shortIndices)
    check values.shape == [2, 3]
    check shortIndices.shape == [2, 3]
    check shortIndices.strides == [3, 1]
    check values.strides == [3, 1]
    check values.offset == 0
    check shortIndices.offset == 0
    check values.storage.raw_buffer == valueBase
    check shortIndices.storage.raw_buffer == indexBase
    check [valueBase[0], valueBase[1], valueBase[2],
      valueBase[3], valueBase[4], valueBase[5]] == valueBefore
    check [indexBase[0], indexBase[1], indexBase[2], indexBase[3]] == indexBefore

    var malformedValues = newTensorWith[int64]([2, 3], 63)
    var validIndices = newTensorWith[int64]([2, 3], 64)
    let malformedBase = malformedValues.storage.raw_buffer
    let validIndexBase = validIndices.storage.raw_buffer
    malformedValues.strides.len = 1
    expect ValueError:
      source.cummin(1, malformedValues, validIndices)
    check malformedValues.shape == [2, 3]
    check malformedValues.strides.len == 1
    check malformedValues.storage.raw_buffer == malformedBase
    check validIndices.storage.raw_buffer == validIndexBase
    check validIndices.shape == [2, 3]
    check validIndices.strides == [3, 1]
    check validIndices.offset == 0
    for position in 0 ..< 6:
      check malformedBase[position] == 63
      check validIndexBase[position] == 64

    var overflowValues = newTensorWith[int64]([2, 3], 71)
    var overflowIndices = newTensorWith[int64]([2, 3], 72)
    let overflowValueBase = overflowValues.storage.raw_buffer
    let overflowIndexBase = overflowIndices.storage.raw_buffer
    overflowValues.offset = 1
    overflowValues.strides[0] = high(int)
    expect ValueError:
      source.cummin(1, overflowValues, overflowIndices)
    check overflowValues.offset == 1
    check overflowValues.strides[0] == high(int)
    check overflowValues.storage.raw_buffer == overflowValueBase
    check overflowIndices.storage.raw_buffer == overflowIndexBase
    for position in 0 ..< 6:
      check overflowValueBase[position] == 71
      check overflowIndexBase[position] == 72
    check overflowIndices.shape == [2, 3]
    check overflowIndices.strides == [3, 1]
    check overflowIndices.offset == 0

  test "Post-resize exact source/value mapping remains safe":
    var source = [[2'i64, 1, 3], [1, 3, 3]].toTensor
    var values = source.reshape(6)
    var indices = newTensorUninit[int64]([0])
    let inputBase = source.storage.raw_buffer
    source.cummin(1, values, indices)
    check values.shape == [2, 3]
    check values == [[2'i64, 1, 1], [1, 1, 1]].toTensor
    check indices == [[0'i64, 1, 1], [0, 0, 0]].toTensor
    check source == values
    check values.storage.raw_buffer == inputBase

  test "Shifted and internally overlapping output aliases reject atomically":
    var backing = newTensorWith[int64]([16], 99)
    backing[4] = 9
    backing[5] = 1
    backing[6] = 7
    backing[7] = 0
    backing[8] = 8
    backing[9] = 2
    let source = backing[4..9].reshape(2, 3)
    var values = backing[3..3]
    var indices = newTensorWith[int64]([2, 3], 55)
    let beforeBacking = backing.clone
    let beforeIndices = indices.clone
    let sourceBase = backing.storage.raw_buffer
    expect ValueError:
      source.cummin(1, values, indices)
    check backing == beforeBacking
    check indices == beforeIndices
    check values.shape == [1]
    check values.offset == 3
    check values.storage.raw_buffer == sourceBase

    var sharedOwner = newTensorWith[int64]([8], 11)
    var shiftedValues = sharedOwner[0..5].reshape(2, 3)
    var shiftedIndices = sharedOwner[1..6].reshape(2, 3)
    let sharedBefore = sharedOwner.clone
    expect ValueError:
      source.cummax(1, shiftedValues, shiftedIndices)
    check sharedOwner == sharedBefore
    check shiftedValues.offset == 0
    check shiftedIndices.offset == 1

    var broadcastOwner = newTensorWith[int64]([1, 3], 17)
    var broadcastValues = broadcastOwner.broadcast(2, 3)
    var ordinaryIndices = newTensorWith[int64]([2, 3], 18)
    let broadcastBefore = broadcastOwner.clone
    let indicesBefore = ordinaryIndices.clone
    expect ValueError:
      source.cummin(1, broadcastValues, ordinaryIndices)
    check broadcastOwner == broadcastBefore
    check ordinaryIndices == indicesBefore
    check broadcastValues.strides[0] == 0

  test "Empty resize keeps representable strides without an unused product overflow":
    var source = newTensor[int64](0)
    source.shape = toMetadata([high(int), 0, high(int)])
    source.strides = toMetadata([high(int), high(int), 1])
    var values = newTensorWith[int64]([1], 17)
    var indices = newTensorWith[int64]([1], 18)
    let valueOwner = values
    let indexOwner = indices
    source.cummax(1, values, indices)
    check values.shape == source.shape
    check indices.shape == source.shape
    check values.strides == [high(int), high(int), 1]
    check indices.strides == values.strides
    check values.storage == valueOwner.storage
    check indices.storage == indexOwner.storage
    check valueOwner[0] == 17
    check indexOwner[0] == 18

  test "Bounded affine proofs never write through an unresolved or overlapping map":
    var source = newTensorUninit[int64](2, 2, 2, 2)
    for position in 0..<16:
      source.storage.raw_buffer[position] = int64((position * 7) mod 5 - 2)
    let sourceBefore = source.clone
    let expected = source.cummax(-1)
    var owner = newTensorWith[int64]([2674], 71)
    var values = source
    values.storage = owner.storage
    values.strides = toMetadata([100, 251, 631, 1591])
    var indices = values
    indices.offset = 2
    var occupied: array[2674, bool]
    for logical in 0..<16:
      let address = ((logical shr 3) and 1) * 100 +
        ((logical shr 2) and 1) * 251 +
        ((logical shr 1) and 1) * 631 + (logical and 1) * 1591
      check not occupied[address]
      check not occupied[address + 2]
      occupied[address] = true
      occupied[address + 2] = true
    let ownerBefore = owner.clone
    try:
      source.cummax(-1, values, indices)
      check values == expected.values
      check indices == expected.indices
      for address in 0..<2674:
        if not occupied[address]:
          check owner[address] == 71
    except ValueError as error:
      # A better proof may accept this disjoint map; an exhausted proof cannot write.
      check error.msg.contains("could not be proven")
      check owner == ownerBefore
    check source == sourceBefore
    check values.strides == [100, 251, 631, 1591]
    check indices.strides == values.strides
    check values.offset == 0
    check indices.offset == 2
    var overlappingIndices = indices
    overlappingIndices.offset = 100
    let beforeOverlap = owner.clone
    expect ValueError:
      source.cummin(0, values, overlappingIndices)
    check owner == beforeOverlap
    check source == sourceBefore
    check values.offset == 0
    check overlappingIndices.offset == 100

  test "Fixed and unknown-capacity external outputs cannot grow or partially resize":
    let source = [[2'i64, 1, 3], [1, 3, 3]].toTensor
    let valueOwner = newTensorWith[int64]([12], 41)
    let indexOwner = newTensorWith[int64]([6], 52)
    var values = valueOwner[3..3]
    var fixedStorage: CpuStorage[int64]
    fixedStorage.cpuStorageFromBuffer(cast[pointer](indexOwner.storage.raw_buffer), 1)
    var indices: Tensor[int64]
    indices.shape = toMetadata([1])
    indices.strides = toMetadata([1])
    indices.storage = fixedStorage
    expect ValueError:
      source.cummin(1, values, indices)
    check values.shape == [1]
    check values.strides == [1]
    check values.offset == 3
    check values.storage == valueOwner.storage
    check indices.shape == [1]
    check indices.strides == [1]
    check indices.offset == 0
    check indices.storage == fixedStorage
    for value in valueOwner:
      check value == 41
    for value in indexOwner:
      check value == 52

    let buffer = allocShared(6 * sizeof(int64))
    let words = cast[ptr UncheckedArray[int64]](buffer)
    for index in 0..<6:
      words[index] = 93
    var unknownStorage: CpuStorage[int64]
    unknownStorage.cpuStorageFromBuffer(buffer, buffer, releaseExternalOutput)
    var unknownValues: Tensor[int64]
    unknownValues.shape = toMetadata([1])
    unknownValues.strides = toMetadata([1])
    unknownValues.storage = unknownStorage
    var unknownIndices = newTensorWith[int64]([1], 94)
    let indexStorage = unknownIndices.storage
    for sameShape in [false, true]:
      if sameShape:
        unknownValues.shape = source.shape
        unknownValues.strides = source.strides
      let oldShape = unknownValues.shape
      let oldStrides = unknownValues.strides
      expect ValueError:
        source.cummax(1, unknownValues, unknownIndices)
      check unknownValues.shape == oldShape
      check unknownValues.strides == oldStrides
      check unknownValues.storage == unknownStorage
      check unknownIndices.storage == indexStorage
      check unknownIndices.shape == [1]
      check unknownIndices[0] == 94
      for index in 0..<6:
        check words[index] == 93
