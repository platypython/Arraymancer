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
import std/[unittest, complex]

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
