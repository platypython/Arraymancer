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

## Exercises fused extrema ordering, tensor views and dtype dispatch boundaries.
import std/[unittest, complex]
import ../../src/arraymancer

proc checkFloatingAminmax[T: SomeFloat]() =
  when sizeof(T) == 4:
    type Word = uint32
    const early = 0x7fc12345'u32
    const late = 0x7fc54321'u32
  else:
    type Word = uint64
    const early = 0x7ff8123456789abc'u64
    const late = 0x7ff876543210abcd'u64
  let negativeZero = cast[Word](T(-0.0))
  let zeros = @[T(-0.0), T(0.0), T(-0.0), T(0.0)].toTensor.reshape(2, 2)
  let dimension = zeros.aminmax(-1)
  check cast[Word](dimension.min[0]) == negativeZero
  check cast[Word](dimension.max[0]) == negativeZero
  check cast[Word](dimension.min[1]) == negativeZero
  check cast[Word](dimension.max[1]) == negativeZero
  when defined(arm64):
    let whole = zeros.aminmax()
    check cast[Word](whole.min.unsafe_raw_offset[0]) == negativeZero
    check cast[Word](whole.max.unsafe_raw_offset[0]) == Word(0)
  var scalar = newTensorUninit[T](newSeq[int]())
  scalar.unsafe_raw_offset[0] = T(-0.0)
  for keepdim in [false, true]:
    let whole = scalar.aminmax(keepdim = keepdim)
    check whole.min.rank == 0 and whole.max.rank == 0
    check cast[Word](whole.min.unsafe_raw_offset[0]) == negativeZero
    check cast[Word](whole.max.unsafe_raw_offset[0]) == negativeZero

  when defined(arm64):
    const width = 16 div sizeof(T)
    var payloads = newTensorUninit[T]([2 * width])
    for index in 0 ..< payloads.size: payloads.unsafe_raw_offset[index] = T(1)
    payloads.unsafe_raw_offset[1] = cast[T](early)
    payloads.unsafe_raw_offset[width] = cast[T](late)
    let axisPair = payloads.aminmax(0)
    check cast[Word](axisPair.min.unsafe_raw_offset[0]) == early
    check cast[Word](axisPair.max.unsafe_raw_offset[0]) == early
    let wholePair = payloads.aminmax()
    check cast[Word](wholePair.min.unsafe_raw_offset[0]) == late
    check cast[Word](wholePair.max.unsafe_raw_offset[0]) == late

  let input = @[T(1), T(1), T(2), T(3), T(3), T(0)].toTensor.reshape(2, 3)
  let reverse = input[_, ^1..0|-1]
  let reversed = reverse.aminmax(1)
  check reversed.min == @[T(1), T(0)].toTensor
  check reversed.max == @[T(2), T(3)].toTensor
  let transpose = input.transpose
  let transposed = transpose.aminmax(0, keepdim = true)
  check transposed.min == @[T(1), T(0)].toTensor.reshape(1, 2)
  check transposed.max == @[T(2), T(3)].toTensor.reshape(1, 2)
  let offset = input[_, 1..^1]
  let offsetPair = offset.aminmax(0)
  check offsetPair.min == @[T(1), T(0)].toTensor
  check offsetPair.max == @[T(3), T(2)].toTensor
  let broadcast = input[0..0, _].broadcast(3, 3)
  let broadcastPair = broadcast.aminmax(0)
  check broadcastPair.min == @[T(1), T(1), T(2)].toTensor
  check broadcastPair.max == broadcastPair.min
  for view in [reverse, transpose, offset, broadcast]:
    let pair = view.aminmax()
    let copied = view.clone.aminmax()
    check pair.min == copied.min
    check pair.max == copied.max
  var independent = input.aminmax(1)
  independent.min.unsafe_raw_offset[0] = T(77)
  check independent.max == @[T(2), T(3)].toTensor
  check input[0, 0] == T(1)

  var columns = newTensorUninit[T]([2, 3, 33])
  for index in 0 ..< columns.size: columns.unsafe_raw_offset[index] = T(index)
  let columnPair = columns.aminmax(1, keepdim = true)
  check columnPair.min.shape == [2, 1, 33]
  check columnPair.max.shape == [2, 1, 33]
  for outer in 0 ..< 2:
    for column in 0 ..< 33:
      check columnPair.min[outer, 0, column] == T(outer * 99 + column)
      check columnPair.max[outer, 0, column] == T(outer * 99 + 66 + column)
  var columnBits = newTensorUninit[T]([3, 33])
  for row in 0 ..< 3:
    for column in 0 ..< 33:
      columnBits.unsafe_raw_offset[row * 33 + column] =
        if (row + column) mod 2 == 0: T(-0.0) else: T(0.0)
  columnBits.unsafe_raw_offset[31] = T(1)
  columnBits.unsafe_raw_offset[32] = T(1)
  columnBits.unsafe_raw_offset[33 + 31] = cast[T](early)
  columnBits.unsafe_raw_offset[66 + 31] = cast[T](late)
  columnBits.unsafe_raw_offset[33 + 32] = cast[T](late)
  columnBits.unsafe_raw_offset[66 + 32] = cast[T](early)
  let bitPair = columnBits.aminmax(0)
  for column in 0 ..< 31:
    let expected = if column mod 2 == 0: negativeZero else: Word(0)
    check cast[Word](bitPair.min[column]) == expected
    check cast[Word](bitPair.max[column]) == expected
  check cast[Word](bitPair.min[31]) == early
  check cast[Word](bitPair.max[31]) == early
  check cast[Word](bitPair.min[32]) == late
  check cast[Word](bitPair.max[32]) == late
  let empty = newTensor[T]([0, 3])
  let emptyPair = empty.aminmax(1)
  check emptyPair.min.shape == [0] and emptyPair.max.shape == [0]
  expect IndexDefect: discard empty.aminmax(0)
  expect IndexDefect: discard empty.aminmax(2)
  expect ValueError: discard empty.aminmax()

proc checkIntegerAminmax[T: SomeInteger]() =
  let input = @[low(T), high(T), T(1), T(0)].toTensor.reshape(2, 2)
  let pair = input.aminmax()
  check pair.min.unsafe_raw_offset[0] == low(T)
  check pair.max.unsafe_raw_offset[0] == high(T)
  let axisPair = input.aminmax(-1)
  check axisPair.min == @[low(T), T(0)].toTensor
  check axisPair.max == @[high(T), T(1)].toTensor

proc checkWideUnsignedAminmax[T: SomeUnsignedInt]() =
  var scalar = newTensorUninit[T](newSeq[int]())
  scalar.unsafe_raw_offset[0] = high(T)
  expect ValueError: discard scalar.aminmax()
  for axis in [-1, 0]:
    let pair = scalar.aminmax(axis)
    check pair.min.unsafe_raw_offset[0] == high(T)
    check pair.max.unsafe_raw_offset[0] == high(T)
  let singleton = @[high(T)].toTensor
  expect ValueError: discard singleton.aminmax(0)
  let empty = newTensor[T]([0, 3])
  expect ValueError: discard empty.aminmax(1)
  expect IndexDefect: discard empty.aminmax(0)
  expect IndexDefect: discard empty.aminmax(2)

suite "CPU joint aminmax":
  test "float32 first-axis and native whole fold bits with independent views":
    checkFloatingAminmax[float32]()
  test "float64 first-axis and native whole fold bits with independent views":
    checkFloatingAminmax[float64]()
  test "signed integer bounds never use floating conversions":
    checkIntegerAminmax[int8]()
    checkIntegerAminmax[int16]()
    checkIntegerAminmax[int32]()
    checkIntegerAminmax[int64]()
  test "uint8 and boolean outputs preserve their element types":
    checkIntegerAminmax[uint8]()
    let input = @[true, false, true, true].toTensor.reshape(2, 2)
    let pair = input.aminmax(1)
    check pair.min == @[false, true].toTensor
    check pair.max == @[true, true].toTensor
    let whole = input.aminmax()
    check not whole.min.unsafe_raw_offset[0]
    check whole.max.unsafe_raw_offset[0]
  test "wider unsigned dispatch distinguishes explicit scalars from empty outputs":
    checkWideUnsignedAminmax[uint16]()
    checkWideUnsignedAminmax[uint32]()
    checkWideUnsignedAminmax[uint64]()
  test "complex dtype rejection precedes invalid axes and empty reduced axes":
    let input = newTensor[Complex[float32]]([0, 3])
    expect ValueError: discard input.aminmax()
    expect ValueError: discard input.aminmax(0)
    expect ValueError: discard input.aminmax(5)
