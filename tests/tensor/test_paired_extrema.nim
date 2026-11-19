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

## Exercises paired value/index correspondence, exact source bits and dispatch precedence.
import ../../src/arraymancer
import std/[unittest, math, complex]

proc checkNumericPairs[T: SomeNumber]() =
  ## Both outputs squeeze the same axis without promoting the selected value.
  let source = [[T(3), T(1), T(3)], [T(2), T(1), T(3)]].toTensor
  let maxima: tuple[values: Tensor[T], indices: Tensor[int64]] = source.maxWithIndices(1)
  check maxima.values == [T(3), T(3)].toTensor
  check maxima.indices == [0'i64, 2].toTensor
  let minima = source.minWithIndices(-1, keepdim = true)
  check minima.values == [[T(1)], [T(1)]].toTensor
  check minima.indices == [[1'i64], [1'i64]].toTensor
  let columns = source.minWithIndices(0)
  check columns.values == [T(2), T(1), T(3)].toTensor
  check columns.indices == [1'i64, 0, 0].toTensor
  let scalar = newTensorWith[T]([], T(7)).maxWithIndices(-1, keepdim = true)
  check scalar.values.shape.len == 0
  check scalar.indices.shape.len == 0
  check scalar.values.atContiguousIndex(0) == T(7)
  check scalar.indices.atContiguousIndex(0) == 0'i64

proc checkFloatingPairs[T: SomeFloat]() =
  ## Tiled and contiguous reducers must return the bits at the first selected index.
  var source = newTensorWith[T]([2, 33], T(0))
  for column in 0 ..< 33:
    source[0, column] = T(-0.0)
  let columns = source.maxWithIndices(0)
  for column in 0 ..< 33:
    check columns.values[column].classify == fcNegZero
    check columns.indices[column] == 0'i64
  let rows = source.minWithIndices(1)
  check rows.values[0].classify == fcNegZero
  check rows.values[1].classify == fcZero
  check rows.indices == [0'i64, 0].toTensor
  when T is float32:
    let firstNan = cast[float32](0x7fc12345'u32)
    let laterNan = cast[float32](0x7fc6789a'u32)
  else:
    let firstNan = cast[float64](0x7ff8123456789abc'u64)
    let laterNan = cast[float64](0x7ff8abcdef123456'u64)
  source[0, 17] = firstNan
  source[0, 19] = laterNan
  source[1, 31] = T(Inf)
  source[1, 32] = T(NegInf)
  let maxima = source.maxWithIndices(1)
  let minima = source.minWithIndices(1)
  check maxima.indices == [17'i64, 31].toTensor
  check minima.indices == [17'i64, 32].toTensor
  when T is float32:
    check cast[uint32](maxima.values[0]) == 0x7fc12345'u32
    check cast[uint32](minima.values[0]) == 0x7fc12345'u32
  else:
    check cast[uint64](maxima.values[0]) == 0x7ff8123456789abc'u64
    check cast[uint64](minima.values[0]) == 0x7ff8123456789abc'u64
  let transposed = source.transpose().maxWithIndices(0)
  check transposed.indices == maxima.indices
  check transposed.values[0].isNaN
  check transposed.values[1] == T(Inf)

proc checkUnsignedPairs[T: SomeUnsignedInt]() =
  ## Paired dispatch accepts rank-zero scalars, never positive-rank singleton axes.
  let scalar = newTensorWith[T]([], high(T)).minWithIndices(0)
  check scalar.values.atContiguousIndex(0) == high(T)
  check scalar.indices.atContiguousIndex(0) == 0'i64
  expect ValueError: discard [T(3)].toTensor.maxWithIndices(0)
  expect ValueError: discard newTensorWith[T]([2, 1, 3], T(3)).minWithIndices(1)
  let empty = newTensor[T](0, 3).maxWithIndices(1, keepdim = true)
  check empty.values.shape == toMetadata([0, 1])
  check empty.indices.shape == toMetadata([0, 1])
  expect IndexDefect: discard newTensor[T](0, 3).minWithIndices(0)
  expect IndexDefect: discard [T(3)].toTensor.maxWithIndices(1)

suite "CPU paired min/max extrema":
  test "Values retain dtype and indices are int64 across numeric widths":
    checkNumericPairs[float32]()
    checkNumericPairs[float64]()
    checkNumericPairs[int8]()
    checkNumericPairs[int16]()
    checkNumericPairs[int32]()
    checkNumericPairs[int64]()
    checkNumericPairs[uint8]()
    checkNumericPairs[int]()

  test "Boolean values support first true and first false indices":
    let source = [[true, false, true], [false, false, true]].toTensor
    let maxima = source.maxWithIndices(1)
    let minima = source.minWithIndices(1)
    check maxima.values == [true, true].toTensor
    check maxima.indices == [0'i64, 2].toTensor
    check minima.values == [false, false].toTensor
    check minima.indices == [1'i64, 0].toTensor
    check source.transpose().maxWithIndices(0).indices == maxima.indices

  test "Float32 values preserve first signed-zero and NaN payload bits":
    checkFloatingPairs[float32]()

  test "Float64 values preserve first signed-zero and NaN payload bits":
    checkFloatingPairs[float64]()

  test "Signed and broadcast views retain value/index correspondence":
    let source = [[9, 1, 9], [0, 8, 8]].toTensor
    let reversed = source[1..0|-1, _].maxWithIndices(0)
    check reversed.values == [9, 8, 9].toTensor
    check reversed.indices == [1'i64, 0, 1].toTensor
    let rows = source[_, 2..0|-1].minWithIndices(1)
    check rows.values == [1, 0].toTensor
    check rows.indices == [1'i64, 2].toTensor
    let broadcast = [[1, 3, 2]].toTensor.broadcast(4, 3).maxWithIndices(0)
    check broadcast.values == [1, 3, 2].toTensor
    check broadcast.indices == [0'i64, 0, 0].toTensor
    let offset = source.transpose()[_, 1..^1].minWithIndices(0)
    check offset.values == [0].toTensor
    check offset.indices == [0'i64].toTensor

  test "Equal-width outputs do not alias each other or their input":
    let source = [9'i64, 2, 9].toTensor
    var pair = source.maxWithIndices(0)
    check pair.values.get_offset_ptr != pair.indices.get_offset_ptr
    pair.values.atContiguousIndex(0) = -77'i64
    pair.indices.atContiguousIndex(0) = 55'i64
    check source == [9'i64, 2, 9].toTensor
    check pair.values.atContiguousIndex(0) == -77'i64
    check pair.indices.atContiguousIndex(0) == 55'i64
    let bounds = [high(int64), high(int64) - 1, low(int64)].toTensor.minWithIndices(0)
    check bounds.values.atContiguousIndex(0) == low(int64)
    check bounds.indices.atContiguousIndex(0) == 2'i64

  test "Unsigned scalar and empty shortcuts differ from index-only reducers":
    checkUnsignedPairs[uint16]()
    checkUnsignedPairs[uint32]()
    checkUnsignedPairs[uint64]()
    checkUnsignedPairs[uint]()

  test "Complex axis validation precedes dtype rejection even for empty inputs":
    let empty = newTensor[Complex[float32]](0, 3)
    expect IndexDefect: discard empty.maxWithIndices(2)
    expect IndexDefect: discard empty.minWithIndices(0)
    expect ValueError: discard empty.maxWithIndices(1)
    let scalar = newTensor[Complex[float64]](newSeq[int]())
    expect IndexDefect: discard scalar.minWithIndices(1)
    expect ValueError: discard scalar.maxWithIndices(-1)
