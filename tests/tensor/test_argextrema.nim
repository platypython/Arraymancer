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

## Exercises first-index selection, signed coordinate carries and dispatch shortcuts.
import ../../src/arraymancer
import std/[unittest, complex]

proc checkIndexDtype[T: SomeNumber]() =
  ## Index dtype and shape do not depend on the input element width.
  let source = [[T(3), T(1), T(3)], [T(2), T(1), T(3)]].toTensor
  let whole: Tensor[int64] = source.argmax()
  check whole.shape.len == 0
  check whole.atContiguousIndex(0) == 0'i64
  check source.argmin().atContiguousIndex(0) == 1'i64
  check source.argmax(keepdim = true).shape == toMetadata([1, 1])
  check source.argmin(0) == [1'i64, 0, 0].toTensor
  check source.argmax(0) == [0'i64, 0, 0].toTensor
  check source.argmin(-1) == [1'i64, 1].toTensor
  check source.argmax(1) == [0'i64, 2].toTensor
  check source.argmax(1, keepdim = true) == [[0'i64], [2'i64]].toTensor
  check source.argmin(0, keepdim = true) == [[1'i64, 0, 0]].toTensor
  let scalar = newTensorWith[T]([], T(7))
  check scalar.argmin().atContiguousIndex(0) == 0'i64
  check scalar.argmax(-1, keepdim = true).shape.len == 0
  var indices = source.argmax(1)
  indices[0] = 55'i64
  check source[0, 0] == T(3)

proc checkNanAndTiles[T: SomeFloat]() =
  ## A partial column tile must not lose an earlier NaN or an equal zero.
  var source = newTensorWith[T]([2, 33], T(1))
  source[0, 17] = T(NaN)
  source[0, 19] = T(NaN)
  source[1, 31] = T(Inf)
  source[1, 32] = T(NegInf)
  check source.argmin().atContiguousIndex(0) == 17'i64
  check source.argmax().atContiguousIndex(0) == 17'i64
  check source.argmin(1) == [17'i64, 32].toTensor
  check source.argmax(1) == [17'i64, 31].toTensor
  let minima = source.argmin(0)
  let maxima = source.argmax(0)
  for column in 0 ..< 33:
    check minima[column] == (if column == 32: 1'i64 else: 0'i64)
    check maxima[column] == (if column == 31: 1'i64 else: 0'i64)
  let transposed = source.transpose()
  check transposed.argmin().atContiguousIndex(0) == 34'i64
  check transposed.argmax().atContiguousIndex(0) == 34'i64
  let offset = transposed[_, 1..^1]
  check offset.argmin().atContiguousIndex(0) == 32'i64
  check offset.argmax().atContiguousIndex(0) == 31'i64
  let zeros = [T(-0.0), T(0), T(-0.0), T(0)].toTensor
  check zeros.argmin().atContiguousIndex(0) == 0'i64
  check zeros.argmax().atContiguousIndex(0) == 0'i64
  check newTensorWith[T]([], T(NaN)).argmin(0).atContiguousIndex(0) == 0'i64
  check newTensorWith[T]([], T(NaN)).argmax().atContiguousIndex(0) == 0'i64

proc checkWideShortcuts[T: SomeUnsignedInt]() =
  ## Scalar dispatch differs from a positive-rank explicit singleton dimension.
  let scalar = newTensorWith[T]([], T(3))
  expect ValueError: discard scalar.argmin(0)
  expect ValueError: discard scalar.argmax(-1)
  let singleton = [T(3)].toTensor
  expect ValueError: discard singleton.argmax()
  check singleton.argmax(0).atContiguousIndex(0) == 0'i64
  let positiveRank = newTensorWith[T]([2, 1, 3], T(5))
  check positiveRank.argmin(1) == newTensor[int64](2, 3)
  check positiveRank.argmax(-2, keepdim = true) == newTensor[int64](2, 1, 3)
  expect ValueError: discard positiveRank.argmax(0)
  expect IndexDefect: discard positiveRank.argmax(3)
  let empty = newTensor[T](0, 3)
  check empty.argmin(1).shape == toMetadata([0])
  check empty.argmax(1, keepdim = true).shape == toMetadata([0, 1])
  expect IndexDefect: discard empty.argmin()
  expect IndexDefect: discard empty.argmax(0)

suite "CPU first-index argmin/argmax":
  test "Indices are int64 with squeezed defaults for every supported input dtype":
    checkIndexDtype[float32]()
    checkIndexDtype[float64]()
    checkIndexDtype[int8]()
    checkIndexDtype[int16]()
    checkIndexDtype[int32]()
    checkIndexDtype[int64]()
    checkIndexDtype[uint8]()
    checkIndexDtype[int]()

  test "Integer bounds are compared without floating-point promotion":
    let signed = [high(int64), high(int64) - 1, low(int64), high(int64)].toTensor
    check signed.argmax().atContiguousIndex(0) == 0'i64
    check signed.argmin().atContiguousIndex(0) == 2'i64
    let bytes = [255'u8, 0, 254, 255].toTensor
    check bytes.argmax().atContiguousIndex(0) == 0'i64
    check bytes.argmin().atContiguousIndex(0) == 1'i64

  test "Float32 packed tails, NaNs and zero ties retain the first logical index":
    checkNanAndTiles[float32]()

  test "Float64 packed tails, NaNs and zero ties retain the first logical index":
    checkNanAndTiles[float64]()

  test "Signed retained/reduced axes and broadcasts preserve logical index order":
    let matrix = [[9, 1, 9], [0, 8, 8]].toTensor
    let reversedRows = matrix[1..0|-1, _]
    check reversedRows.argmax(1) == [1'i64, 0].toTensor
    check reversedRows.argmin(1) == [0'i64, 1].toTensor
    check reversedRows.argmax(0) == [1'i64, 0, 1].toTensor
    check reversedRows.argmin(0) == [0'i64, 1, 0].toTensor
    check reversedRows.argmax().atContiguousIndex(0) == 3'i64
    let reverse = ([2, 1, 3, 1, 3].toTensor)[4..0|-1]
    check reverse.argmax().atContiguousIndex(0) == 0'i64
    check reverse.argmin(0).atContiguousIndex(0) == 1'i64
    let broadcast = [[1, 3, 2]].toTensor.broadcast(4, 3)
    check broadcast.argmax(0) == [0'i64, 0, 0].toTensor
    check broadcast.argmax(1) == [1'i64, 1, 1, 1].toTensor
    check broadcast.argmax().atContiguousIndex(0) == 1'i64

  test "Wider unsigned dispatch skips only explicit singleton axes and empty outputs":
    checkWideShortcuts[uint16]()
    checkWideShortcuts[uint32]()
    checkWideShortcuts[uint64]()
    checkWideShortcuts[uint]()

  test "Bool/complex dtype rejection precedes invalid or empty dimensions":
    let flags = newTensor[bool](0, 3)
    expect ValueError: discard flags.argmax(2)
    expect ValueError: discard flags.argmin(0)
    expect ValueError: discard newTensorWith[bool]([], true).argmax()
    let complexEmpty = newTensor[Complex[float32]](0, 3)
    expect ValueError: discard complexEmpty.argmin(2)
    expect ValueError: discard complexEmpty.argmax(0)
    let empty = newTensor[float64](2, 0, 3)
    check empty.argmax(0).shape == toMetadata([0, 3])
    expect IndexDefect: discard empty.argmin(1)
    expect IndexDefect: discard empty.argmax()
    expect IndexDefect: discard newTensorWith[float64]([], 1.0).argmin(1)
