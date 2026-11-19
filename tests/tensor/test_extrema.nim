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

## Covers extrema shape, view, validation and floating-point edge contracts.
import ../../src/arraymancer
import std/[unittest, math, complex]

proc checkIntegerExtremaTails[T: SomeInteger]() =
  ## A final extreme must survive both short vectors and four-vector tails.
  const width = 16 div sizeof(T)
  for count in [1, width - 1, width, width + 1, 4 * width - 1, 4 * width, 4 * width + 1]:
    var source = newTensorWith[T]([count], T(2))
    source[count - 1] = high(T)
    check source.amax().atContiguousIndex(0) == high(T)
    source[count - 1] = low(T)
    check source.amin().atContiguousIndex(0) == low(T)
    check source[count - 1..0|-1].amin().atContiguousIndex(0) == low(T)
  let extent = 4 * width + 1
  var cube = newTensorWith[T]([3, 5, extent], T(2))
  for column in 0 ..< 5:
    cube[2, column, extent - 1] = high(T)
  check cube.amax([0, 2]) == newTensorWith[T]([5], high(T))
  for column in 0 ..< 5:
    cube[2, column, extent - 1] = low(T)
  check cube.amin([2, 0]) == newTensorWith[T]([5], low(T))

proc checkFloatingExtrema[F: SomeFloat]() =
  ## NaNs must survive either scalar tails or full vector column tiles.
  var source = newTensor[F](3, 17)
  source[0, 0] = F(NaN)
  source[1, 1] = F(-Inf)
  source[2, 2] = F(Inf)
  let minima = source.amin(0)
  let maxima = source.amax(0)
  check minima.shape == [17]
  check maxima.shape == [17]
  check minima[0].isNaN and maxima[0].isNaN
  check minima[1] == F(-Inf) and maxima[1] == 0
  check minima[2] == 0 and maxima[2] == F(Inf)
  check source.amin().atContiguousIndex(0).isNaN
  check source.amax().atContiguousIndex(0).isNaN
  let scalar = newTensorWith[F]([], F(-0.0))
  check scalar.amin(keepdim = true).shape == newSeq[int]()
  check scalar.amax(-1).rank == 0
  check scalar.amin([0]).atContiguousIndex(0).classify == fcNegZero
  check scalar.amax().atContiguousIndex(0).classify == fcNegZero
  when defined(arm64):
    const width = 16 div sizeof(F)
    const columns = 4 * width + 1
    var zeros = newTensor[F](2, columns)
    for column in 0 ..< columns: zeros[1, column] = F(-0.0)
    let columnMinima = zeros.amin(0)
    let columnMaxima = zeros.amax(0)
    for column in 0 ..< columns - 1:
      check columnMinima[column].classify == fcNegZero
      check columnMaxima[column].classify == fcZero
    check columnMinima[columns - 1].classify == fcZero
    check columnMaxima[columns - 1].classify == fcZero
    var sparse = newTensor[F](columns)
    sparse[width] = F(-0.0)
    check sparse.amin().atContiguousIndex(0).classify == fcNegZero
    check sparse.amax().atContiguousIndex(0).classify == fcZero
    sparse[width] = F(0.0)
    sparse[columns - 1] = F(-0.0)
    check sparse.amin().atContiguousIndex(0).classify == fcZero
    check sparse.amax().atContiguousIndex(0).classify == fcZero
    var physical = newTensor[F](4, 8, 3)
    physical.unsafe_raw_offset[4] = F(-0.0)
    let fortran = physical.permute(2, 1, 0)
    let fortranMinima = fortran.amin([0, 1])
    let fortranMaxima = fortran.amax([0, 1])
    check fortranMinima.shape == [4]
    check fortranMinima[0].classify == fcNegZero
    for index in 1 ..< 4:
      check fortranMinima[index].classify == fcZero
    for index in 0 ..< 4:
      check fortranMaxima[index].classify == fcZero

suite "CPU amin/amax reductions":
  test "Multi-axis extrema preserve dtype, dimension order and independent storage":
    let cube = [0'i64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23].toTensor.reshape(2, 3, 4)
    var minima: Tensor[int64] = cube.amin([0, 2])
    check minima == [0'i64, 4, 8].toTensor
    check cube.amax([-1, 0]) == [15'i64, 19, 23].toTensor
    check cube.amin([2, 0], keepdim = true) == [0'i64, 4, 8].toTensor.reshape(1, 3, 1)
    check cube.amax([2, 1]) == [11'i64, 23].toTensor
    check cube.amin([]).shape == newSeq[int]()
    check cube.amax(keepdim = true).shape == [1, 1, 1]
    minima[0] = -99
    check cube[0, 0, 0] == 0
    let endpoints = [high(int64), low(int64), 0'i64].toTensor
    check endpoints.amin().atContiguousIndex(0) == low(int64)
    check endpoints.amax().atContiguousIndex(0) == high(int64)

  test "Extrema read offsets, negative strides, transposes and broadcasts in place":
    let owner = [99'i32, 3, -4, 7, 2, -1, 88].toTensor
    let offset = owner[1..5]
    check offset.amin().atContiguousIndex(0) == -4
    check offset[4..0|-1].amax().atContiguousIndex(0) == 7
    let matrix = [0'i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].toTensor.reshape(3, 4)
    check matrix.transpose.amin(0) == [0'i32, 4, 8].toTensor
    check matrix.transpose.amax(1) == [8'i32, 9, 10, 11].toTensor
    let broadcasted = [1'i32, -2, 3].toTensor.reshape(1, 3, 1).broadcast(2, 3, 4)
    check broadcasted.amin([0, 2]) == [1'i32, -2, 3].toTensor
    check broadcasted.amax([-1, 0], keepdim = true) ==
      [1'i32, -2, 3].toTensor.reshape(1, 3, 1)
    let booleans = [[true, false, true], [false, false, true]].toTensor
    check booleans.amin(0) == [false, false, true].toTensor
    check booleans.amax(0) == [true, false, true].toTensor
    check booleans.amin(1) == [false, false].toTensor
    check booleans.amax(1) == [true, true].toTensor

  test "Three separated reduced dimensions advance and reset their coordinates":
    var source = newTensorUninit[int64](2, 2, 2, 2, 2)
    for index in 0 ..< source.size: source.unsafe_raw_offset[index] = int64(index)
    check source.amin([0, 2, 4]) == [[0'i64, 2], [8, 10]].toTensor
    check source.amax([4, 0, 2]) == [[21'i64, 23], [29, 31]].toTensor
    var booleans = newTensor[bool](2, 2, 2, 2, 2)
    booleans[0, 0, 0, 0, 0] = true
    booleans[1, 0, 1, 1, 1] = true
    booleans[0, 1, 1, 1, 0] = true
    check booleans.amax([0, 2, 4]) == [[true, true], [false, true]].toTensor
    check booleans.amin([0, 2, 4]) == [[false, false], [false, false]].toTensor

  test "Scalars and empty outputs retain validation precedence without dtype promotion":
    let scalar = newTensorWith[float64]([], -0.0)
    expect IndexDefect: discard scalar.amin(1)
    expect ValueError: discard scalar.amax([0, -1])
    let emptyRows = newTensor[float32](0, 3)
    expect IndexDefect: discard emptyRows.amin([0, 0])
    expect ValueError: discard emptyRows.amax()
    check emptyRows.amin(1).shape == [0]
    check emptyRows.amax(1, keepdim = true).shape == [0, 1]
    let emptyMiddle = newTensor[int16](2, 0, 3)
    expect ValueError: discard emptyMiddle.amin([0, 0])
    expect IndexDefect: discard emptyMiddle.amax([0, 1])
    check emptyMiddle.amin([0, -1]).shape == [0]
    let unsigned = [1'u16, 2].toTensor
    expect ValueError: discard unsigned.amin()
    expect IndexDefect: discard unsigned.amax(2)
    let complexInput = [complex32(1, 2)].toTensor
    expect ValueError: discard complexInput.amax()
    check newTensor[uint16](0, 3).amin(1).shape == [0]
    check newTensor[Complex[float64]](2, 0, 3).amax([0, 2]).shape == [0]

  test "Integer extrema retain exact endpoints across SIMD and scalar tails":
    checkIntegerExtremaTails[int8]()
    checkIntegerExtremaTails[int16]()
    checkIntegerExtremaTails[int32]()
    checkIntegerExtremaTails[int64]()
    checkIntegerExtremaTails[uint8]()

  test "Float32 extrema preserve NaN propagation and native signed-zero folds":
    checkFloatingExtrema[float32]()

  test "Float64 extrema preserve NaN propagation and native signed-zero folds":
    checkFloatingExtrema[float64]()
