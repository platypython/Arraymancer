# Copyright 2017 the Arraymancer contributors
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

import ../../src/arraymancer
import std / [unittest, math]

proc checkPromotedIntegerScans[T]() =
  ## Check widening before arithmetic, including metadata-only input views.
  let input =
    when T is bool: [true, false, true].toTensor
    else: [T(100), T(2), T(3)].toTensor
  let sums: Tensor[int64] = input.cumsum()
  let products: Tensor[int64] = input.cumprod()
  when T is bool:
    check sums == [1'i64, 1, 2].toTensor
    check products == [1'i64, 0, 0].toTensor
  else:
    check sums == [100'i64, 102, 105].toTensor
    check products == [100'i64, 200, 600].toTensor
    check input[2..0|-1].cumsum() == [3'i64, 5, 105].toTensor
    check input[2..0|-1].cumprod() == [3'i64, 6, 600].toTensor
  let rows = input.reshape(1, 3).broadcast(2, 3)
  when T is bool:
    check rows.cumsum(0) == [[1'i64, 0, 1], [2'i64, 0, 2]].toTensor
    check rows.cumprod(0) == [[1'i64, 0, 1], [1'i64, 0, 1]].toTensor
  else:
    check rows.cumsum(0) == [[100'i64, 2, 3], [200'i64, 4, 6]].toTensor
    check rows.cumprod(0) == [[100'i64, 2, 3], [10000'i64, 4, 9]].toTensor
  let scalar = newTensorWith[T]([], input[0])
  var scalarSum: Tensor[int64] = scalar.cumsum(-1)
  let scalarProduct: Tensor[int64] = scalar.cumprod()
  check scalarSum.rank == 0
  check scalarProduct.rank == 0
  let first = when T is bool: 1'i64 else: 100'i64
  check scalarSum.atContiguousIndex(0) == first
  check scalarProduct.atContiguousIndex(0) == first
  scalarSum.unsafe_raw_offset[0] = -7
  check scalar.atContiguousIndex(0) == input[0]
  let empty = newTensor[T](2, 0, 3)
  let emptySum: Tensor[int64] = empty.cumsum(-2)
  let emptyProduct: Tensor[int64] = empty.cumprod(1)
  check emptySum.shape == [2, 0, 3]
  check emptyProduct.shape == [2, 0, 3]
  check emptySum.size == 0
  check emptyProduct.size == 0

proc checkPromotedIntegerReductions[T]() =
  ## Widen before arithmetic and read dense, signed and broadcast layouts alike.
  let input =
    when T is bool: [true, false, true, true, true].toTensor
    else: [T(100), T(100), T(3), T(2), T(1)].toTensor
  let expectedSum = when T is bool: 4'i64 else: 206'i64
  let expectedProduct = when T is bool: 0'i64 else: 60000'i64
  let total = input.sum
  let product = input.product
  check typeof(total) is int64
  check typeof(product) is int64
  check total == expectedSum
  check product == expectedProduct
  let axisSum: Tensor[int64] = input.sum(-1)
  let axisProduct: Tensor[int64] = input.product(0)
  check axisSum == [expectedSum].toTensor
  check axisProduct == [expectedProduct].toTensor
  let broadcasted = input.reshape(1, 5).broadcast(2, 5)
  for rows in [broadcasted, broadcasted.clone]:
    when T is bool:
      check rows.sum(0) == [[2'i64, 0, 2, 2, 2]].toTensor
      check rows.product(-2) == [[1'i64, 0, 1, 1, 1]].toTensor
    else:
      check rows.sum(0) == [[200'i64, 200, 6, 4, 2]].toTensor
      check rows.product(-2) == [[10000'i64, 10000, 9, 4, 1]].toTensor
    check rows.sum(-1) == [[expectedSum], [expectedSum]].toTensor
    check rows.product(1) == [[expectedProduct], [expectedProduct]].toTensor
    check rows.sum == expectedSum * 2
    check rows.product == expectedProduct * expectedProduct
    check rows.transpose.sum(0) == [[expectedSum, expectedSum]].toTensor
    check rows.transpose.product(0) == [[expectedProduct, expectedProduct]].toTensor
  let middleAxis = input.reshape(1, 5, 1).broadcast(2, 5, 3).clone
  check middleAxis.sum(1) == newTensorWith[int64]([2, 1, 3], expectedSum)
  check middleAxis.product(-2) == newTensorWith[int64]([2, 1, 3], expectedProduct)
  var owner = newTensorWith[T](7, input[0])
  for index in 0 ..< 5: owner[index + 1] = input[index]
  check owner[1..5].sum == expectedSum
  check owner[1..5].product == expectedProduct
  check owner[5..1|-1].sum == expectedSum
  check owner[5..1|-1].product == expectedProduct
  let scalar = newTensorWith[T]([], input[0])
  let first = when T is bool: 1'i64 else: 100'i64
  check scalar.sum == first
  check scalar.product == first
  var scalarSum: Tensor[int64] = scalar.sum(-1)
  let scalarProduct: Tensor[int64] = scalar.product(0)
  check scalarSum.rank == 0
  check scalarProduct.rank == 0
  check scalarSum.atContiguousIndex(0) == first
  check scalarProduct.atContiguousIndex(0) == first
  scalarSum.unsafe_raw_offset[0] = -7
  check scalar.atContiguousIndex(0) == input[0]
  let empty = newTensor[T](2, 0, 3)
  check empty.sum == 0'i64
  check empty.product == 1'i64
  check empty.sum(-2) == zeros[int64](2, 1, 3)
  check empty.product(1) == ones[int64](2, 1, 3)
  check newTensor[T](0, 2, 3).sum(1).shape == [0, 1, 3]
  check newTensor[T](0, 2, 3).product(1).shape == [0, 1, 3]
  expect IndexDefect: discard input.sum(1)
  expect IndexDefect: discard input.product(-2)


proc main() =
  suite "[Core] Testing aggregation functions":
    let t = [[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]].toTensor()
    let t_c = t.asType(Complex[float64])

    test "Empty reductions retain unreduced dimensions and neutral elements":
      let a = newTensor[float64](2, 0, 3)
      check a.sum == 0.0
      check a.product == 1.0
      check a.sum(-2) == zeros[float64](2, 1, 3)
      check a.product(-2) == ones[float64](2, 1, 3)
      let average = a.mean(-2)
      check average.shape == [2, 1, 3]
      for value in average:
        check value != value
      let outer = newTensor[float64](0, 2, 3)
      check outer.sum(1).shape == [0, 1, 3]
      check outer.product(1).shape == [0, 1, 3]
      check outer.mean(1).shape == [0, 1, 3]
      check newTensor[int](0).product == 1
      check newTensor[Complex[float64]](0).product == complex(1.0)
      check newTensor[Complex[float64]](2, 0).product(-1) ==
        ones[Complex[float64]](2, 1)
      expect ValueError:
        discard newTensor[int](2, 0).mean(1)

    test "Scalar axis reductions copy and strided negative axes reduce":
      let scalar = newTensorWith[float64]([], 3.0)
      for axis in [0, -1]:
        var total = scalar.sum(axis)
        check total.shape == []
        check total[] == 3.0
        check scalar.product(axis)[] == 3.0
        check scalar.mean(axis)[] == 3.0
        for value in total.mitems:
          value = 9.0
        check scalar[] == 3.0
      let view = [[1.0, 2, 3], [4.0, 5, 6]].toTensor.transpose[_, 1..^1]
      check view.sum(-2) == [[15.0]].toTensor
      check view.product(-2) == [[120.0]].toTensor
      check view.mean(-2) == [[5.0]].toTensor
      for axis in [-3, 2]:
        expect IndexDefect:
          discard view.sum(axis)
        expect IndexDefect:
          discard view.product(axis)
        expect IndexDefect:
          discard view.mean(axis)

    test "Float32 whole reductions preserve offsets, strides and empty means":
      let owner = [99'f32, 1, 2, 3, 4, 99].toTensor
      check owner[1..4].sum == 10'f32
      check owner[1..4].mean == 2.5'f32
      check owner[4..1|-1].sum == 10'f32
      check [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.transpose.sum == 21'f32
      check [1'f32, 2, 3].toTensor.reshape(1, 3).broadcast(2, 3).sum == 12'f32
      check newTensorWith[float32]([], 3'f32).sum == 3'f32
      check newTensor[float32](0).sum == 0'f32
      let emptyMean = newTensor[float32](0).mean
      check emptyMean != emptyMean

    test "Float32 packed reductions handle offset, SIMD tails and every axis":
      for rows in [1, 3, 4, 5, 17]:
        for columns in [1, 3, 4, 7, 15, 16, 17, 31, 32, 33]:
          let count = 2 * rows * columns
          var owner = newTensorWith[float32]([count + 2], 99'f32)
          let raw = owner.unsafe_raw_offset
          const values = [1'f32, 2, 0.5, -1, 0.5, 2, -1, 1]
          for index in 0 ..< count: raw[index + 1] = values[index mod values.len]
          let input = owner[1..count].reshape(2, rows, columns)
          let reference = input.asType(float64)
          # Pinned Torch CPU float32 lane products overflow on these balanced factors.
          case count
          of 544, 578, 1088, 1122: check input.product.classify == fcInf
          of 1054: check input.product.classify == fcNegInf
          else: check input.product == float32(reference.product)
          for axis in 0 ..< input.rank:
            let total = input.sum(axis)
            let product = input.product(axis)
            let average = input.mean(axis)
            let expectedTotal = reference.sum(axis).asType(float32)
            let expectedProduct = reference.product(axis).asType(float32)
            let expectedAverage = reference.mean(axis).asType(float32)
            check total.shape == expectedTotal.shape
            check product.shape == expectedProduct.shape
            check average.shape == expectedAverage.shape
            check (total - expectedTotal).abs.sum < 5e-4'f32
            check product == expectedProduct
            check (average - expectedAverage).abs.sum < 5e-4'f32

    test "Float32 packed products and sums preserve nonfinite values and signed zero":
      var input = ones[float32](33)
      input[1] = Inf.float32
      check input.product.classify == fcInf
      input[17] = 0'f32
      check input.product.classify == fcNan
      check input.reshape(1, 33).product(1)[0, 0].classify == fcNan
      input[1] = NaN.float32
      check input.sum(0)[0].classify == fcNan
      check input.mean(0)[0].classify == fcNan
      for value in input.mitems: value = 1'f32
      input[17] = -0'f32
      check input.product.classify == fcNegZero
      check input.reshape(1, 33).product(1)[0, 0].classify == fcNegZero

    test "Float64 packed reductions cover odd lanes, offsets, and axis means":
      let values = [
        1.0, 2, 3, 4, 5,  2, 1, 0.5, -1, 2,  3, -2, 1, 0, 1,
        -1, 1, 2, 3, 4,  0.5, 2, 1, -1, 3,  1, 2, 4, 2, -2
      ]
      var owner = newTensorWith[float64](values.len + 2, 99.0)
      for index, value in values:
        owner[index + 1] = value
      let source = owner[1..values.len].reshape(2, 3, 5)

      check source.sum == 44.0
      check source.mean == 44.0 / 30.0
      check source.product == 0.0
      check source.sum(0) == [
        [[0.0, 3, 5, 7, 9],
         [2.5, 3, 1.5, -2, 5],
         [4.0, 0, 5, 2, -1]]
      ].toTensor
      check source.sum(1) == [
        [[6.0, 1, 4.5, 3, 8]],
        [[0.5, 5, 7, 4, 5]]
      ].toTensor
      check source.sum(2) == [
        [[15.0], [4.5], [3.0]],
        [[9.0], [5.5], [7.0]]
      ].toTensor
      check source.product(0) == [
        [[-1.0, 2, 6, 12, 20],
         [1.0, 2, 0.5, 1, 6],
         [3.0, -4, 4, 0, -2]]
      ].toTensor
      check source.product(1) == [
        [[6.0, -4, 1.5, 0, 10]],
        [[-0.5, 4, 8, -6, -24]]
      ].toTensor
      check source.product(2) == [
        [[120.0], [-2.0], [0.0]],
        [[-24.0], [-3.0], [-32.0]]
      ].toTensor
      for axis in 0 ..< source.rank:
        let total = source.sum(axis)
        let average = source.mean(axis)
        check (average * float64(source.shape[axis]) - total).abs.sum < 1e-12
      check source.transpose.sum == 44.0
      check owner[values.len..1|-1].sum == 44.0
      let broadcast = [1.0, 2, 3].toTensor.reshape(1, 3).broadcast(2, 3)
      check broadcast.sum == 12.0
      check broadcast.product == 36.0
      check broadcast.sum(0) == [[2.0, 4, 6]].toTensor
      check broadcast.product(0) == [[1.0, 4, 9]].toTensor
      check broadcast.mean(0) == [[1.0, 2, 3]].toTensor


      let scalar = newTensorWith[float64]([], -3.0)
      check scalar.sum == -3.0
      check scalar.product == -3.0
      check scalar.mean == -3.0
      for axis in [0, -1]:
        check scalar.sum(axis)[] == -3.0
        check scalar.product(axis)[] == -3.0
        check scalar.mean(axis)[] == -3.0

      var nonfinite = ones[float64](5)
      nonfinite[1] = Inf
      check nonfinite.sum.classify == fcInf
      nonfinite[2] = 0.0
      check nonfinite.product.classify == fcNan
      check nonfinite.reshape(1, 5).product(1)[0, 0].classify == fcNan
      nonfinite[3] = NaN
      check nonfinite.mean.classify == fcNan

      var signedZero = ones[float64](5)
      signedZero[2] = cast[float64](0x8000000000000000'u64)
      check signedZero.product.classify == fcNegZero
      check signedZero.reshape(1, 5).product(1)[0, 0].classify == fcNegZero

    test "integer and boolean reductions promote before accumulating":
      checkPromotedIntegerReductions[bool]()
      checkPromotedIntegerReductions[int8]()
      checkPromotedIntegerReductions[int16]()
      checkPromotedIntegerReductions[int32]()
      checkPromotedIntegerReductions[int64]()
      checkPromotedIntegerReductions[int]()
      checkPromotedIntegerReductions[uint8]()
      checkPromotedIntegerReductions[uint16]()
      checkPromotedIntegerReductions[uint32]()
      checkPromotedIntegerReductions[uint64]()
      checkPromotedIntegerReductions[uint]()

    test "integer reductions wrap and preserve uint64 bit patterns":
      let signed = [high(int64), 2'i64, -3'i64, high(int64), -1'i64].toTensor
      check signed.sum == -4'i64
      check signed.product == 6'i64
      check signed.reshape(1, 5).sum(1) == [[-4'i64]].toTensor
      check signed.reshape(1, 5).product(-1) == [[6'i64]].toTensor
      let unsigned = [high(uint64), 1'u64, high(uint64) - 1].toTensor
      check unsigned.sum == -2'i64
      check unsigned.product == 2'i64
      check unsigned[2..0|-1].sum == -2'i64
      check unsigned[2..0|-1].product == 2'i64
      check unsigned.reshape(1, 3).sum(1) == [[-2'i64]].toTensor
      check unsigned.reshape(1, 3).product(1) == [[2'i64]].toTensor

    test "Sum":
      check: t.sum == 66
      check: t_c.sum == complex(66'f64)
      let row_sum = [[18, 22, 26]].toTensor()
      let col_sum = [[3],
                    [12],
                    [21],
                    [30]].toTensor()
      check: t.sum(axis=0) == row_sum.asType(int64)
      check: t.sum(axis=1) == col_sum.asType(int64)
      check: t_c.sum(axis=0) == row_sum.asType(Complex[float64])
      check: t_c.sum(axis=1) == col_sum.asType(Complex[float64])

    test "Mean":
      check: t.asType(float).mean == 5.5 # Note: may fail due to float rounding
      check: t_c.mean == complex(5.5) # Note: may fail due to float rounding

      let row_mean = [[4.5, 5.5, 6.5]].toTensor()
      let col_mean = [[1.0],
                      [4.0],
                      [7.0],
                      [10.0]].toTensor()
      check: t.asType(float).mean(axis=0) == row_mean
      check: t.asType(float).mean(axis=1) == col_mean
      check: t_c.mean(axis=0) == row_mean.asType(Complex[float64])
      check: t_c.mean(axis=1) == col_mean.asType(Complex[float64])
      let integers = [[-5'i8, 1, 7], [2'i8, 4, 6]].toTensor
      check integers.mean == 2'i8
      let integerRows: Tensor[int8] = integers.mean(0)
      let integerColumns: Tensor[int8] = integers.mean(1)
      check integerRows == [[-1'i8, 2, 6]].toTensor
      check integerColumns == [[1'i8], [4'i8]].toTensor

    test "Product":
      let a = [[1,2,4],[8,16,32]].toTensor()
      let a_c = a.asType(Complex[float64])
      check: t.product() == 0
      check: a.product() == 32768
      check: a.asType(float).product() == 32768.0
      check: a.product(0) == [[8'i64,32,128]].toTensor()
      check: a.product(1) == [[8'i64],[4096]].toTensor()
      check: t_c.product() == complex(0.0)
      check: a_c.product() == complex(32768.0)
      check: a_c.product(0) == [[8,32,128]].toTensor().asType(Complex[float64])
      check: a_c.product(1) == [[8],[4096]].toTensor().asType(Complex[float64])

    test "Min":
      let a = [2,-1,3,-3,5,0].toTensor()
      check: a.min() == -3
      check: a.asType(float32).min() == -3.0f

      let b = [[1,2,3,-4],[0,4,-2,5]].toTensor()
      check: b.min(0) == [[0,2,-2,-4]].toTensor()
      check: b.min(1) == [[-4],[-2]].toTensor()
      check: b.asType(float32).min(0) == [[0.0f,2,-2,-4]].toTensor()
      check: b.asType(float32).min(1) == [[-4.0f],[-2.0f]].toTensor()

    test "Max":
      let a = [2,-1,3,-3,5,0].toTensor()
      check: a.max() == 5
      check: a.asType(float32).max() == 5.0f

      let b = [[1,2,3,-4],[0,4,-2,5]].toTensor()
      check: b.max(0) == [[1,4,3,5]].toTensor()
      check: b.max(1) == [[3],[5]].toTensor()
      check: b.asType(float32).max(0) == [[1.0f,4,3,5]].toTensor()
      check: b.asType(float32).max(1) == [[3.0f],[5.0f]].toTensor()

    test "Variance":
      let a = [-3.0,-2,-1,0,1,2,3].toTensor()
      check: abs(a.variance() - 4.6666666666667) < 1e-8
      let b = [[1.0,2,3,-4],[0.0,4,-2,5]].toTensor()
      check: b.variance(0) == [[0.5,2.0,12.5,40.5]].toTensor()
      check: (
        b.variance(1) -
        [[9.666666666666666], [10.91666666666667]].toTensor()
      ).abs().sum() < 1e-8

    test "Standard Deviation":
      let a = [-3.0,-2,-1,0,1,2,3].toTensor()
      check: abs(a.std() - 2.1602468994693) < 1e-8
      let b = [[1.0,2,3,-4],[0.0,4,-2,5]].toTensor()
      check: abs(
        b.std(0) -
        [[0.7071067811865476,1.414213562373095,
          3.535533905932738,6.363961030678928]].toTensor()
      ).abs().sum() < 1e-8
      check: abs(
        b.std(1) -
        [[3.109126351029605],[3.304037933599835]].toTensor()
      ).abs().sum() < 1e-8

    test "Argmax":
      let a =  [[0, 4, 7],
                [1, 9, 5],
                [3, 4, 1]].toTensor
      check: argmax(a, 0) == [[2, 1, 0]].toTensor
      check: argmax(a, 1) == [[2],
                              [1],
                              [1]].toTensor

      block:
        let a =  [[0, 1, 2],
                  [3, 4, 5]].toTensor
        check: argmax(a, 0) == [[1, 1, 1]].toTensor
        check: argmax(a, 1) == [[2],
                                [2]].toTensor

    test "Argmax_3D":
      let a = [
        [[1, 10, 5, 5, 7, 3], [8, 3, 7, 9, 3, 8], [5, 3, 7, 1, 4, 5], [8, 10, 5, 8, 9, 1], [10, 5, 2, 1, 5, 8]],
        [[10, 0, 1, 9, 0, 4], [5, 7, 10, 0, 7, 5], [6, 1, 1, 10, 2, 2], [6, 10, 1, 9, 7, 8], [10, 7, 5, 9, 1, 3]],
        [[9, 1, 2, 1, 5, 10], [6, 1, 7, 9, 3, 0], [2, 1, 4, 8, 5, 7], [5, 7, 0, 4, 3, 2], [2, 7, 5, 8, 5, 6]],
        [[2, 8, 5, 9, 1, 5], [5, 10, 6, 8, 0, 1], [0, 10, 0, 8, 6, 7], [5, 1, 4, 9, 3, 0], [1, 1, 4, 3, 9, 4]]
      ].toTensor

      check: argmax(a, 0) == [
        [[1, 0, 0, 1, 0, 2], [0, 3, 1, 0, 1, 0], [1, 3, 0, 1, 3, 2], [0, 0, 0, 1, 0, 1], [0, 1, 1, 1, 3, 0]]
      ].toTensor

      check: argmax(a, 1) == [
        [[4, 0, 1, 1, 3, 1]], [[0, 3, 1, 2, 1, 3]], [[0, 3, 1, 1, 0, 0]], [[1, 1, 1, 0, 4, 2]]
      ].toTensor

      check: argmax(a, 2) == [
        [[1], [3], [2], [1], [0]], [[0], [2], [3], [1], [0]], [[5], [3], [3], [1], [3]], [[3], [1], [1], [3], [4]]
      ].toTensor

    block:
      let a = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
      ].toTensor

      check: argmax(a, 0) == [
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
      ].toTensor

      check: argmax(a, 1) == [
        [[2, 2, 2, 2]], [[2, 2, 2, 2]]
      ].toTensor

      check: argmax(a, 2) == [
        [[3], [3], [3]], [[3], [3], [3]]
      ].toTensor

    test "Argmin":
      let a =  [[0, 4, 7],
                [1, 9, 5],
                [3, 4, 1]].toTensor
      check: argmin(a, 0) == [[0, 0, 2]].toTensor
      check: argmin(a, 1) == [[0],
                              [0],
                              [2]].toTensor

      block:
        let a =  [[0, 1, 2],
                  [3, 4, 5]].toTensor
        check: argmin(a, 0) == [[0, 0, 0]].toTensor
        check: argmin(a, 1) == [[0],
                                [0]].toTensor

    test "Argmin_3D":
      let a = [
        [[1 , 10, 5, 5, 7, 3 ], [8, 3 , 7 , 9, 3, 8], [5, 3 , 7, 1 , 4, 5], [8, 10, 5, 8, 9, 1], [10, 5, 2, 1, 5, 8]],
        [[10, 0 , 1, 9, 0, 4 ], [5, 7 , 10, 0, 7, 5], [6, 1 , 1, 10, 2, 2], [6, 10, 1, 9, 7, 8], [10, 7, 5, 9, 1, 3]],
        [[9 , 1 , 2, 1, 5, 10], [6, 1 , 7 , 9, 3, 0], [2, 1 , 4, 8 , 5, 7], [5, 7 , 0, 4, 3, 2], [2 , 7, 5, 8, 5, 6]],
        [[2 , 8 , 5, 9, 1, 5 ], [5, 10, 6 , 8, 0, 1], [0, 10, 0, 8 , 6, 7], [5, 1 , 4, 9, 3, 0], [1 , 1, 4, 3, 9, 4]]
      ].toTensor

      check: argmin(a, 0) == [
        [[0, 1, 1, 2, 1, 0], [1, 2, 3, 1, 3, 2], [3, 1, 3, 0, 1, 1], [2, 3, 2, 2, 2, 3], [3, 3, 0, 0, 1, 1]]
      ].toTensor

      check: argmin(a, 1) == [
        [[0, 1, 4, 2, 1, 3]], [[1, 0, 0, 1, 0, 2]], [[2, 0, 3, 0, 1, 1]], [[2, 3, 2, 4, 1, 3]]
      ].toTensor

      check: argmin(a, 2) == [
        [[0], [1], [3], [5], [3]], [[1], [3], [1], [2], [4]], [[1], [5], [1], [2], [0]], [[4], [4], [0], [5], [0]]
      ].toTensor

    block:
      let a = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
      ].toTensor

      check: argmin(a, 0) == [
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
      ].toTensor

      check: argmin(a, 1) == [
        [[0, 0, 0, 0]], [[0, 0, 0, 0]]
      ].toTensor

      check: argmin(a, 2) == [
        [[0], [0], [0]], [[0], [0], [0]]
      ].toTensor

      test "cumsum":
        let a = [[0, 4, 7],
                 [1, 9, 5],
                 [3, 4, 1]].toTensor

        check: cumsum(a, 0) == [[0, 4,  7],
                                [1, 13, 12],
                                [4, 17, 13]].toTensor.asType(int64)

        check: cumsum(a, 1) == [[0, 4, 11],
                                [1, 10, 15],
                                [3, 7, 8]].toTensor.asType(int64)

      test "integer and boolean scans promote before accumulating":
        checkPromotedIntegerScans[bool]()
        checkPromotedIntegerScans[int8]()
        checkPromotedIntegerScans[int16]()
        checkPromotedIntegerScans[int32]()
        checkPromotedIntegerScans[int64]()
        checkPromotedIntegerScans[int]()
        checkPromotedIntegerScans[uint8]()
        checkPromotedIntegerScans[uint16]()
        checkPromotedIntegerScans[uint32]()
        checkPromotedIntegerScans[uint64]()
        checkPromotedIntegerScans[uint]()

      test "int64 scans wrap and uint64 inputs convert by bit pattern":
        check [high(int64), 1'i64, 1'i64].toTensor.cumsum() ==
          [high(int64), low(int64), low(int64) + 1].toTensor
        check [low(int64), -1'i64].toTensor.cumsum() ==
          [low(int64), high(int64)].toTensor
        check [high(int64), 2'i64, -1'i64].toTensor.cumprod() ==
          [high(int64), -2'i64, 2'i64].toTensor
        check [low(int64), -1'i64].toTensor.cumprod() ==
          [low(int64), low(int64)].toTensor
        let unsigned = [high(uint64), 1'u64, high(uint64) - 1].toTensor
        check unsigned.cumsum() == [-1'i64, 0, -2].toTensor
        check unsigned.cumprod() == [-1'i64, -1, 2].toTensor
        check newTensorWith[uint64]([], high(uint64)).cumsum().atContiguousIndex(0) == -1'i64

      test "cumsum handles floating precision, scalar signs, and axes":
        check [100_000_000'f32, 1'f32, -100_000_000'f32].toTensor.cumsum ==
          [100_000_000'f32, 100_000_000'f32, 1'f32].toTensor
        check [100_000_000.0, 1.0, -100_000_000.0].toTensor.cumsum ==
          [100_000_000.0, 100_000_001.0, 1.0].toTensor

        let negativeZero = cast[float64](0x8000000000000000'u64)
        let scalar = newTensorWith[float64]([], negativeZero)
        check scalar.cumsum().atContiguousIndex(0).classify == fcNegZero
        for value in [negativeZero, negativeZero, 0.0].toTensor.cumsum:
          check value.classify == fcZero

        let empty = newTensor[float32](2, 0, 3)
        let emptyResult = empty.cumsum(1)
        check emptyResult.shape == [2, 0, 3]
        check emptyResult.size == 0

        let matrix = [[0, 4, 7], [1, 9, 5], [3, 4, 1]].toTensor
        expect ValueError:
          discard matrix.cumsum(-3)
        expect ValueError:
          discard matrix.cumsum(2)
        expect ValueError:
          discard scalar.cumsum(1)

      test "float64 scans retain ordered forward and reverse carries":
        let cancellation = [1e16, 1.0, -1e16].toTensor
        check cancellation.cumsum() == [1e16, 1e16, 0.0].toTensor
        check cancellation.cumsum(reverse = true) ==
          [0.0, -1e16, -1e16].toTensor

        let factors = [1e200, 1e200, 1e-200].toTensor
        let forward = factors.cumprod()
        check forward[0] == 1e200
        check forward[1].classify == fcInf
        check forward[2].classify == fcInf
        let backward = factors.cumprod(reverse = true)
        check backward[0] == 1e200
        check backward[1] == 1.0
        check backward[2] == 1e-200

        let matrix = [[1.0, 2, 3], [4.0, 5, 6]].toTensor
        check matrix.cumsum(1, reverse = true) ==
          [[6.0, 5, 3], [15.0, 11, 6]].toTensor
        check matrix.cumprod(1, reverse = true) ==
          [[6.0, 6, 3], [120.0, 30, 6]].toTensor
        check matrix.cumsum(0, reverse = true) ==
          [[5.0, 7, 9], [4.0, 5, 6]].toTensor
        check matrix.cumprod(0, reverse = true) ==
          [[4.0, 10, 18], [4.0, 5, 6]].toTensor

        let reverseView = [99.0, 1, 2, 3, 99].toTensor[3..1|-1]
        check reverseView.cumsum(reverse = true) == [6.0, 3, 1].toTensor
        check reverseView.cumprod(reverse = true) == [6.0, 2, 1].toTensor
        let broadcast = [1.0, 2, 3].toTensor.reshape(1, 3).broadcast(2, 3)
        check broadcast.cumsum(0) == [[1.0, 2, 3], [2.0, 4, 6]].toTensor
        check broadcast.cumsum(0, reverse = true) ==
          [[2.0, 4, 6], [1.0, 2, 3]].toTensor
        check broadcast.cumprod(0, reverse = true) ==
          [[1.0, 4, 9], [1.0, 2, 3]].toTensor

        check newTensor[float64](2, 0, 3).cumsum(1, reverse = true).shape ==
          [2, 0, 3]
        check newTensor[float64](2, 0, 3).cumprod(1, reverse = true).size == 0

      test "cumsum reads offset, reversed, transposed, and broadcast views":
        let owner = [99, 1, 2, 3, 4, 99].toTensor
        check owner[4..1|-1].cumsum == [4, 7, 9, 10].toTensor.asType(int64)

        let transposed = [[1, 4], [2, 5], [3, 6]].toTensor.transpose
        check transposed.cumsum(0) == [[1, 2, 3], [5, 7, 9]].toTensor.asType(int64)

        let rows = [10, 20].toTensor.reshape(1, 2).broadcast(3, 2)
        check rows.cumsum(0) == [[10, 20], [20, 40], [30, 60]].toTensor.asType(int64)
      test "cumprod":
        let a = [[0, 4, 7],
                 [1, 9, 5],
                 [3, 4, 1]].toTensor

        check: cumprod(a, 0) == [[0, 4,   7],
                                 [0, 36,  35],
                                 [0, 144, 35]].toTensor.asType(int64)

        check: cumprod(a, 1) == [[0, 0, 0],
                                 [1, 9, 45],
                                 [3, 12, 12]].toTensor.asType(int64)

      test "cumprod keeps wide carries, scalar signs and empty shapes":
        let large = [1e20'f32, 1e20'f32, 1e-20'f32].toTensor
        let largePrefixes = large.cumprod()
        check largePrefixes[1].classify == fcInf
        check largePrefixes[2] == large[0]
        let small = [1e-30'f32, 1e-30'f32, 1e30'f32].toTensor
        let smallPrefixes = small.cumprod()
        check smallPrefixes[1] == 0'f32
        check smallPrefixes[2] == small[0]
        check [1e200, 1e200, 1e-200].toTensor.cumprod()[2].classify == fcInf

        let scalar = newTensorWith[float64]([], cast[float64](0x8000000000000000'u64))
        var scalarResult = scalar.cumprod(-1)
        check scalarResult.rank == 0
        check scalarResult.atContiguousIndex(0).classify == fcNegZero
        scalarResult.unsafe_raw_offset[0] = 2.0
        check scalar.atContiguousIndex(0).classify == fcNegZero
        for axis in 0 ..< 3:
          let empty = newTensor[float32](2, 0, 3).cumprod(axis)
          check empty.shape == [2, 0, 3]
          check empty.size == 0
        expect ValueError:
          discard scalar.cumprod(1)
        expect ValueError:
          discard newTensor[float32](2, 0, 3).cumprod(-4)

      test "cumprod handles packed SIMD tails and signed or broadcast views":
        var storage = newTensor[float32](195)
        let pattern = [1e20'f32, 1e20'f32, 1e-20'f32, -1'f32, 0.5'f32,
                       2'f32, cast[float32](0x80000000'u32)]
        for index in 0 ..< storage.size:
          storage[index] = pattern[index mod pattern.len]
        storage[87] = Inf.float32
        storage[88] = NegInf.float32
        storage[177] = NaN.float32
        let source = storage[3..191].reshape(3, 7, 9)
        for axis in 0 ..< 3:
          let actual = source.cumprod(axis - 3)
          let actual64 = source.asType(float64).cumprod(axis)
          check actual.shape == source.shape
          for i in 0 ..< 3:
            for j in 0 ..< 7:
              for k in 0 ..< 9:
                let coordinate = [i, j, k]
                var expected = 1.0
                for prefix in 0 .. coordinate[axis]:
                  var inputCoordinate = coordinate
                  inputCoordinate[axis] = prefix
                  expected *= float64(source[inputCoordinate[0], inputCoordinate[1],
                                             inputCoordinate[2]])
                if expected.classify == fcNan:
                  check actual[i, j, k].classify == fcNan
                  check actual64[i, j, k].classify == fcNan
                else:
                  check cast[uint32](actual[i, j, k]) == cast[uint32](float32(expected))
                  check cast[uint64](actual64[i, j, k]) == cast[uint64](expected)

        check [99, 1, 2, 3, 4, 99].toTensor[4..1|-1].cumprod() ==
          [4, 12, 24, 24].toTensor.asType(int64)
        check [[1, 4], [2, 5], [3, 6]].toTensor.transpose.cumprod(0) ==
          [[1, 2, 3], [4, 10, 18]].toTensor.asType(int64)
        check [2, -3].toTensor.reshape(1, 2).broadcast(3, 2).cumprod(0) ==
          [[2, -3], [4, 9], [8, -27]].toTensor.asType(int64)

    test "diff":
      let a = arange(12).reshape([3, 4])
      block:
        # Test diffs along the default axis
        let expected_diff1_axis1 = ones[int](3, 3)
        let expected_diff2_axis1 = zeros[int](3, 2)
        check a.diff_discrete(0) == a
        check a.diff_discrete == expected_diff1_axis1
        check a.diff_discrete(2) == expected_diff2_axis1
      block:
        # Test diffs along a different axis
        let expected_diff1_axis0 = 4 * ones[int](2, 4)
        let expected_diff2_axis0 = zeros[int](1, 4)
        check a.diff_discrete(0, axis=0) == a
        check a.diff_discrete(axis=0) == expected_diff1_axis0
        check a.diff_discrete(2, axis=0) == expected_diff2_axis0
      block:
        # Test boolean diffs
        let b = [true, true, false, false, true].toTensor
        let expected_bool_diff1 = [false, true, false, true].toTensor
        let expected_bool_diff2 = [true, true, true].toTensor
        check b.diff_discrete(0) == b
        check b.diff_discrete() == expected_bool_diff1
        check b.diff_discrete(2) == expected_bool_diff2

  test "unwrap_period":
    block:
      # Single dimension unwrap_period
      let phase_deg = (linspace(0, 720, 9) mod 360).asType(int) -. 180
      let expected = linspace(0, 720, 9).asType(int) -. 180
      check unwrap_period(phase_deg, period=360) == expected

      # Check that unwrap_period also works with floats
      check unwrap_period(phase_deg.asType(float), period=360.0) == expected.asType(float)

      let phase_rad = (linspace(0.0, 4*PI, 9).floorMod(2*PI)) -. PI
      let expected_rad = linspace(0.0, 4*PI, 9) -. PI
      check unwrap_period(phase_rad, period=2*PI) == expected_rad
      check unwrap_period(phase_rad) == expected_rad
    block:
      # Multiple dimension unwrap_period through axis
      let a = arange(0, 60*5, 5).reshape([3,4,5])
      let expected_axis0 = [[[ 0,  5, 10, 15, 20],
                             [25, 30, 35, 40, 45],
                             [50, 55, 60, 65, 70],
                             [75, 80, 85, 90, 95]],
                            [[-2,  3,  8, 13, 18],
                             [23, 28, 33, 38, 43],
                             [48, 53, 58, 63, 68],
                             [73, 78, 83, 88, 93]],
                            [[-4,  1,  6, 11, 16],
                             [21, 26, 31, 36, 41],
                             [46, 51, 56, 61, 66],
                             [71, 76, 81, 86, 91]]].toTensor
      let expected_axis2 = [[[ 0, -1, -2, -3, -4],
                             [25, 24, 23, 22, 21],
                             [50, 49, 48, 47, 46],
                             [75, 74, 73, 72, 71]],
                            [[100,  99,  98,  97,  96],
                             [125, 124, 123, 122, 121],
                             [150, 149, 148, 147, 146],
                             [175, 174, 173, 172, 171]],
                            [[200, 199, 198, 197, 196],
                             [225, 224, 223, 222, 221],
                             [250, 249, 248, 247, 246],
                             [275, 274, 273, 272, 271]]].toTensor
      check unwrap_period(a, axis=0, period=6) == expected_axis0
      check unwrap_period(a, axis=2, period=6) == expected_axis2
      # When the axis is not specified, the innermost axis is used
      check unwrap_period(a, period=6) == expected_axis2
      # Check that unwrap_period also works with floats
      check unwrap_period(a.asType(float), period=6.0) == expected_axis2.asType(float)
  suite "Testing logic functions":
    test "Nonzero":
      block:
        let a = [[3, 0, 0], [0, 4, 0], [5, 6, 0]].toTensor()
        let exp = [[0, 1, 2, 2], [0, 1, 0, 1]].toTensor
        check a.nonzero == exp

      block:
        let a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].toTensor
        let mask = a >. 3
        let exp = [[1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2]].toTensor
        check nonzero(mask) == exp

    test "all":
      let a = [[1, 2, 3], [-4, -5, -6]].toTensor
      let b = [[0, 2, 3], [4, 5, 6]].toTensor
      let ca = a.asType(Complex64)
      let cb = b.asType(Complex64)

      check:
        # Tensor[int]
        all(a) == true
        all(b) == false
        # Tensor[float]
        all(a.asType(float)) == true
        all(b.asType(float)) == false
        # Tensor[bool]
        all(a !=. 0) == true
        all(b >. 0) == false
        # Tensor[complex]
        all(ca) == true
        all(cb) == false

    test "any":
      let a = [0, 0, 1].toTensor
      let b = [0, 0, 0].toTensor
      let ca = a.asType(Complex64)
      let cb = b.asType(Complex64)

      check:
        # Tensor[int]
        any(a) == true
        any(b) == false
        # Tensor[float]
        any(a.asType(float)) == true
        any(b.asType(float)) == false
        # Tensor[bool]
        any(a !=. 0) == true
        any(b >. 0) == false
        # Tensor[complex]
        any(ca) == true
        any(cb) == false

main()
GC_fullCollect()
