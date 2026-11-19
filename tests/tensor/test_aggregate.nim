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
import std / [unittest, math, complex]

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

    test "Explicit sum dtypes cast before arithmetic and retain unsigned values":
      template checkIntegerSum(T: typedesc) =
        block:
          let input = [T(1), T(2), T(4), T(8), T(16)].toTensor
          check input.sum(dtype = float32) == 31'f32
          check input.sum(dtype = float64) == 31.0
          check input.reshape(1, 5).sum(1, dtype = float32) == [[31'f32]].toTensor
          check input.reshape(5, 1).sum(0, dtype = float64) == [[31.0]].toTensor
      checkIntegerSum(int8)
      checkIntegerSum(int16)
      checkIntegerSum(int32)
      checkIntegerSum(int64)
      checkIntegerSum(uint8)
      checkIntegerSum(uint16)
      checkIntegerSum(uint32)
      checkIntegerSum(uint64)
      let flags = [true, false, true].toTensor
      check flags.sum(dtype = float32) == 2'f32
      check flags.reshape(3, 1).sum(0, dtype = float64) == [[2.0]].toTensor
      let upper = [high(uint64), 2'u64, 3'u64].toTensor
      check upper.sum == 4'i64
      check upper.sum(dtype = float64) == float64(high(uint64))
      check upper.reshape(3, 1).sum(0, dtype = float64) ==
            [[float64(high(uint64))]].toTensor
      let narrow = [16777217.0, -16777216.0].toTensor
      check narrow.sum(dtype = float32) == 0'f32
      check narrow.sum(dtype = float64) == 1.0
      check narrow.reshape(1, 2).sum(1, dtype = float32) == [[0'f32]].toTensor
      let narrowInteger = [16777217'i32, -16777216'i32].toTensor
      check narrowInteger.sum(dtype = float32) == 0'f32
      check narrowInteger.sum(dtype = float64) == 1.0
      let wideIntegers = [high(int64), low(int64)].toTensor
      check wideIntegers.sum(dtype = float64) == 0.0
      let widen = [16777216'f32, 1'f32, -16777216'f32].toTensor
      check widen.sum(dtype = float64) == 1.0

    test "Explicit sum dtypes retain strided, scalar and empty identities":
      let owner = [[1'i64, 2, 3], [4'i64, 5, 6]].toTensor
      check owner.transpose.sum(dtype = float32) == 21'f32
      check owner.transpose.sum(0, dtype = float64) == [[6.0, 15.0]].toTensor
      check owner.transpose.sum(1, dtype = float32) ==
            [[5'f32], [7'f32], [9'f32]].toTensor
      let reversed = owner[^1..0|-1, _]
      check reversed.sum(1, dtype = float64) == [[15.0], [6.0]].toTensor
      let offset = owner[1..1, _]
      check offset.sum(dtype = float64) == 15.0
      check offset.sum(1, dtype = float32) == [[15'f32]].toTensor
      let broadcast = owner[0..0, 0..0].broadcast(3, 5)
      check broadcast.sum(dtype = float32) == 15'f32
      check broadcast.sum(0, dtype = float64) == [[3.0, 3, 3, 3, 3]].toTensor
      var output = offset.sum(0, dtype = float64)
      output[0, 0] = 99.0
      check owner == [[1'i64, 2, 3], [4'i64, 5, 6]].toTensor
      check newTensor[int32](0).sum(dtype = float64) == 0.0
      check newTensor[bool](2, 0).sum(1, dtype = float32) == [[0'f32], [0'f32]].toTensor
      check newTensor[uint64](0, 2, 3).sum(1, dtype = float64).shape == [0, 1, 3]
      let scalar = newTensorWith[bool]([], true)
      check scalar.sum(dtype = float64) == 1.0
      for axis in [0, -1]:
        let value = scalar.sum(axis, dtype = float32)
        check value.shape.len == 0
        check value[] == 1'f32
      for axis in [1, -2]:
        expect IndexDefect: discard scalar.sum(axis, dtype = float32)
      check classify([Inf, -Inf].toTensor.sum(dtype = float32)) == fcNan
      check classify([NaN].toTensor.sum(dtype = float32)) == fcNan

    test "Converted reductions preserve packed cast order and SIMD tails":
      for length in [1, 3, 4, 7, 15, 16, 17, 33, 127, 129, 1023, 1025]:
        var owner = newTensor[int32](length + 2)
        var expected = 0'f32
        for index in 0 ..< length:
          owner[index + 1] = int32(index mod 13 - 6)
          expected += float32(owner[index + 1])
        let input = owner[1..length]
        check input.sum(dtype = float32) == expected
        check (input +. 6'i32).asType(uint32).sum(dtype = float32) ==
          expected + float32(6 * length)
        check input.asType(float64).sum(dtype = float32) == expected
      var narrow = newTensor[int32](16)
      narrow[0] = 16777217
      narrow[1] = -16777216
      check narrow.sum(dtype = float32) == 0'f32
      check narrow.asType(float64).sum(dtype = float32) == 0'f32
      var unsigned = newTensor[uint32](16)
      unsigned[0] = 16777217
      unsigned[1] = 1
      check unsigned.sum(dtype = float32) == 16777216'f32

    test "Converted byte columns handle offset, tail and bounded signed carries":
      template checkColumns(T: typedesc) =
        for rows in [1, 3, 9, 127, 128, 129]:
          for columns in [3, 7, 8, 9, 17]:
            var owner = newTensor[T](rows * columns + 2)
            for index in 0 ..< rows * columns:
              when T is bool: owner[index + 1] = index mod 3 != 0
              else: owner[index + 1] = int8(index mod 256 - 128)
            let input = owner[1..rows * columns].reshape(rows, columns)
            let summed = input.sum(0, dtype = float32)
            let averaged = input.mean(0, dtype = float32)
            check summed.shape == [1, columns]
            check averaged.shape == [1, columns]
            for column in 0 ..< columns:
              var expected = 0
              for row in 0 ..< rows:
                when T is bool: expected += ord(input[row, column])
                else: expected += int(input[row, column])
              check summed[0, column] == float32(expected)
              check averaged[0, column] == float32(expected) / float32(rows)
      checkColumns(bool)
      checkColumns(int8)

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
      expect ValueError: discard integers.mean
      expect ValueError: discard integers.mean(0)
      expect ValueError: discard integers.mean(1)
      check integers.mean(dtype = float64) == 2.5
      check integers.mean(0, dtype = float64) == [[-1.5, 2.5, 6.5]].toTensor
      check integers.mean(1, dtype = float32) == [[1'f32], [4'f32]].toTensor

    test "Explicit mean dtypes cast before arithmetic without signed reinterpretation":
      template checkIntegerMean(T: typedesc) =
        block:
          let input = [T(1), T(2), T(4), T(8), T(16)].toTensor
          expect ValueError: discard input.mean
          expect ValueError: discard input.mean(99)
          expect ValueError: discard newTensor[T](0).mean
          expect ValueError: discard newTensorWith[T]([], T(1)).mean
          check input.mean(dtype = float32) == 31'f32 / 5'f32
          check input.mean(dtype = float64) == 31.0 / 5.0
          check input.reshape(1, 5).mean(1, dtype = float32) ==
                [[31'f32 / 5'f32]].toTensor
          check input.reshape(5, 1).mean(0, dtype = float64) ==
                [[31.0 / 5.0]].toTensor
      checkIntegerMean(int8)
      checkIntegerMean(int16)
      checkIntegerMean(int32)
      checkIntegerMean(int64)
      checkIntegerMean(uint8)
      checkIntegerMean(uint16)
      checkIntegerMean(uint32)
      checkIntegerMean(uint64)
      let flags = [true, false, true, true, false].toTensor
      expect ValueError: discard flags.mean
      expect ValueError: discard flags.mean(-99)
      expect ValueError: discard newTensor[bool](0).mean
      expect ValueError: discard newTensorWith[bool]([], true).mean
      check flags.mean(dtype = float32) == 3'f32 / 5'f32
      check flags.reshape(5, 1).mean(0, dtype = float64) == [[0.6]].toTensor
      let upper = [high(uint64), 2'u64, 3'u64].toTensor
      check upper.mean(dtype = float64) == float64(high(uint64)) / 3.0
      check upper.reshape(3, 1).mean(0, dtype = float64) ==
            [[float64(high(uint64)) / 3.0]].toTensor
      let narrow = [16777217.0, -16777216.0].toTensor
      check narrow.mean(dtype = float32) == 0'f32
      check narrow.mean(dtype = float64) == 0.5
      check narrow.reshape(1, 2).mean(1, dtype = float32) == [[0'f32]].toTensor
      let widen = [16777216'f32, 1'f32, -16777216'f32].toTensor
      check widen.mean(dtype = float64) == 1.0 / 3.0

    test "Explicit mean dtypes retain view, empty and scalar contracts":
      let owner = [[1'i64, 2, 3], [4'i64, 5, 6]].toTensor
      check owner.transpose.mean(dtype = float32) == 3.5'f32
      check owner.transpose.mean(0, dtype = float64) == [[2.0, 5.0]].toTensor
      check owner.transpose.mean(1, dtype = float32) ==
            [[2.5'f32], [3.5'f32], [4.5'f32]].toTensor
      let reversed = owner[^1..0|-1, _]
      check reversed.mean(0, dtype = float64) == [[2.5, 3.5, 4.5]].toTensor
      check reversed.mean(1, dtype = float64) == [[5.0], [2.0]].toTensor
      let offset = owner[1..1, _]
      check offset.mean(dtype = float64) == 5.0
      check offset.mean(1, dtype = float32) == [[5'f32]].toTensor
      let broadcast = owner[0..0, 0..0].broadcast(3, 5)
      check broadcast.mean(dtype = float32) == 1'f32
      check broadcast.mean(0, dtype = float64) == [[1.0, 1, 1, 1, 1]].toTensor
      var output = offset.mean(0, dtype = float64)
      output[0, 0] = 99.0
      check owner == [[1'i64, 2, 3], [4'i64, 5, 6]].toTensor
      check classify(newTensor[int32](0).mean(dtype = float64)) == fcNan
      let empty = newTensor[bool](2, 0).mean(1, dtype = float32)
      check empty.shape == [2, 1]
      for value in empty: check classify(value) == fcNan
      check newTensor[uint64](0, 2, 3).mean(1, dtype = float64).shape == [0, 1, 3]
      let scalar = newTensorWith[bool]([], true)
      check scalar.mean(dtype = float64) == 1.0
      for axis in [0, -1]:
        let value = scalar.mean(axis, dtype = float32)
        check value.shape.len == 0
        check value[] == 1'f32
      for axis in [1, -2]:
        expect IndexDefect: discard scalar.mean(axis, dtype = float32)
      check classify([Inf, -Inf].toTensor.mean(dtype = float32)) == fcNan
      check classify([NaN].toTensor.mean(dtype = float32)) == fcNan

    test "Floating mean identities preserve positive zero before division":
      template checkMeanZero(T, U: typedesc) =
        block:
          let scalar = newTensorWith[T]([], T(-0.0))
          let pair = [T(-0.0), T(-0.0)].toTensor
          check scalar.mean(dtype = U).classify == fcZero
          check pair.mean(dtype = U).classify == fcZero
          for axis in [0, -1]:
            check scalar.mean(axis, dtype = U)[].classify == fcZero
            check pair.mean(axis, dtype = U)[0].classify == fcZero
      checkMeanZero(float32, float32)
      checkMeanZero(float32, float64)
      checkMeanZero(float64, float32)
      checkMeanZero(float64, float64)

    test "Converted float32 reductions bound long carries for packed and strided inputs":
      var input = newTensor[uint16](128, 128)
      let pattern = [65534'u16, 65535'u16, 1'u16, 2'u16]
      for row in 0 ..< 128:
        for column in 0 ..< 128:
          input[row, column] = pattern[column mod 4]
      check input.sum(dtype = float32) == 536870912'f32
      check input.transpose.sum(dtype = float32) == 536870912'f32
      check input[^1..0|-1, _].sum(dtype = float32) == 536870912'f32
      check input.mean(dtype = float32) == 32768'f32
      check input.transpose.mean(dtype = float32) == 32768'f32
      check input[^1..0|-1, _].mean(dtype = float32) == 32768'f32
      check input.asType(float64).transpose.mean(dtype = float32) == 32768'f32
      let broadcast = [[65535'u16]].toTensor.broadcast(128, 128)
      check broadcast.mean(dtype = float32) == 65535'f32
      check broadcast.mean(0, dtype = float32) == newTensorWith[float32]([1, 128], 65535'f32)
      let longColumns = newTensorWith[uint16]([16384, 2], 65535'u16)
      check longColumns.mean(0, dtype = float32) == [[65535'f32, 65535'f32]].toTensor
      check longColumns.sum(0, dtype = float32) ==
            [[65535'f32 * 16384'f32, 65535'f32 * 16384'f32]].toTensor

    test "Complex means cast before component summation and reciprocal rounding":
      let rounded = [complex32(1.9, 2), complex32(2.9, -3), complex32(0, 4)].toTensor
      check cast[uint32](rounded.mean.re) == 1070386382'u32
      check rounded.mean.im == 1'f32
      check cast[uint32](rounded.mean(0)[0].re) == 1070386382'u32
      let precise = [complex(16777217.0, 16777217.0),
                     complex(-16777216.0, -16777216.0)].toTensor
      check precise.mean(dtype = Complex[float32]) == complex32(0, 0)
      check precise.mean(dtype = Complex[float64]) == complex(0.5, 0.5)
      let wide = [complex(9007199254740992.0, 9007199254740992.0),
                  complex(1.0, 1.0), complex(-9007199254740992.0, -9007199254740992.0)].toTensor
      check wide.mean == complex(1.0 / 3.0, 1.0 / 3.0)
      check wide.mean(dtype = float64) == 0.0
      let columns = wide.reshape(3, 1).broadcast(3, 2).clone()
      check columns.mean(0) == [[complex(0.0), complex(0.0)]].toTensor
      check [true, false, true].toTensor.mean(dtype = Complex[float64]) == complex(2.0 / 3.0)
      check [high(uint64), high(uint64)].toTensor.mean(dtype = Complex[float64]).re > 1.8e19

    test "Complex means preserve scalar identities, empty shapes and nonfinite division":
      let negativeZero = cast[float64](0x8000000000000000'u64)
      let scalar = newTensorWith[Complex[float64]]([], complex(negativeZero, negativeZero))
      check scalar.mean.re.classify == fcZero
      check scalar.mean.im.classify == fcZero
      check scalar.mean(-1)[].re.classify == fcZero
      let empty = newTensor[Complex[float64]](2, 0, 3)
      check empty.mean.re.isNaN and empty.mean.im.isNaN
      let reduced = empty.mean(1, dtype = Complex[float32])
      check reduced.shape == [2, 1, 3]
      for value in reduced: check value.re.isNaN and value.im.isNaN
      check empty.mean(0).shape == [1, 0, 3]
      let nonfinite = [complex(7.0, NaN)].toTensor
      check nonfinite.mean.re.isNaN and nonfinite.mean.im.isNaN
      check nonfinite.mean(dtype = float32) == 7'f32
      expect IndexDefect: discard scalar.mean(1)
      expect ValueError: discard empty.mean(dtype = bool)
      expect ValueError: discard empty.mean(99, dtype = uint16)

    test "Complex mean dtype views retain signed strides and independent outputs":
      let owner = [complex(99.0, 1.0), complex(1.0, 1.0), complex(2.0, 2.0),
                   complex(3.0, 3.0), complex(99.0, 1.0)].toTensor
      let reversed = owner[3..1|-1]
      check reversed.mean(dtype = Complex[float32]) == complex32(2, 2)
      check reversed.mean(dtype = float64) == 2.0
      let broadcast = reversed.reshape(3, 1).broadcast(3, 5)
      var columns = broadcast.mean(0, dtype = Complex[float32])
      check columns.shape == [1, 5]
      for value in columns: check value == complex32(2, 2)
      columns[0, 0] = complex32(100, 100)
      check owner[1] == complex(1.0, 1.0)

    test "Complex sum dtypes cast before component arithmetic and numeric wrapping":
      let precise = [16777217.0, -16777216.0].toTensor
      check precise.sum(dtype = Complex[float32]) == complex32(0, 0)
      check precise.sum(dtype = Complex[float64]) == complex(1.0)
      let unsigned = [0x8000008000000001'u64].toTensor
      check cast[uint32](unsigned.sum(dtype = Complex[float32]).re) == 0x5f000001'u32
      check [true, false, true].toTensor.sum(dtype = Complex[float64]) == complex(2.0)
      let wide = [complex(9007199254740992.0, 9007199254740992.0),
                  complex(1.0, 1.0), complex(-9007199254740992.0, -9007199254740992.0)].toTensor
      check wide.sum == complex(1.0, 1.0)
      check wide.sum(dtype = float64) == 0.0
      check wide.sum(dtype = int64) == 1'i64
      let negative = [complex(-127.9, 1.0), complex(-128.9, -2.0), complex(-0.9, 4.0)].toTensor
      check negative.sum(dtype = int8) == 1'i8
      check negative.sum(dtype = int64) == -255'i64
      check [complex(200.9), complex(100.9)].toTensor.sum(dtype = uint8) == 44'u8
      check [complex(0.0, 4.0)].toTensor.sum(dtype = bool)

    test "Complex sums preserve cascade block and scalar-tail boundaries":
      var boundary = newTensor[Complex[float64]](129)
      boundary[0] = complex(9007199254740992.0, 9007199254740992.0)
      boundary[8] = complex(1.0, 1.0)
      boundary[16] = complex(-9007199254740992.0, -9007199254740992.0)
      boundary[128] = complex(1.0, 1.0)
      check boundary[0..127].sum == complex(0.0)
      check boundary.sum == complex(1.0, 1.0)
      check boundary[0..127].reshape(1, 128).sum(1) == [[complex(0.0)]].toTensor
      check boundary[0..127].mean == complex(0.0)
      check boundary.mean == complex(1.0 / 129.0, 1.0 / 129.0)

    test "Complex sums retain zero identities, empty shapes and nonfinite components":
      let negativeZero = cast[float64](0x8000000000000000'u64)
      let scalar = newTensorWith[Complex[float64]]([], complex(negativeZero, negativeZero))
      check cast[uint64](scalar.sum.re) == 0'u64
      check cast[uint64](scalar.sum.im) == 0'u64
      check cast[uint64](scalar.sum(-1)[].re) == 0'u64
      let empty = newTensor[Complex[float64]](2, 0, 3)
      check empty.sum == complex(0.0)
      check empty.sum(1, dtype = Complex[float32]) == newTensor[Complex[float32]](2, 1, 3)
      check empty.sum(0).shape == [1, 0, 3]
      check empty.sum(dtype = uint64) == 0'u64
      check empty.sum(1, dtype = uint16) == newTensor[uint16](2, 1, 3)
      let nonfinite = [complex(7.0, NaN)].toTensor
      check nonfinite.sum.re == 7.0 and nonfinite.sum.im.isNaN
      check nonfinite.sum(dtype = float32) == 7'f32
      expect ValueError: discard scalar.sum(dtype = uint16)
      expect ValueError: discard scalar.sum(0, dtype = uint64)
      expect IndexDefect: discard scalar.sum(99, dtype = uint16)

    test "Complex sum dtype views retain signed strides and independent outputs":
      let owner = [complex(99.0, 1.0), complex(1.0, 1.0), complex(2.0, 2.0),
                   complex(3.0, 3.0), complex(99.0, 1.0)].toTensor
      let reversed = owner[3..1|-1]
      check reversed.sum(dtype = Complex[float32]) == complex32(6, 6)
      check reversed.sum(dtype = int32) == 6'i32
      let broadcast = reversed.reshape(3, 1).broadcast(3, 5)
      var columns = broadcast.sum(0, dtype = Complex[float32])
      check columns == newTensorWith[Complex[float32]]([1, 5], complex32(6, 6))
      check broadcast.sum(1, dtype = float32) == [[15'f32], [10'f32], [5'f32]].toTensor
      let truth = [[complex(0.0), complex(0.0, 4.0), complex(0.0)],
                   [complex(0.0), complex(0.0), complex(2.0)]].toTensor
      check truth.sum(0, dtype = bool) == [[false, true, true]].toTensor
      check truth.sum(1, dtype = bool) == [[true], [true]].toTensor
      columns[0, 0] = complex32(100, 100)
      check owner[1] == complex(1.0, 1.0)

    test "Complex product dtypes cast before component arithmetic and wrapping":
      let precise = [16777217.0, 1.0].toTensor
      check precise.product(dtype = Complex[float32]) == complex32(16777216, 0)
      check precise.product(dtype = Complex[float64]) == complex(16777217.0)
      var narrowedColumns = newTensorWith[Complex[float64]]([2, 9],
        complex(1.0000000596046448))
      for column in 0 ..< 9:
        narrowedColumns[0, column] = complex(1.0000001192092896)
      check narrowedColumns.product(0, dtype = Complex[float32]) ==
        newTensorWith[Complex[float32]]([1, 9], complex32(1.0000001192092896, 0))
      let unsigned = [0x8000008000000001'u64].toTensor
      check cast[uint32](unsigned.product(dtype = Complex[float32]).re) == 0x5f000001'u32
      let rounded = [complex(3.9, 2.0), complex(100.0, -1.0)].toTensor
      check rounded.product(dtype = int8) == 44'i8
      check rounded.product(dtype = int64) == 300'i64
      check [complex(200.0), complex(2.0)].toTensor.product(dtype = uint8) == 144'u8
      check [complex(0.0, 1.0), complex(2.0)].toTensor.product(dtype = bool)
      check not [complex(0.0), complex(2.0)].toTensor.product(dtype = bool)
      check [complex(0.0, NaN), complex(2.0)].toTensor.product(dtype = bool)
      let negativeZero = cast[float32](0x80000000'u32)
      var truth = newTensorWith[Complex[float32]]([2, 9], complex32(1, 0))
      truth[0, 0] = complex32(negativeZero, negativeZero)
      truth[1, 8] = complex32(0, float32(NaN))
      check not truth.product(dtype = bool)
      check truth.product(0, dtype = bool) ==
        [[false, true, true, true, true, true, true, true, true]].toTensor
      check truth.product(1, dtype = bool) == [[false], [true]].toTensor
      check [true, true].toTensor.product(dtype = Complex[float32]) == complex32(1, 0)
      check [true, false].toTensor.product(dtype = Complex[float64]) == complex(0.0)
      let discarded = [complex(7.0, NaN), complex(2.0)].toTensor
      check discarded.product(dtype = float32) == 14'f32
      check discarded.product(dtype = float64) == 14.0

    test "Complex products fold packed vectors before scalar tails":
      when defined(arm64):
        var source = newTensorWith[Complex[float32]]([65], complex32(1, 0))
        source[0] = complex32(1e30, 0)
        source[1] = complex32(1e30, 0)
        source[2] = complex32(1e-30, 0)
        source[3] = complex32(1e-30, 0)
        let scalar = source[0..6].product
        check scalar.re.isNaN and scalar.im.isNaN
        for length in [8, 9, 17, 65]:
          check source[0..<length].product == complex32(1, 0)
        let tiny = [complex(1e-100), complex(1e-100), complex(1e-100),
                    complex(1e-100), complex(1e200)].toTensor
        check tiny.product == complex(0.0)
      var cycle = newTensorUninit[Complex[float64]](128, 128)
      for index in 0 ..< cycle.size:
        let position = index mod 17
        cycle[index div 128, index mod 128] =
          if position < 8: complex(1.0, 1.0)
          elif position < 16: complex(0.5, -0.5)
          else: complex(1.0)
      check cycle.product == complex(-2.0, 2.0)
      check cycle.product(dtype = Complex[float32]) == complex32(-2, 2)

    test "Complex products retain scalar zero signs and empty dtype exceptions":
      let negative = cast[float32](0x80000000'u32)
      let scalar = newTensorWith[Complex[float32]]([], complex32(negative, negative))
      let whole = scalar.product
      let axis = scalar.product(-1)
      check cast[uint32](whole.re) == 0'u32
      check cast[uint32](whole.im) == 0x80000000'u32
      check axis.shape.len == 0
      check cast[uint32](axis.atContiguousIndex(0).re) == 0'u32
      check cast[uint32](axis.atContiguousIndex(0).im) == 0x80000000'u32
      let empty = newTensor[Complex[float64]](2, 0, 3)
      check empty.product == complex(1.0)
      check empty.product(dtype = uint16) == 1'u16
      check empty.product(1, dtype = Complex[float32]) ==
        newTensorWith[Complex[float32]]([2, 1, 3], complex32(1, 0))
      check empty.product(1, dtype = uint64) == newTensorWith[uint64]([2, 1, 3], 1'u64)
      expect ValueError: discard scalar.product(dtype = uint16)
      expect ValueError: discard scalar.product(0, dtype = uint64)
      expect IndexDefect: discard scalar.product(2, dtype = uint16)

    test "Complex product column tiles preserve nonfinite and signed-zero identities":
      let source64 = [[complex64(Inf, 0), complex64(0, Inf),
        complex64(NaN, 1), complex64(1, NaN), complex64(-0.0, -0.0),
        complex64(-0.0, 0.0), complex64(0.0, -0.0), complex64(0.0, 0.0),
        complex64(2, -3)]].toTensor
      var source32 = newTensorUninit[Complex[float32]](1, 9)
      for column in 0..<9:
        source32[0, column] = complex32(float32(source64[0, column].re),
          float32(source64[0, column].im))
      template checkColumns(expression: untyped) =
        block:
          let observed = expression
          check observed.shape == [1, 9]
          check observed[0, 0].re == Inf and observed[0, 0].im.isNaN
          check observed[0, 1].re.isNaN and observed[0, 1].im == Inf
          for column in 2..3:
            check observed[0, column].re.isNaN and observed[0, column].im.isNaN
          check observed[0, 4].re.classify == fcZero
          check observed[0, 4].im.classify == fcNegZero
          check observed[0, 5].re.classify == fcNegZero
          check observed[0, 5].im.classify == fcZero
          for column in 6..7:
            check observed[0, column].re.classify == fcZero
            check observed[0, column].im.classify == fcZero
          check observed[0, 8].re == 2 and observed[0, 8].im == -3
      checkColumns(source64.product(0))
      checkColumns(source64.product(0, dtype = Complex[float32]))
      checkColumns(source32.product(0))
      checkColumns(source32.product(0, dtype = Complex[float64]))

    test "Complex product views preserve signed strides, truth and output independence":
      let owner = [complex(99.0, 1.0), complex(1.0, 1.0), complex(2.0, -1.0),
                   complex(3.0), complex(99.0, 1.0)].toTensor
      let reversed = owner[3..1|-1]
      check reversed.product(dtype = Complex[float32]) == complex32(9, 3)
      check reversed.product(dtype = int32) == 6'i32
      let broadcast = reversed.reshape(3, 1).broadcast(3, 5)
      var columns = broadcast.product(0, dtype = Complex[float32])
      check columns == newTensorWith[Complex[float32]]([1, 5], complex32(9, 3))
      check broadcast.product(1, dtype = float32) == [[243'f32], [32'f32], [1'f32]].toTensor
      let truth = [[complex(1.0), complex(0.0, 4.0), complex(0.0)],
                   [complex(2.0), complex(0.0, 1.0), complex(2.0)]].toTensor
      check truth.product(0, dtype = bool) == [[true, true, false]].toTensor
      check truth.product(1, dtype = bool) == [[false], [true]].toTensor
      columns[0, 0] = complex32(100, 100)
      check owner[1] == complex(1.0, 1.0)

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

      test "Explicit product dtypes cast before output-dtype arithmetic":
        let rounded = newTensorWith[float64](64, 1.0 + 1.0 / 16777216.0)
        check rounded.product(dtype = float32) == 1'f32
        check rounded.product(dtype = float64) > 1.000003
        check [1, 2, 3].toTensor.product(dtype = float64) == 6.0
        var upperRounding = newTensorWith[uint64](16, 1'u64)
        upperRounding[0] = (1'u64 shl 63) + (1'u64 shl 39) + 1
        check upperRounding.product(dtype = float32) == float32((1'u64 shl 63) + (1'u64 shl 40))
        let overflow = [1e20, 1e20, 1e-20].toTensor
        check overflow.product(dtype = float32).classify == fcInf
        check abs(overflow.product(dtype = float64) / 1e20 - 1.0) < 1e-14
        let underflow = [1e-30, 1e-30, 1e30].toTensor
        check underflow.product(dtype = float32) == 0'f32
        check underflow.product(dtype = float64) > 0
        let zero = [1e20, 1e20, 0.0].toTensor
        check zero.product(dtype = float32).classify == fcNan
        check zero.product(dtype = float64) == 0.0
        let upper = [high(uint64), high(uint64), 0'u64].toTensor
        check upper.product(dtype = float32).classify == fcNan
        check upper.product(dtype = float64) == 0.0
        check [0.0, 1e20, 1e20, 1e20].toTensor.product(dtype = float32) == 0'f32

      test "Converted byte products retain signs, offsets and packed tails":
        template checkBytes(T: typedesc) =
          block:
            var owner = newTensor[T](24)
            for index in 0 ..< owner.size:
              when T is int8:
                owner[index] = if index mod 5 == 0: T(-2)
                               elif index mod 3 == 0: T(3)
                               else: T(1)
              else:
                owner[index] = T(1 + index mod 3)
            when T is int8: owner[1] = low(int8)
            else: owner[1] = high(uint8)
            for count in 7 .. 21:
              let view = owner[1..count]
              var expected = 1'f32
              for value in view: expected *= float32(value)
              check view.product(dtype = float32) == expected
              check view.product(0, dtype = float32)[0] == expected
        checkBytes(int8)
        checkBytes(uint8)

      test "Products preserve lane folding and ordered column range":
        when defined(macosx) and defined(arm64):
          var lanes = newTensorWith[float64](16, 1.0)
          lanes[0] = 1e-20
          lanes[1] = 1e-20
          lanes[2] = 1e20
          lanes[3] = 1e20
          let expected = 0.9999946355819702'f32
          check lanes.product(dtype = float32) == expected
          check lanes.asType(float32).product == expected
          check lanes.product(0, dtype = float32)[0] == expected
        var columns = newTensorWith[float64]([4, 5], 1e200)
        for column in 0 ..< 5: columns[0, column] = 0.0
        check columns.product(0) == newTensor[float64](1, 5)
        check columns.product(0, dtype = float64) == newTensor[float64](1, 5)
        for value in columns.product(0, dtype = float32):
          check value.classify == fcNan

      test "Explicit products preserve signed views, empty identities and scalar signs":
        var owner = newTensor[uint32](107)
        for index in 0 ..< 105: owner[index + 1] = uint32(1 + index mod 2)
        let input = owner[1..105].reshape(3, 5, 7)
        for view in [input, input[_, _, 6..0|-1],
                     input[0..0, _, _].broadcast(3, 5, 7)]:
          for axis in 0 ..< 3:
            let output = view.product(axis - 3, dtype = float32)
            for i in 0 ..< output.shape[0]:
              for j in 0 ..< output.shape[1]:
                for k in 0 ..< output.shape[2]:
                  var coordinate = [i, j, k]
                  var expected = 1'f32
                  for position in 0 ..< view.shape[axis]:
                    coordinate[axis] = position
                    expected *= float32(view[coordinate[0], coordinate[1], coordinate[2]])
                  check output[i, j, k] == expected
        let scalar = newTensorWith[float64]([], -0.0)
        check scalar.product(dtype = float32).classify == fcNegZero
        for axis in [0, -1]:
          let output = scalar.product(axis, dtype = float32)
          check output.rank == 0
          check output.atContiguousIndex(0).classify == fcNegZero
        let empty = newTensor[bool](2, 0, 3)
        check empty.product(dtype = float64) == 1.0
        check empty.product(1, dtype = float32) == newTensorWith[float32]([2, 1, 3], 1'f32)
        check empty.product(0, dtype = float64).shape == [1, 0, 3]
        expect IndexDefect: discard scalar.product(1, dtype = float32)
        expect IndexDefect: discard empty.product(-4, dtype = float64)
        var output = input.product(0, dtype = float64)
        output[0, 0, 0] = 99
        check input[0, 0, 0] == 1

      test "Integer output dtypes truncate before arithmetic and wrap every prefix":
        let signed = [129'i64, -130, 3].toTensor
        check signed.sum(dtype = int8) == 2'i8
        check signed.product(dtype = int8) == 122'i8
        check signed.cumsum(dtype = int8) == [-127'i8, -1, 2].toTensor
        check signed.cumprod(dtype = int8) == [-127'i8, 126, 122].toTensor
        check signed.cumsum(dtype = uint8) == [129'u8, 255, 2].toTensor
        let upper = [high(uint64), high(uint64) - 1, 0'u64].toTensor
        check upper.sum(dtype = uint8) == 253'u8
        check upper.cumsum(dtype = int8) == [-1'i8, -3, -3].toTensor
        check upper.cumprod(dtype = uint8) == [255'u8, 2, 0].toTensor
        check [0.9, 0.9, -0.1].toTensor.sum(dtype = int8) == 0'i8
        check [1.9, -2.9, 3.9].toTensor.product(dtype = int16) == -6'i16
        check [127.9, -128.9, 1.9].toTensor.cumsum(dtype = int8) ==
          [127'i8, -1, 0].toTensor
        check [255.9, 0.9, 1.9].toTensor.cumsum(dtype = uint8) ==
          [255'u8, 255, 0].toTensor
        check [high(int64), 1'i64].toTensor.cumsum(dtype = int64) ==
          [high(int64), low(int64)].toTensor
        check [128'i64, 2, 3, 9].toTensor.cumprod(dtype = int8) ==
          [-128'i8, 0, 0, 0].toTensor
        check [9'i64, 3, 2, 128].toTensor.cumprod(dtype = uint8, reverse = true) ==
          [0'u8, 0, 0, 128].toTensor
        let edge = [[0.0, 0.0], [-2147483648.9, 2147483647.9]].toTensor
        check edge.sum(0, dtype = int32) == [[low(int32), high(int32)]].toTensor
        check edge.cumsum(0, dtype = int32) ==
          [[0'i32, 0], [low(int32), high(int32)]].toTensor

      test "Boolean reductions and unsupported integer kernels retain shape exceptions":
        check [0.0, 2.0, -3.0].toTensor.sum(dtype = bool)
        check not [0.0, 2.0, -3.0].toTensor.product(dtype = bool)
        check [NaN, Inf, -Inf].toTensor.product(dtype = bool)
        check not [-0.0].toTensor.sum(dtype = bool)
        template checkTruthBits(T, Word: typedesc) =
          block:
            var truth = newTensorWith[T](37, cast[T](Word(1)))
            check truth.product(dtype = bool)
            truth[4] = T(NaN)
            check truth.reshape(1, 37).product(1, dtype = bool)[0, 0]
            truth[36] = T(-0.0)
            check not truth.product(dtype = bool)
            truth[36] = T(Inf)
            truth[31] = T(0.0)
            check not truth.product(dtype = bool)
        checkTruthBits(float32, uint32)
        checkTruthBits(float64, uint64)
        let scalar = newTensorWith[float64]([], 3.9)
        let wide = newTensorWith[float64]([], 1.8e19).cumsum(dtype = uint64)
        check wide.rank == 0
        check wide.atContiguousIndex(0) == 18000000000000000000'u64
        check scalar.cumprod(-1, dtype = uint16).atContiguousIndex(0) == 3'u16
        check scalar.cumsum(dtype = bool).atContiguousIndex(0)
        let empty = newTensor[float64](2, 0, 3)
        check empty.sum(dtype = uint64) == 0'u64
        check empty.product(dtype = uint64) == 1'u64
        check empty.sum(1, dtype = uint32) == newTensor[uint32](2, 1, 3)
        check empty.product(1, dtype = bool) == newTensorWith[bool]([2, 1, 3], true)
        check empty.product(2, dtype = uint16).shape == [2, 0, 1]
        check empty.cumsum(1, dtype = bool).shape == [2, 0, 3]
        check empty.cumprod(-1, dtype = uint64).shape == [2, 0, 3]
        expect ValueError: discard scalar.sum(dtype = uint64)
        expect ValueError: discard scalar.product(-1, dtype = uint16)
        expect ValueError: discard [1].toTensor.cumsum(dtype = bool)
        expect ValueError: discard [1].toTensor.cumprod(dtype = uint32)
        expect ValueError: discard empty.cumsum(3, dtype = bool)
        expect ValueError: discard scalar.cumprod(1, dtype = uint64)
        expect ValueError: discard empty.mean(dtype = int8)
        expect ValueError: discard empty.mean(1, dtype = bool)
        expect IndexDefect: discard empty.sum(3, dtype = uint64)

      test "Integer output dtypes preserve signed views and independent scan storage":
        let owner = [0'i64, 129, -130, 3, 127, 2, -1, 0].toTensor
        let input = owner[1..6].reshape(2, 3)
        let reversed = input[_, 2..0|-1]
        check reversed.sum(1, dtype = int8) == [[2'i8], [-128'i8]].toTensor
        check reversed.cumsum(1, dtype = int8) ==
          [[3'i8, -127, 2], [-1'i8, 1, -128]].toTensor
        check input.cumprod(1, dtype = int8, reverse = true) ==
          [[122'i8, 122, 3], [2'i8, -2, -1]].toTensor
        check input.transpose.sum(0, dtype = int8) == [[2'i8, -128]].toTensor
        check input.transpose.product(0, dtype = int8) == [[122'i8, 2]].toTensor
        let broadcast = input[0..0, _].broadcast(2, 3)
        check broadcast.cumsum(0, dtype = int8) ==
          [[-127'i8, 126, 3], [2'i8, -4, 6]].toTensor
        let fractional = [[1.9, -2.9, 3.9], [2.9, 2.9, -1.9]].toTensor.transpose
        check fractional.cumprod(0, dtype = int8) ==
          [[1'i8, 2], [-2'i8, 4], [-6'i8, -4]].toTensor
        var output = input.cumsum(1, dtype = int8)
        output[0, 0] = 0
        check owner[1] == 129'i64

      test "Explicit scan dtypes cast before wide ordered carries":
        let narrow = [16777217'i32, -16777216'i32].toTensor
        check narrow.cumsum(dtype = float32) == [16777216'f32, 0].toTensor
        check narrow.cumsum(dtype = float64) == [16777217.0, 1].toTensor
        check narrow.asType(float64).cumsum(dtype = float32) ==
          [16777216'f32, 0].toTensor
        check [16777216'f32, 1, -16777216].toTensor.cumsum(dtype = float64) ==
          [16777216.0, 16777217, 1].toTensor
        let upper = [high(uint64), high(uint64)].toTensor
        check upper.cumsum(dtype = float64) ==
          [float64(high(uint64)), 2.0 * float64(high(uint64))].toTensor
        let overflow = [1e20, 1e20, 1e-20].toTensor.cumprod(dtype = float32)
        check overflow[1].classify == fcInf
        check overflow[2] == float32(1e20)
        let underflow = [1e-30, 1e-30, 1e30].toTensor.cumprod(dtype = float32)
        check underflow[1] == 0'f32
        check underflow[2] == float32(1e-30)

      test "Explicit scan dtypes preserve packed tails and signed or broadcast views":
        template checkScan(T, U: typedesc) =
          block:
            var owner = newTensor[T](107)
            for index in 0 ..< 105:
              when T is bool: owner[index + 1] = index mod 3 != 0
              elif T is SomeUnsignedInt: owner[index + 1] = T(index mod 7 + 1)
              elif T is SomeFloat: owner[index + 1] = T(float64(index mod 7 - 3) + 0.125)
              else: owner[index + 1] = T(index mod 7 - 3)
            let input = owner[1..105].reshape(3, 5, 7)
            for view in [input, input[_, _, 6..0|-1],
                         input[0..0, _, _].broadcast(3, 5, 7)]:
              for axis in 0 ..< 3:
                for reverse in [false, true]:
                  let summed = view.cumsum(axis - 3, dtype = U, reverse = reverse)
                  let multiplied = view.cumprod(axis - 3, dtype = U, reverse = reverse)
                  check summed.shape == [3, 5, 7]
                  check multiplied.shape == [3, 5, 7]
                  for i in 0 ..< 3:
                    for j in 0 ..< 5:
                      for k in 0 ..< 7:
                        var coordinate = [i, j, k]
                        let last = coordinate[axis]
                        var expectedSum = 0.0
                        var expectedProduct = 1.0
                        let count = if reverse: view.shape[axis] - last else: last + 1
                        for step in 0 ..< count:
                          coordinate[axis] = if reverse: view.shape[axis] - 1 - step else: step
                          when T is bool:
                            let value = float64(U(ord(view[coordinate[0], coordinate[1], coordinate[2]])))
                          else:
                            let value = float64(U(view[coordinate[0], coordinate[1], coordinate[2]]))
                          expectedSum += value
                          expectedProduct *= value
                        check summed[i, j, k] == U(expectedSum)
                        check multiplied[i, j, k] == U(expectedProduct)
        template checkType(T: typedesc) =
          checkScan(T, float32)
          checkScan(T, float64)
        checkType(bool)
        checkType(int8)
        checkType(int16)
        checkType(int32)
        checkType(int64)
        checkType(uint8)
        checkType(uint16)
        checkType(uint32)
        checkType(uint64)
        checkType(float32)
        checkType(float64)

      test "Explicit scan dtypes preserve scalar signs, empty axes and output independence":
        let scalar = newTensorWith[float64]([], -0.0)
        for axis in [0, -1]:
          let summed = scalar.cumsum(axis, dtype = float32)
          let multiplied = scalar.cumprod(axis, dtype = float32)
          check summed.rank == 0
          check multiplied.rank == 0
          check summed.atContiguousIndex(0).classify == fcNegZero
          check multiplied.atContiguousIndex(0).classify == fcNegZero
        check [-0.0].toTensor.cumsum(dtype = float32)[0].classify == fcZero
        check [-0.0].toTensor.cumprod(dtype = float32)[0].classify == fcNegZero
        for axis in 0 ..< 3:
          check newTensor[uint32](2, 0, 3).cumsum(axis, dtype = float64).shape == [2, 0, 3]
          check newTensor[bool](2, 0, 3).cumprod(axis, dtype = float32).shape == [2, 0, 3]
        expect ValueError: discard scalar.cumsum(1, dtype = float32)
        expect ValueError: discard newTensor[int8](2, 0).cumprod(-3, dtype = float64)
        let source = [1, 2, 3].toTensor
        var output = source.cumsum(dtype = float64)
        output[0] = 99
        check source[0] == 1

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

      test "Complex scan dtypes cast components before wide arithmetic":
        let precise = [complex(16777217.0, 16777217.0),
                       complex(-16777216.0, -16777216.0)].toTensor
        check precise.cumsum(dtype = Complex[float32])[1] == complex32(0, 0)
        check precise.cumsum(dtype = Complex[float64])[1] == complex(1.0, 1.0)
        let upper = [0x8000008000000001'u64, 0, 1].toTensor
        let upperPrefixes = upper.cumsum(dtype = Complex[float32])
        check cast[uint32](upperPrefixes[0].re) == 0x5f000001'u32
        check upperPrefixes[0].im == 0'f32
        check [true, false, true].toTensor.cumsum(dtype = Complex[float64]) ==
          [complex(1.0), complex(1.0), complex(2.0)].toTensor
        let fractional = [complex(127.9, 5.0), complex(-128.9, -3.0),
                          complex(3.9, 9.0)].toTensor
        check fractional.cumsum(dtype = int8) == [127'i8, -1, 2].toTensor
        check fractional.cumprod(dtype = int8) == [127'i8, -128, -128].toTensor
        let unsigned = [complex(255.9, 2.0), complex(1.9, -3.0),
                        complex(2.9, 4.0)].toTensor
        check unsigned.cumsum(dtype = uint8) == [255'u8, 0, 2].toTensor
        check unsigned.cumprod(dtype = uint8) == [255'u8, 255, 254].toTensor
        let narrowed = precise.cumsum(dtype = float32)
        check narrowed == [16777216'f32, 0].toTensor
        let columns = [complex(1.9, 2.0), complex(-2.9, 3.0), complex(3.9, -2.0),
                       complex(4.9, 9.0), complex(5.9, 7.0), complex(-6.9, 4.0)].toTensor.reshape(2, 3)
        check columns.cumsum(0, dtype = int64) == [[1'i64, -2, 3], [5, 3, -3]].toTensor
        check columns.cumprod(0, dtype = int64) == [[1'i64, -2, 3], [4, -10, -18]].toTensor
        check columns.cumsum(0, reverse = true, dtype = int64) ==
          [[5'i64, 3, -3], [4, 5, -6]].toTensor

      test "Complex scan casts retain scalar truth, signs and dtype exceptions":
        let imaginary = newTensorWith[Complex[float64]]([], complex(0.0, 4.0))
        check imaginary.cumsum(dtype = bool).atContiguousIndex(0)
        check imaginary.cumprod(dtype = uint64).atContiguousIndex(0) == 0'u64
        let negativeZero = cast[float64](0x8000000000000000'u64)
        let scalar = newTensorWith[Complex[float64]]([], complex(negativeZero, negativeZero))
        let scalarComplex = scalar.cumsum(dtype = Complex[float32]).atContiguousIndex(0)
        check scalarComplex.re.classify == fcNegZero
        check scalarComplex.im.classify == fcNegZero
        check scalar.cumprod(dtype = float32).atContiguousIndex(0).classify == fcNegZero
        let empty = newTensor[Complex[float64]](2, 0, 3)
        check empty.cumsum(1, dtype = bool).shape == [2, 0, 3]
        check empty.cumprod(-1, dtype = uint32).shape == [2, 0, 3]
        expect ValueError: discard [complex(1.0, 2.0)].toTensor.cumsum(dtype = bool)
        expect ValueError: discard [complex(1.0, 2.0)].toTensor.cumprod(dtype = uint16)
        expect ValueError: discard scalar.cumsum(1, dtype = Complex[float32])
        expect ValueError: discard empty.cumprod(3, dtype = float64)

      test "Complex scan dtype casts read signed views and keep output independent":
        let owner = [complex(99.0, 1.0), complex(1.0, 1.0), complex(2.0, 2.0),
                     complex(3.0, 3.0), complex(99.0, 1.0)].toTensor
        let reversed = owner[3..1|-1]
        check reversed.cumsum(dtype = Complex[float32]) ==
          [complex32(3, 3), complex32(5, 5), complex32(6, 6)].toTensor
        check reversed.cumprod(dtype = Complex[float32]) ==
          [complex32(3, 3), complex32(0, 12), complex32(-12, 12)].toTensor
        check reversed.cumsum(dtype = float64) == [3.0, 5, 6].toTensor
        check reversed.cumprod(dtype = int16) == [3'i16, 6, 6].toTensor
        let broadcast = reversed.reshape(3, 1).broadcast(3, 5)
        var columns = broadcast.cumsum(0, dtype = Complex[float32])
        for column in 0 ..< 5:
          check columns[2, column] == complex32(6, 6)
        columns[2, 0] = complex32(100, 100)
        check owner[1] == complex(1.0, 1.0)

      test "complex64 scans keep both components wide after rounded stores":
        let cancellation = [complex32(16777216, 16777216), complex32(1, 1),
                            complex32(-16777216, -16777216)].toTensor
        check cancellation.cumsum()[2] == complex32(1, 1)
        check cancellation.cumsum(reverse = true)[0] == complex32(1, 1)
        let overflow = [complex32(1e20, 0), complex32(1e20, 0),
                        complex32(1e-20, 0)].toTensor
        let large = overflow.cumprod()
        check large[1].re.classify == fcInf
        check large[2] == overflow[0]
        let underflow = [complex32(1e-30, 0), complex32(1e-30, 0),
                         complex32(1e30, 0)].toTensor
        let small = underflow.cumprod()
        check small[1].re == 0'f32
        check abs(small[2].re / underflow[0].re - 1'f32) < 1e-6'f32
        let columns = cancellation.reshape(3, 1).broadcast(3, 5)
        let sums = columns.cumsum(0)
        for column in 0 ..< 5:
          check sums[2, column] == complex32(1, 1)
        check cancellation[2..0|-1].cumsum()[2] == complex32(1, 1)

      test "complex scans preserve scalar signs and execute vector identities":
        let negativeZero = cast[float64](0x8000000000000000'u64)
        let value = complex(negativeZero, negativeZero)
        let scalar = newTensorWith[Complex[float64]]([], value)
        var copied = scalar.cumprod(-1)
        check copied.atContiguousIndex(0).re.classify == fcNegZero
        check copied.atContiguousIndex(0).im.classify == fcNegZero
        copied.unsafe_raw_offset[0] = complex(2.0, 3.0)
        check scalar.atContiguousIndex(0).re.classify == fcNegZero
        let vector = [value].toTensor
        let sum = vector.cumsum()[0]
        let product = vector.cumprod()[0]
        check sum.re.classify == fcZero
        check sum.im.classify == fcZero
        check product.re.classify == fcZero
        check product.im.classify == fcNegZero
        let infinite = [complex(Inf, 0.0)].toTensor.cumprod()[0]
        check infinite.re.classify == fcInf
        check infinite.im.classify == fcNan
        expect ValueError: discard scalar.cumsum(1)
        expect ValueError: discard vector.cumprod(-2)
        check newTensor[Complex[float32]](2, 0, 3).cumsum(1).shape == [2, 0, 3]

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
