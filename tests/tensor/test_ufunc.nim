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

proc main() =
  suite "Universal functions":
    test "As type with slicing":
      let a = [1, 2, 3, 4].toTensor()
      let b = a[1..2].asType(float)
      check b == [2.0'f64,3.0'f64].toTensor()

      let c = a[1..2].asType(Complex[float64])
      check c == [complex64(2.0'f64,0'f64), complex64(3.0'f64,0'f64)].toTensor()

    test "Common math functions are exported":
      let a = @[@[1.0,2,3],@[4.0,5,6]]
      let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]]
      # let c = @[@[complex64(1.0,0.0),complex64(2.0,0.0),complex64(3.0,0.0)],
      #           @[complex64(4.0,0.0),complex64(5.0,0.0),complex64(6.0,0.0)]]

      let ta = a.toTensor()
      let tb = b.toTensor()
      # let tc = c.toTensor()

      let expected_a = @[@[cos(1'f64),cos(2'f64),cos(3'f64)],@[cos(4'f64),cos(5'f64),cos(6'f64)]]
      let expected_b = @[@[ln(7'f64), ln(8'f64)],@[ln(9'f64), ln(10'f64)],@[ln(11'f64), ln(12'f64)]]
      # let expected_c = @[@[cos(complex64(1'f64,0.0)),cos(complex64(2'f64,0.0)),cos(complex64(3'f64,0.0))],
      #                    @[cos(complex64(4'f64,0.0)),cos(complex64(5'f64,0.0)),cos(complex64(6'f64,0.0))]]

      check: cos(ta).mean_absolute_error(expected_a.toTensor()) <= 1e-15 # We have slight precision loss with OpenMP
      check: ln(tb) == expected_b.toTensor()

      # TODO - complex errors
      # check: abs(mean_absolute_error(cos(tc), expected_c.toTensor())) <= 1e-15

    test "Creating custom universal functions is supported":
      proc square_plus_one(x: int): int = x ^ 2 + 1
      proc square_plus_one(x: Complex[float64]): Complex[float64] = x.pow(2) + 1
      makeUniversalLocal(square_plus_one)

      let c = @[@[2,4,8],@[3,9,27]]
      let tc = c.toTensor()
      let tcc = tc.asType(Complex[float64])

      let expected_c = @[@[5, 17, 65],@[10, 82, 730]]
      let expected_cc = @[@[complex64(5'f64,0.0), complex64(17'f64,0.0), complex64(65'f64,0.0)],
                          @[complex64(10'f64,0.0), complex64(82'f64,0.0), complex64(730'f64,0.0)]]

      check: square_plus_one(tc) == expected_c.toTensor()
      check: square_plus_one(tcc) == expected_cc.toTensor()

    ## MakeUniversal cannot change Tensor[B,T] to Tensor[B,U] for now
    ## map must be used instead
    test "Universal functions that change types are supported":
      let d = @[@[2,4,8],@[3,9,27]]
      let e = @[@["2","4","8"],@["3","9","27"]]

      proc stringify(n: int): string = $n
      # makeUniversalLocal(stringify)

      let td = d.toTensor()
      let te = e.toTensor()

      when compiles(td == te): check: false

      check: td.map(stringify) == te
      check: td.map(stringify)[0,1] == "4"

      when compileOption("boundChecks") and not defined(openmp):
        expect(IndexDefect):
          discard td.map(stringify)[1,3]
      else:
        echo "Bound-checking is disabled or OpenMP is used. The incorrect seq shape test has been skipped."

    test "Abs":
      let a = [-2,-1,0,1,2].toTensor()
      check abs(a) == [2,1,0,1,2].toTensor()
      let b = [-2.0,-1,0,1,2].toTensor()
      check abs(b) == [2.0,1,0,1,2].toTensor()
      # abs complex -> float
      let c = [-2.0,-1,0,1,2].toTensor().asType(Complex[float64])
      check abs(c) == [2.0,1,0,1,2].toTensor().asType(float64)

    test "Bulk unary operations preserve IEEE values and preallocated storage":
      let source = [-Inf.float32, -4'f32, -0'f32, 0'f32, 1'f32, Inf.float32, NaN.float32].toTensor
      template compareOperation(operation: static[UnaryMathOp]; reference: untyped) =
        block:
          var output = newTensor[float32](source.shape)
          let storage = output.storage
          output.applyUnary(source, operation)
          check output.storage == storage
          let expected = source.map_inline(reference)
          for i in 0 ..< source.size:
            let actual = output[i]
            let wanted = expected[i]
            if wanted != wanted:
              check actual != actual
            elif wanted == 0'f32:
              check cast[uint32](actual) == cast[uint32](wanted)
            elif abs(wanted) == Inf.float32:
              check actual == wanted
            else:
              check abs(actual - wanted) <= 1e-6'f32 * max(1'f32, abs(wanted))
      compareOperation(unaryExp, math.exp(x))
      compareOperation(unaryLn, math.ln(x))
      compareOperation(unarySqrt, math.sqrt(x))

    test "Square-root spans preserve subnormal lanes and vector tails":
      let values = [cast[float32](1'u32), cast[float32](0x007fffff'u32),
        cast[float32](0x00800000'u32), -0'f32, 0'f32, NaN.float32,
        2'f32, Inf.float32, 9'f32]
      var sourceOwner = newTensorWith[float32](values.len + 2, 99'f32)
      var outputOwner = newTensorWith[float32](values.len + 2, -99'f32)
      for index, value in values: sourceOwner[index + 1] = value
      for count in 1 .. values.len:
        let source = sourceOwner[1..count]
        var output = outputOwner[1..count]
        output.applyUnary(source, unarySqrt)
        check outputOwner[0] == -99'f32
        check outputOwner[count + 1] == -99'f32
        for index in 0 ..< count:
          let wanted = math.sqrt(values[index])
          let actual = output[index]
          if wanted != wanted:
            check actual != actual
          else:
            check cast[uint32](actual) == cast[uint32](wanted)
          check cast[uint32](source[index]) == cast[uint32](values[index])

    test "Bulk unary operations preserve views, aliases and scalar/empty shapes":
      let owner = [99'f32, 1, 4, 9, 16, 99].toTensor
      let reversed = owner[4..1|-1]
      check reversed.sqrt == [4'f32, 3, 2, 1].toTensor
      var outputOwner = newTensorWith[float32](6, -1'f32)
      var output = outputOwner[1..4]
      output.applyUnary(owner[1..4], unarySqrt)
      check outputOwner == [-1'f32, 1, 2, 3, 4, -1].toTensor
      var alias = owner[1..4]
      alias.applyUnary(alias, unarySqrt)
      check owner == [99'f32, 1, 2, 3, 4, 99].toTensor
      check [[1'f32, 4], [9'f32, 16]].toTensor.transpose.sqrt ==
        [[1'f32, 3], [2'f32, 4]].toTensor
      check [1'f32, 4].toTensor.reshape(1, 2).broadcast(3, 2).sqrt ==
        [[1'f32, 2], [1'f32, 2], [1'f32, 2]].toTensor
      check [1'f64, 4, 9, 16].toTensor[3..0|-1].sqrt == [4'f64, 3, 2, 1].toTensor
      var scalar = newTensorWith[float32]([], 4'f32)
      scalar.applyUnary(scalar, unarySqrt)
      check scalar.shape == []
      check scalar.item == 2'f32
      var empty = newTensor[float32](2, 0, 3)
      empty.applyUnary(empty, unaryExp)
      check empty.exp.shape == [2, 0, 3]
      expect ValueError:
        output.applyUnary(newTensor[float32](2, 2), unaryLn)

    test "As type with empty tensor returns empty tensor":
      let a = newTensor[int](0)
      let b = a.asType(float)
      check b.size() == 0

    test "Empty element-wise outputs preserve every axis":
      for shape in [@[0], @[2, 0, 3], @[0, 2, 1]]:
        let a = newTensor[float64](shape)
        for output in [a.exp, a.sqrt, a.cos, a.ln, a ^. 2.0, 2.0 ^. a,
                       a +. 2.0, 2.0 +. a, a -. 2.0, 2.0 -. a,
                       a *. 2.0, 2.0 *. a, a /. 2.0, 2.0 /. a,
                       a * 2.0, 2.0 * a, a / 2.0, a mod 2.0, 2.0 mod a]:
          check output.shape == shape
          check output.size == 0
        check a.asType(float32).shape == shape
        check a.asType(Complex[float64]).shape == shape
        check (a >. 2.0).shape == shape
        check a.isNaN.shape == shape
        check a.isNotNaN.shape == shape
        let integers = newTensor[int](shape)
        check (integers div 2).shape == shape
        check (integers / 2).shape == shape
      var sentinel: Tensor[float64]
      check sentinel.exp.shape == @[0]
      check (sentinel ^. 2.0).shape == @[0]

    test "Rank-zero element-wise outputs contain the scalar value":
      let a = newTensorWith[float64]([], 4.0)
      check a.sqrt.rank == 0
      check a.sqrt.item == 2.0
      check (a ^. 2.0).rank == 0
      check (a ^. 2.0).item == 16.0
      check (a +. 2.0).item == 6.0
      check a.asType(float32).item == 4.0'f32
      check a.asType(Complex[float64]).item == complex64(4.0, 0.0)
      check (a >. 2.0).rank == 0
      check (a >. 2.0).item


main()
GC_fullCollect()
