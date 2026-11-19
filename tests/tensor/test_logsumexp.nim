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

## Exercises logarithmic reduction stability, promotion and strided axis plans.
import std/[complex, math, unittest]
import ../../src/arraymancer

proc logScalar[T](value: T): Tensor[T] =
  result = newTensorUninit[T](newSeq[int]())
  result.unsafe_raw_offset[0] = value

proc checkStableReal[T: SomeFloat]() =
  let source = @[T(1000), T(999), T(-1000), T(-999)].toTensor.reshape(2, 2)
  let output = source.logsumexp(1)
  let correction = T(ln(T(1) + exp(T(-1))))
  check output.shape == @[2]
  check output.unsafe_raw_offset[0] == T(T(1000) + correction)
  check output.unsafe_raw_offset[1] == T(T(-999) + correction)
  let zero = logScalar(T(-0.0)).logsumexp()
  when T is float32: check cast[uint32](zero.unsafe_raw_offset[0]) == 0'u32
  else: check cast[uint64](zero.unsafe_raw_offset[0]) == 0'u64
  let infinity = @[T(Inf), T(1)].toTensor.logsumexp()
  check infinity.unsafe_raw_offset[0] == T(Inf)
  let negativeInfinity = @[T(-Inf), T(-Inf)].toTensor.logsumexp()
  check negativeInfinity.unsafe_raw_offset[0] == T(-Inf)
  let nan = @[T(NaN), T(1)].toTensor.logsumexp()
  check nan.unsafe_raw_offset[0] != nan.unsafe_raw_offset[0]

proc checkComplexBranches[F: SomeFloat]() =
  type C = Complex[F]
  let tolerance = when F is float32: 1e-6 else: 1e-12
  let scalar = logScalar(C(re: F(2), im: F(7))).logsumexp()
  check scalar.rank == 0
  check abs(float64(scalar.unsafe_raw_offset[0].re) - 2) <= tolerance
  check abs(float64(scalar.unsafe_raw_offset[0].im) - (7 - 2 * PI)) <= tolerance
  for imaginary in [F(Inf), F(-Inf), F(NaN)]:
    let negative = C(re: F(-Inf), im: imaginary)
    let emptyMass = logScalar(negative).logsumexp()
    check emptyMass.unsafe_raw_offset[0].re == F(-Inf)
    check emptyMass.unsafe_raw_offset[0].im == F(0)
    let finiteMass = @[negative, C(re: F(1), im: F(0))].toTensor.logsumexp()
    check finiteMass.unsafe_raw_offset[0].re == F(1)
    check finiteMass.unsafe_raw_offset[0].im == F(0)
    let positive = logScalar(C(re: F(Inf), im: imaginary)).logsumexp()
    check positive.unsafe_raw_offset[0].re == F(Inf)
    check positive.unsafe_raw_offset[0].im != positive.unsafe_raw_offset[0].im
  let realInfinity = logScalar(C(re: F(Inf), im: F(-0.0))).logsumexp()
  check realInfinity.unsafe_raw_offset[0].re == F(Inf)
  check realInfinity.unsafe_raw_offset[0].im == F(0)
  let empty = newTensor[C]([2, 0]).logsumexp(1)
  check empty.shape == @[2]
  for value in empty:
    check value.re == F(-Inf)
    check value.im == F(0)

proc checkColumnTails[T: SomeFloat]() =
  var input = newTensorUninit[T]([17, 33])
  for row in 0 ..< 17:
    for column in 0 ..< 33:
      input[row, column] = T(row - 8) + T(column) / T(16)
    input[row, 30] = T(-Inf)
  input[3, 0] = T(Inf)
  input[2, 31] = T(NaN)
  let output = input.logsumexp(0)
  check output.shape == @[33]
  check output.unsafe_raw_offset[0] == T(Inf)
  check output.unsafe_raw_offset[30] == T(-Inf)
  check output.unsafe_raw_offset[31] != output.unsafe_raw_offset[31]
  let tolerance = when T is float32: 1e-6 else: 1e-12
  for column in 1 ..< 33:
    if column in [30, 31]: continue
    let maximum = input[16, column]
    var total = 0.0
    for row in 0 ..< 17:
      total += float64(exp(T(input[row, column] - maximum)))
    let expected = T(ln(T(total)) + maximum)
    check abs(float64(output.unsafe_raw_offset[column] - expected)) <= tolerance

proc checkComplexTiles[F: SomeFloat]() =
  type C = Complex[F]
  let tolerance = when F is float32: 1e-6 else: 1e-12
  var input = newTensorUninit[C]([17, 33])
  for row in 0 ..< 17:
    for column in 0 ..< 33:
      input[row, column] = C(re: F(row) / F(16), im: F(column - 16) / F(64))
    input[row, 0] = C(re: F(-Inf), im: F(Inf))
    input[row, 1] = C(re: F(-Inf), im: F(NaN))
    input[row, 2] = C(re: F(Inf), im: F(Inf))
    input[row, 3] = C(re: F(Inf), im: F(-0.0))
    input[row, 4] = C(re: F(1), im: F(Inf))
  let output = input.logsumexp(0)
  for column in [0, 1]:
    check output.unsafe_raw_offset[column].re == F(-Inf)
    check output.unsafe_raw_offset[column].im == F(0)
  check output.unsafe_raw_offset[2].re == F(Inf)
  check output.unsafe_raw_offset[2].im != output.unsafe_raw_offset[2].im
  check output.unsafe_raw_offset[3].re == F(Inf)
  check output.unsafe_raw_offset[3].im == F(0)
  check output.unsafe_raw_offset[4].re != output.unsafe_raw_offset[4].re
  check output.unsafe_raw_offset[4].im != output.unsafe_raw_offset[4].im
  for column in 5 ..< 33:
    var total: Complex[float64]
    for row in 0 ..< 17:
      let value = input[row, column]
      let term = complex.exp(C(re: F(value.re - F(1)), im: value.im))
      total.re += float64(term.re)
      total.im += float64(term.im)
    let logarithm = complex.ln(C(re: F(total.re), im: F(total.im)))
    check abs(float64(output.unsafe_raw_offset[column].re - F(logarithm.re + F(1)))) <= tolerance
    check abs(float64(output.unsafe_raw_offset[column].im - logarithm.im)) <= tolerance
  let rowResults = input[_, 5..32].logsumexp(1)
  for row in 0 ..< 17:
    var total: Complex[float64]
    for column in 5 ..< 33:
      let term = complex.exp(C(re: F(0), im: input[row, column].im))
      total.re += float64(term.re)
      total.im += float64(term.im)
    let logarithm = complex.ln(C(re: F(total.re), im: F(total.im)))
    check abs(float64(rowResults.unsafe_raw_offset[row].re -
      F(logarithm.re + F(row) / F(16)))) <= tolerance
    check abs(float64(rowResults.unsafe_raw_offset[row].im - logarithm.im)) <= tolerance

suite "CPU stable logarithmic reductions":
  test "Output dtype promotion remains numeric for bool and high unsigned values":
    let boolean = @[false, true].toTensor.logsumexp()
    static: doAssert typeof(boolean) is Tensor[float32]
    check abs(boolean.unsafe_raw_offset[0] - ln(1'f32 + exp(1'f32))) < 1e-6'f32
    let unsigned = @[0'u64, high(uint64)].toTensor.logsumexp()
    static: doAssert typeof(unsigned) is Tensor[float32]
    check unsigned.unsafe_raw_offset[0] == float32(high(uint64))
    let signed = @[low(int64), high(int64)].toTensor.logsumexp()
    static: doAssert typeof(signed) is Tensor[float32]
    check signed.unsafe_raw_offset[0] == float32(high(int64))

  test "Stable real folds preserve large ranges, nonfinite classes and scalar zero":
    checkStableReal[float32]()
    checkStableReal[float64]()

  test "Nonadjacent axes, signed strides and retained dimensions use logical groups":
    var cube = newTensorUninit[float64]([2, 3, 4])
    for index in 0 ..< cube.size: cube.unsafe_raw_offset[index] = float64(index) / 8
    let output = cube.logsumexp([0, 2])
    let reversed = cube[_, _, ^1..0|-1].logsumexp([2, 0], keepdim = true)
    check output.shape == @[3]
    check reversed.shape == @[1, 3, 1]
    for middle in 0 ..< 3:
      let maximum = cube[1, middle, 3]
      var total = 0.0
      for outer in 0 ..< 2:
        for inner in 0 ..< 4: total += exp(cube[outer, middle, inner] - maximum)
      let expected = ln(total) + maximum
      check abs(output.unsafe_raw_offset[middle] - expected) < 1e-12
      check abs(reversed.unsafe_raw_offset[middle] - expected) < 1e-12
    check cube.logsumexp([1, 2]).shape == @[2]
    let selected = cube[_, _, 1..2]
    check abs(selected.logsumexp().unsafe_raw_offset[0] -
      selected.clone.logsumexp().unsafe_raw_offset[0]) < 1e-12

  test "Broadcast and offset reads return independent output storage":
    var base = @[@[1.0, 2.0, 3.0], @[4.0, 5.0, 6.0]].toTensor
    let broadcast = base[0..0, _].broadcast(4, 3)
    var output = broadcast.logsumexp(0)
    check output.shape == @[3]
    for index in 0 ..< 3:
      check abs(output.unsafe_raw_offset[index] - (base[0, index] + ln(4.0))) < 1e-12
    output.unsafe_raw_offset[0] = -10
    check base[0, 0] == 1
    let offset = base.transpose[_, 1..1]
    check abs(offset.logsumexp().unsafe_raw_offset[0] -
      @[4.0, 5.0, 6.0].toTensor.logsumexp().unsafe_raw_offset[0]) < 1e-12

  test "Empty identities and dimension validation survive empty outputs":
    let empty = newTensor[float32]([2, 0])
    check empty.logsumexp(0).shape == @[0]
    let reduced = empty.logsumexp(1, keepdim = true)
    check reduced.shape == @[2, 1]
    for value in reduced: check value == -Inf.float32
    check empty.logsumexp(newSeq[int]()).unsafe_raw_offset[0] == -Inf.float32
    expect ValueError: discard empty.logsumexp([0, -2])
    expect IndexDefect: discard empty.logsumexp(2)
    expect ValueError: discard @[1.0, 2.0].toTensor.logsumexp(newSeq[int]())
    let retained = @[@[1.0, 2.0], @[3.0, 4.0]].toTensor.logsumexp(newSeq[int](), keepdim = true)
    check retained.shape == @[1, 1]
    check logScalar(2.0).logsumexp(-1).rank == 0
    expect ValueError: discard logScalar(2.0).logsumexp([0, -1])

  test "Complex principal phases and IEEE infinite exponential branches survive stabilization":
    checkComplexBranches[float32]()
    checkComplexBranches[float64]()

  test "Column and exponential tile tails retain independent nonfinite groups":
    checkColumnTails[float32]()
    checkColumnTails[float64]()

  test "Complex vector tiles preserve IEEE branches and row/column tails":
    checkComplexTiles[float32]()
    checkComplexTiles[float64]()
