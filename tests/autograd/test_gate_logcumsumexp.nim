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

## Exercises rounded reverse prefixes, complex phases, view mapping and tape ownership.
import ../../src/arraymancer
import std/[complex, math, unittest]

proc closeValue[F: SomeFloat](actual, expected: F) =
  let tolerance = when F is float32: 1e-6 else: 1e-12
  if expected != expected: check actual != actual
  elif abs(expected) == F(Inf): check actual == expected
  else: check abs(float64(actual) - float64(expected)) <= tolerance * max(1.0, abs(float64(expected)))

proc closeValue[F: SomeFloat](actual, expected: Complex[F]) =
  closeValue(actual.re, expected.re)
  closeValue(actual.im, expected.im)

proc gradient[T: SomeFloat or Complex](source, seed: Tensor[T]; axis: int): Tensor[T] =
  let context = newContext Tensor[T]
  let input = context.variable(source, requires_grad = true)
  input.logcumsumexp(axis).backprop(seed)
  input.grad

proc checkReal[F: SomeFloat]() =
  ## The float32 fixture differs visibly from an unrounded reverse-softmax formula.
  let source = [F(1000), F(999), F(-1000)].toTensor
  let actual = gradient(source, newTensorWith[F]([3], F(2)), 0)
  when F is float32:
    let expected = [F(4.924576282501221), F(1.075859785079956), F(0)]
  else:
    let expected = [F(4.924234314519506), F(1.0757656854798796), F(0)]
  for index in 0 ..< 3: closeValue(actual[index], expected[index])
  let finite = [F(0), F(1), F(-2)].toTensor
  let nanSeed = gradient(finite, newTensorWith[F]([3], F(NaN)), 0)
  for value in nanSeed: check value.classify == fcZero
  let infinite = gradient(newTensorWith[F]([3], F(-Inf)), newTensor[F](3), 0)
  for value in infinite: check value.classify == fcNan

proc checkComplex[F: SomeFloat]() =
  ## Principal-log seed phases and conjugation survive the continuous forward phase.
  type C = Complex[F]
  let source = [C(re: F(2), im: F(7)), C(re: F(1), im: F(-8)),
                C(re: F(-2), im: F(0.5))].toTensor
  let actual = gradient(source, newTensorWith[C]([3], C(re: F(2), im: F(-0.5))), 0)
  when F is float32:
    let reals = [6.534970760345459, -0.5806584358215332, 0.04568593576550484]
    let imaginaries = [-3.370687961578369, 1.8871705532073975, -0.01648484542965889]
  else:
    let reals = [6.534971619988172, -0.580657556439131, 0.04568593645095871]
    let imaginaries = [-3.3706860644754597, 1.8871709147385354, -0.01648485026307507]
  for index in 0 ..< 3:
    closeValue(actual[index], C(re: F(reals[index]), im: F(imaginaries[index])))
  let zeros = gradient(source, newTensor[C](3), 0)
  for index in 0 ..< 3:
    check zeros[index].re.classify == (if index == 1: fcNegZero else: fcZero)
    check zeros[index].im.classify == (if index == 1: fcZero else: fcNegZero)

proc viewValue[T: SomeFloat or Complex](index: int): T =
  when T is Complex:
    type F = typeof(default(T).re)
    T(re: F(index mod 7 - 3) / F(8), im: F(index mod 11 - 5) / F(32))
  else: T(index mod 7 - 3) / T(8)

proc checkViews[T: SomeFloat or Complex]() =
  ## Independent packed comparisons isolate signed input and seed address mapping.
  var storage = newTensorUninit[T](3, 4, 33)
  for index in 0 ..< storage.size: storage.unsafe_raw_offset[index] = viewValue[T](index)
  let views = [storage[^1..0|-1, _, ^1..0|-1], storage[1..2, _, 1..32],
               storage[0..0, _, _].broadcast(3, 4, 33), storage.transpose]
  for view in views:
    var seedShape = view.shape
    seedShape[^1] *= 2
    var backing = newTensorUninit[T](seedShape)
    for index in 0 ..< backing.size:
      backing.unsafe_raw_offset[index] = viewValue[T](index + 3)
    let seed = backing[_, _, ^2..0|-2]
    let original = view.clone
    let originalSeed = seed.clone
    for axis in 0 ..< view.rank:
      var actual = gradient(view, seed, axis - view.rank)
      let expected = gradient(original, originalSeed, axis)
      check actual.shape == view.shape
      check actual.is_C_contiguous
      for index in 0 ..< actual.size:
        closeValue(actual.atContiguousIndex(index), expected.atContiguousIndex(index))
      actual.unsafe_raw_offset[0] = default(T)
      check view.atContiguousIndex(0) == original.atContiguousIndex(0)
      check seed.atContiguousIndex(0) == originalSeed.atContiguousIndex(0)

proc checkGraph[T: SomeFloat or Complex]() =
  ## Scalar VJPs copy seeds even through NaN values; retained tapes add contributions.
  var unit, negativeZero, nan: T
  when T is Complex:
    type F = typeof(default(T).re)
    unit = T(re: F(1), im: F(0))
    negativeZero = T(re: F(-0.0), im: F(-0.0))
    nan = T(re: F(NaN), im: F(NaN))
  else:
    unit = T(1)
    negativeZero = T(-0.0)
    nan = T(NaN)
  let scalarGradient = gradient(newTensorWith[T]([], nan), newTensorWith[T]([], negativeZero), -1)
  when T is Complex:
    check scalarGradient.atContiguousIndex(0).re.classify == fcNegZero
    check scalarGradient.atContiguousIndex(0).im.classify == fcNegZero
  else: check scalarGradient.atContiguousIndex(0).classify == fcNegZero
  let context = newContext Tensor[T]
  let input = context.variable(newTensorWith[T]([], nan), requires_grad = true)
  let output = input.logcumsumexp(-1)
  let seed = newTensorWith[T]([], unit)
  expect ValueError: output.backprop(newTensorWith[T]([1], unit))
  output.backprop(seed, retain_graph = true)
  output.backprop(seed)
  closeValue(input.grad.atContiguousIndex(0), unit + unit)
  check seed.atContiguousIndex(0) == unit
  expect ValueError: output.backprop(seed)
  let empty = context.variable(newTensor[T](2, 0, 3), requires_grad = true)
  empty.logcumsumexp(1).backprop(newTensor[T](2, 0, 3))
  check empty.grad.shape == [2, 0, 3]
  check empty.grad.size == 0
  var detached: Variable[Tensor[T]]
  context.no_grad_mode: detached = input.logcumsumexp()
  check not detached.requires_grad
  expect ValueError: detached.backprop(seed)
  expect IndexDefect: discard input.logcumsumexp(1)

suite "CPU logarithmic cumulative seeded gradients":
  test "Float32 rounded reverse prefixes and nonfinite seeds": checkReal[float32]()
  test "Float64 rounded reverse prefixes and nonfinite seeds": checkReal[float64]()
  test "Float32 rounded exponential channels preserve cancellation bits":
    var source = newTensorUninit[float32](128)
    var seed = newTensorUninit[float32](128)
    for index in 0 ..< 128:
      let original = index + 1536
      source[index] = if index == 0: 4'f32
                      elif index == 127: -4'f32
                      else: float32(original mod 63 - 31) / 8'f32
      seed[index] = float32(original mod 5 - 2)
    let actual = gradient(source, seed, 0)
    check actual.shape == [128]
    check actual[0] == -0.0218505859375'f32
  test "Complex64 phase-sensitive gradients and exact zero signs": checkComplex[float32]()
  test "Complex128 phase-sensitive gradients and exact zero signs": checkComplex[float64]()
  test "Float32 signed input and seed coordinates": checkViews[float32]()
  test "Float64 signed input and seed coordinates": checkViews[float64]()
  test "Complex64 signed input and seed coordinates": checkViews[Complex[float32]]()
  test "Complex128 signed input and seed coordinates": checkViews[Complex[float64]]()
  test "Real scalar, retained graph, seed-shape and empty contracts":
    checkGraph[float32]()
    checkGraph[float64]()
  test "Complex scalar, retained graph, seed-shape and empty contracts":
    checkGraph[Complex[float32]]()
    checkGraph[Complex[float64]]()
