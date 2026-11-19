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

## Exercises rounded-forward VJPs, complex conjugation and logical seed mapping.
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

proc checkRoundedForward[F: SomeFloat]() =
  ## Differentiating an unrounded softmax misses the pinned large-input result.
  let context = newContext Tensor[F]
  let source = context.variable([F(1000), F(999), F(-1000)].toTensor, requires_grad = true)
  source.logsumexp().backprop(newTensorWith[F]([], F(2)))
  when F is float32:
    closeValue(source.grad[0], F(1.4621598720550537))
    closeValue(source.grad[1], F(0.537898600101471))
  else:
    closeValue(source.grad[0], F(1.4621171572600336))
    closeValue(source.grad[1], F(0.537882842739999))
  check source.grad[2].classify == fcZero

  let nonfinite = context.variable([[F(Inf), F(1)], [F(-Inf), F(-Inf)]].toTensor,
    requires_grad = true)
  nonfinite.logsumexp(-1).backprop([F(-0.0), F(2)].toTensor)
  check nonfinite.grad[0, 0].classify == fcNan
  check nonfinite.grad[0, 1].classify == fcNegZero
  check nonfinite.grad[1, 0].classify == fcNan
  check nonfinite.grad[1, 1].classify == fcNan

proc checkLogicalMapping[T: SomeFloat or Complex]() =
  ## Nonadjacent axes and signed seed/input views cannot use flat broadcast indices.
  var storage = newTensorUninit[T](2, 3, 129)
  for index in 0 ..< storage.size:
    when T is Complex:
      type F = typeof(default(T).re)
      storage.unsafe_raw_offset[index] = T(re: F(index mod 7) / F(8), im: F(index mod 11) / F(32))
    else:
      storage.unsafe_raw_offset[index] = T(index mod 7) / T(8)
  let input = storage[^1..0|-1, _, ^1..0|-1]
  var seedStorage = newTensorUninit[T](3)
  for index in 0 ..< 3:
    when T is Complex:
      type F = typeof(default(T).re)
      seedStorage.unsafe_raw_offset[index] = T(re: F(index + 1), im: F(-0.5))
    else: seedStorage.unsafe_raw_offset[index] = T(index + 1)
  let context = newContext Tensor[T]
  for keepdim in [false, true]:
    let source = context.variable(input, requires_grad = true)
    let output = source.logsumexp([2, 0], keepdim)
    let seed = if keepdim: seedStorage[^1..0|-1].reshape(1, 3, 1)
               else: seedStorage[^1..0|-1]
    output.backprop(seed)
    check source.grad.shape == input.shape
    check source.grad.is_C_contiguous
    for outer in 0 ..< 2:
      for row in 0 ..< 3:
        let normalizer = output.value.atContiguousIndex(row)
        let weight = seedStorage[2 - row]
        for column in 0 ..< 129:
          let difference = input[outer, row, column] - normalizer
          let expected =
            when T is Complex: weight * complex.exp(difference).conjugate
            else: weight * exp(difference)
          closeValue(source.grad[outer, row, column], expected)
    source.grad[0, 0, 0] = default(T)
    check input[0, 0, 0] == storage[1, 0, 128]
    check seedStorage[0] != default(T)

  let broadcast = storage[0..0, 0..0, _].broadcast(2, 3, 129)
  let source = context.variable(broadcast, requires_grad = true)
  let output = source.logsumexp(1, keepdim = true)
  let seed = newTensorWith[T]([2, 1, 129], seedStorage[0])
  output.backprop(seed)
  for outer in 0 ..< 2:
    for row in 0 ..< 3:
      for column in 0 ..< 129:
        let difference = broadcast[outer, row, column] - output.value[outer, 0, column]
        let expected =
          when T is Complex: seedStorage[0] * complex.exp(difference).conjugate
          else: seedStorage[0] * exp(difference)
        closeValue(source.grad[outer, row, column], expected)

proc checkComplexSigns[F: SomeFloat]() =
  ## Componentwise negative-zero identities preserve the first complex VJP signs.
  type C = Complex[F]
  let values = [[C(re: F(0), im: F(0.25)), C(re: F(1), im: F(-0.5)), C(re: F(-2), im: F(7))],
                [C(re: F(3), im: F(1)), C(re: F(-1), im: F(-2)), C(re: F(2), im: F(0))]].toTensor
  let context = newContext Tensor[C]
  let source = context.variable(values, requires_grad = true)
  source.logsumexp(-1).backprop(newTensorWith[C]([2], C(re: F(-0.0), im: F(0))))
  let realSigns = [fcZero, fcNegZero, fcZero, fcZero, fcZero, fcNegZero]
  let imaginarySigns = [fcZero, fcZero, fcZero, fcZero, fcNegZero, fcZero]
  for index in 0 ..< 6:
    check source.grad.atContiguousIndex(index).re.classify == realSigns[index]
    check source.grad.atContiguousIndex(index).im.classify == imaginarySigns[index]

proc checkGraphContracts[T: SomeFloat or Complex]() =
  ## Retained traversals accumulate leaves without corrupting other tapes or seeds.
  let context = newContext Tensor[T]
  let source = context.variable(newTensorWith[T]([], default(T)), requires_grad = true)
  let output = source.logsumexp(-1)
  var unit: T
  when T is Complex:
    type F = typeof(default(T).re)
    unit = T(re: F(1), im: F(0))
  else: unit = T(1)
  let seed = newTensorWith[T]([], unit)
  expect ValueError: output.backprop(newTensorWith[T]([1], unit))
  output.backprop(seed, retain_graph = true)
  output.backprop(seed)
  closeValue(source.grad.atContiguousIndex(0), unit + unit)
  check seed.atContiguousIndex(0) == unit
  expect ValueError: output.backprop(seed)
  let empty = context.variable(newTensor[T](2, 0, 3), requires_grad = true)
  empty.logsumexp(1).backprop(newTensorWith[T]([2, 3], unit))
  check empty.grad.shape == [2, 0, 3]
  check empty.grad.size == 0
  var detached: Variable[Tensor[T]]
  context.no_grad_mode: detached = source.logsumexp()
  check not detached.requires_grad
  expect ValueError: detached.backprop(seed)
  expect ValueError: discard context.variable(newTensorWith[T]([2], unit), requires_grad = true).logsumexp([], false)
  expect ValueError: discard source.logsumexp([0, -1])
  expect IndexDefect: discard source.logsumexp(1)

suite "CPU logarithmic seeded gradients":
  test "Float32 rounded output and nonfinite derivative arithmetic": checkRoundedForward[float32]()
  test "Float64 rounded output and nonfinite derivative arithmetic": checkRoundedForward[float64]()
  test "Float32 nonadjacent and broadcast gradient coordinates": checkLogicalMapping[float32]()
  test "Float64 nonadjacent and broadcast gradient coordinates": checkLogicalMapping[float64]()
  test "Complex64 conjugation and nonadjacent gradient coordinates": checkLogicalMapping[Complex[float32]]()
  test "Complex128 conjugation and nonadjacent gradient coordinates": checkLogicalMapping[Complex[float64]]()
  test "Complex64 first-gradient component zero signs": checkComplexSigns[float32]()
  test "Complex128 first-gradient component zero signs": checkComplexSigns[float64]()
  test "Real graph, scalar, seed-shape and empty contracts":
    checkGraphContracts[float32]()
    checkGraphContracts[float64]()
  test "Complex graph, scalar, seed-shape and empty contracts":
    checkGraphContracts[Complex[float32]]()
    checkGraphContracts[Complex[float64]]()
