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

## Exercises inclusive scan precision, strided inputs and seeded reverse-scan VJPs.
import ../../src/arraymancer
import std / [unittest, math]

proc checkBasicCumsumVjp[T: SomeFloat]() =
  let ctx = newContext Tensor[T]
  let x = ctx.variable([1.T, 2, 3].toTensor, requires_grad = true)
  let output = x.cumsum(axis = -1)
  check output.value == [1.T, 3, 6].toTensor
  output.backprop([2.T, -1, 3].toTensor)
  check x.grad == [4.T, 2, 3].toTensor

proc checkStridedCumsumVjp[T: SomeFloat]() =
  let context = newContext Tensor[T]
  let owner = arange(0, 120).asType(T).reshape(4, 5, 6)
  let view = owner[1..3, _, 1..5|2]
  let x = context.variable(view, requires_grad = true)
  let output = x.cumsum(axis = -2)
  let expected = [[[T(31), 33, 35], [68, 72, 76], [111, 117, 123],
                   [160, 168, 176], [215, 225, 235]],
                  [[T(61), 63, 65], [128, 132, 136], [201, 207, 213],
                   [280, 288, 296], [365, 375, 385]],
                  [[T(91), 93, 95], [188, 192, 196], [291, 297, 303],
                   [400, 408, 416], [515, 525, 535]]].toTensor
  check output.value == expected

  let seedBase = arange(1, 10).asType(T).reshape(3, 1, 3)
  let seed = seedBase.broadcast(3, 5, 3)
  output.backprop(seed)
  let expectedGradient = [[[T(5), 10, 15], [4, 8, 12], [3, 6, 9],
                           [2, 4, 6], [1, 2, 3]],
                          [[T(20), 25, 30], [16, 20, 24], [12, 15, 18],
                           [8, 10, 12], [4, 5, 6]],
                          [[T(35), 40, 45], [28, 32, 36], [21, 24, 27],
                           [14, 16, 18], [7, 8, 9]]].toTensor
  check x.grad == expectedGradient

proc checkNonfiniteCumsumVjp() =
  let context = newContext Tensor[float64]
  let x = context.variable([Inf, NegInf, 2.0].toTensor, requires_grad = true)
  let output = x.cumsum()
  check output.value[0].classify == fcInf
  check output.value[1].classify == fcNan
  check output.value[2].classify == fcNan
  output.backprop([2.0, -1.0, 3.0].toTensor)
  check x.grad == [4.0, 2.0, 3.0].toTensor

proc checkCumsumScalarEmptyAndInvalidAxes() =
  block:
    let context = newContext Tensor[float64]
    let x = context.variable(newTensorWith[float64]([], 3.0), requires_grad = true)
    let output = x.cumsum(axis = -1)
    check output.value.rank == 0
    check output.value.atContiguousIndex(0) == 3.0
    output.backprop(newTensorWith[float64]([], 4.0))
    check x.grad.rank == 0
    check x.grad.atContiguousIndex(0) == 4.0
    expect ValueError:
      discard x.cumsum(1)

  block:
    let context = newContext Tensor[float64]
    let x = context.variable(newTensor[float64](2, 0, 3), requires_grad = true)
    let output = x.cumsum(1)
    check output.value.shape == [2, 0, 3]
    check output.value.size == 0
    output.backprop(zeros[float64](2, 0, 3))
    check x.grad.shape == [2, 0, 3]
    check x.grad.size == 0

proc checkCumsumGraphModes() =
  let context = newContext Tensor[float64]
  let x = context.variable([1.0, 2, 3].toTensor, requires_grad = true)
  let output = x.cumsum()
  output.backprop([2.0, -1, 3].toTensor, retain_graph = true)
  output.backprop([1.0, 1, 1].toTensor, retain_graph = true)
  check x.grad == [7.0, 4, 4].toTensor

  var noGradOutput: Variable[Tensor[float64]]
  context.no_grad_mode:
    noGradOutput = x.cumsum()
  check not noGradOutput.requires_grad


proc checkPackedCumsumPrecision() =
  ## Independent double-precision scans expose lane mixing and rounded carries.
  var storage = newTensor[float32](195)
  let pattern = [1e8'f32, 1, -1e8, 3, -0.0, 2, -2]
  for index in 0 ..< storage.size:
    storage[index] = pattern[index mod pattern.len]
  storage[87] = Inf.float32
  storage[88] = NegInf.float32
  storage[177] = NaN.float32
  let source = storage[3..191].reshape(3, 7, 9)
  let reference = source.asType(float64)
  for axis in 0 ..< 3:
    let context = newContext Tensor[float32]
    let referenceContext = newContext Tensor[float64]
    let x = context.variable(source, requires_grad = true)
    let referenceX = referenceContext.variable(reference, requires_grad = true)
    let output = x.cumsum(axis)
    let referenceOutput = referenceX.cumsum(axis)
    output.backprop(source)
    referenceOutput.backprop(reference)
    for index in 0 ..< source.size:
      for pair in [(output.value.atContiguousIndex(index),
                    float32(referenceOutput.value.atContiguousIndex(index))),
                   (x.grad.atContiguousIndex(index),
                    float32(referenceX.grad.atContiguousIndex(index)))]:
        let (actual, expected) = pair
        if expected.classify == fcNan:
          check actual.classify == fcNan
        else:
          check cast[uint32](actual) == cast[uint32](expected)

proc main() =
  suite "Autograd cumulative sum":
    test "Reverse inclusive scan computes the seeded VJP":
      checkBasicCumsumVjp[float32]()
      checkBasicCumsumVjp[float64]()

    test "3D strided inputs and broadcast seeds preserve VJP values":
      checkStridedCumsumVjp[float32]()
      checkStridedCumsumVjp[float64]()

    test "Packed scan lanes preserve wide carries, offsets, tails and nonfinite values":
      checkPackedCumsumPrecision()

    test "Backward is independent of non-finite forward inputs":
      checkNonfiniteCumsumVjp()

    test "Scalar, empty, and invalid axes retain their semantics":
      checkCumsumScalarEmptyAndInvalidAxes()

    test "Retained backward accumulates and no-grad disables tracking":
      checkCumsumGraphModes()

main()
