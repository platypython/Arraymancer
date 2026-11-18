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

## Exercises rounded prefixes, first-zero masks and seeded cumulative-product VJPs.
import ../../src/arraymancer
import std/[math, unittest]

proc checkFiniteViews[T: SomeFloat]() =
  ## O(n²) omitted products are independent of the optimized finite-input masks.
  var owner = newTensor[T](3, 4, 6)
  for index in 0 ..< owner.size:
    owner.unsafe_raw_offset[index] = T(index mod 5 - 2)
  let view = owner[1..2, 3..1|-1, 0..4|2]
  let repeated = [[[T(2), T(-1), T(0)]]].toTensor.broadcast(2, 3, 3)
  let seed = [T(-2), T(1), T(3)].toTensor.reshape(1, 1, 3).broadcast(2, 3, 3)
  for source in [view, view.clone, repeated]:
    for axis in 0 ..< 3:
      let context = newContext Tensor[T]
      let x = context.variable(source, requires_grad = true)
      let output = x.cumprod(axis - 3)
      output.backprop(seed)
      check output.value.shape == source.shape
      check x.grad.shape == source.shape
      for i in 0 ..< 2:
        for j in 0 ..< 3:
          for k in 0 ..< 3:
            let coordinate = [i, j, k]
            var expectedPrefix = 1.0
            for prefix in 0 .. coordinate[axis]:
              var position = coordinate
              position[axis] = prefix
              expectedPrefix *= float64(source[position[0], position[1], position[2]])
            check output.value[i, j, k] == T(expectedPrefix)
            var expectedGradient = 0.0
            for outputPosition in coordinate[axis] ..< source.shape[axis]:
              var derivative = 1.0
              for prefix in 0 .. outputPosition:
                if prefix == coordinate[axis]: continue
                var position = coordinate
                position[axis] = prefix
                derivative *= float64(source[position[0], position[1], position[2]])
              var position = coordinate
              position[axis] = outputPosition
              expectedGradient += derivative * float64(seed[position[0], position[1], position[2]])
            check x.grad[i, j, k] == T(expectedGradient)

proc checkNonfiniteMasks[T: SomeFloat]() =
  ## Masking seeds does not erase Inf*0 in the carried first-zero product.
  for sample in [
      (@[T(2), T(Inf), T(3)], @[T(1), T(2), T(3)], @[T(Inf), T(NaN), T(Inf)]),
      (@[T(0), T(Inf), T(0), T(3)], @[T(1), T(2), T(3), T(4)],
       @[T(NaN), T(0), T(0), T(0)]),
      (@[T(2), T(0), T(NaN), T(0)], @[T(1), T(2), T(3), T(4)],
       @[T(1), T(NaN), T(0), T(0)]),
      (@[T(2), T(0), T(0), T(3)], @[T(1), T(2), T(3), T(Inf)],
       @[T(1), T(4), T(0), T(0)]),
      (@[T(2), T(0), T(Inf)], @[T(1), T(0), T(0)], @[T(1), T(NaN), T(0)])]:
    let context = newContext Tensor[T]
    let x = context.variable(sample[0].toTensor, requires_grad = true)
    let output = x.cumprod()
    output.backprop(sample[1].toTensor)
    for index in 0 ..< x.grad.size:
      if sample[2][index].classify == fcNan:
        check x.grad[index].classify == fcNan
      else:
        check x.grad[index] == sample[2][index]

proc checkIdentityShapes[T: SomeFloat]() =
  block:
    let context = newContext Tensor[T]
    let x = context.variable(newTensorWith[T]([], T(NaN)), requires_grad = true)
    let output = x.cumprod(-1)
    output.backprop(newTensorWith[T]([], T(3)))
    check x.grad.rank == 0
    check x.grad.atContiguousIndex(0) == T(3)
    expect ValueError:
      discard x.cumprod(1)
  block:
    let context = newContext Tensor[T]
    let x = context.variable([[T(NaN)], [T(Inf)], [T(NegInf)]].toTensor,
                             requires_grad = true)
    let output = x.cumprod(1)
    let seed = [[T(2)], [T(-3)], [T(4)]].toTensor
    output.backprop(seed)
    check x.grad == seed
  block:
    let context = newContext Tensor[T]
    let x = context.variable(newTensor[T](2, 0, 3), requires_grad = true)
    let output = x.cumprod(1)
    output.backprop(zeros[T](2, 0, 3))
    check output.value.shape == [2, 0, 3]
    check x.grad.shape == [2, 0, 3]
    check x.grad.size == 0
    expect ValueError:
      discard x.cumprod(-4)

suite "Autograd cumulative product":
  test "Seeded finite VJPs preserve every packed, signed and broadcast axis":
    checkFiniteViews[float32]()
    checkFiniteViews[float64]()

  test "Nonfinite first-zero masks retain pinned derivative distinctions":
    checkNonfiniteMasks[float32]()
    checkNonfiniteMasks[float64]()

  test "Float32 backward uses rounded prefixes, not unrounded forward carries":
    block:
      let context = newContext Tensor[float32]
      let x = context.variable([1e-30'f32, 1e-30'f32, 1e30'f32].toTensor,
                               requires_grad = true)
      let output = x.cumprod()
      output.backprop([1'f32, 2'f32, 3'f32].toTensor)
      check x.grad == [4'f32, 3'f32, 0'f32].toTensor
    block:
      let context = newContext Tensor[float32]
      let x = context.variable([1e20'f32, 1e20'f32, 1e-20'f32].toTensor,
                               requires_grad = true)
      let output = x.cumprod()
      output.backprop([1'f32, 2'f32, 3'f32].toTensor)
      for value in x.grad: check value.classify == fcInf

  test "Packed column packets keep rounded weights, wide carries and independent tails":
    let values = [[1e-30'f32, 2, 1, 1e20, 0.5],
                  [1e-30'f32, -3, 2, 1e20, 2],
                  [1e30'f32, 4, 3, 1e-20, -1],
                  [1'f32, 0.5, 4, 1, -1]].toTensor
    let seeds = [[1'f32, 1, Inf.float32, 1, -2],
                 [2'f32, 2, 1, 2, 0],
                 [3'f32, 3, 0, 3, 3],
                 [4'f32, 4, -1, 4, 1]].toTensor
    let expected = [[8'f32, -65, Inf.float32, Inf.float32, -6],
                    [7'f32, 44, -11, Inf.float32, -1],
                    [0'f32, -30, -8, Inf.float32, 2],
                    [4e-30'f32, -96, -6, 4e20, -1]].toTensor
    var owner = newTensor[float32](4, 4, 5)
    var seedOwner = newTensor[float32](4, 4, 5)
    for batch in 0 ..< 2:
      for row in 0 ..< 4:
        for column in 0 ..< 5:
          owner[batch + 1, row, column] = values[row, column]
          seedOwner[batch + 1, row, column] = seeds[row, column]
    seedOwner[2, 2, 2] = NaN.float32
    let context = newContext Tensor[float32]
    let x = context.variable(owner[1..2, _, _], requires_grad = true)
    let output = x.cumprod(1)
    output.backprop(seedOwner[1..2, _, _])
    check output.value.shape == [2, 4, 5]
    check x.grad.shape == [2, 4, 5]
    check output.value[0, 1, 0] == 0'f32
    check output.value[0, 3, 0] > 0'f32
    for batch in 0 ..< 2:
      for row in 0 ..< 4:
        for column in 0 ..< 5:
          let actual = x.grad[batch, row, column]
          let target = expected[row, column]
          if batch == 1 and column == 2 and row < 3:
            check actual.classify == fcNan
          elif target.classify == fcInf:
            check actual.classify == fcInf
          else:
            check abs(actual - target) <= 2e-6'f32 * max(1e-35'f32, abs(target))

  test "Scalar, singleton-axis and empty scans retain identity VJPs":
    checkIdentityShapes[float32]()
    checkIdentityShapes[float64]()

  test "Retained backward accumulates and no-grad avoids tracking":
    let context = newContext Tensor[float64]
    let x = context.variable([2.0, -3, 4].toTensor, requires_grad = true)
    let output = x.cumprod()
    output.backprop([1.0, 2, 3].toTensor, retain_graph = true)
    output.backprop([1.0, 1, 1].toTensor, retain_graph = true)
    check x.grad == [-55.0, 38, -24].toTensor
    var untracked: Variable[Tensor[float64]]
    context.no_grad_mode:
      untracked = x.cumprod()
    check not untracked.requires_grad
