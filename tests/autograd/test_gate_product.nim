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

import ../../src/arraymancer
import std / [unittest, math]

proc checkFiniteProduct[T: SomeFloat]() =
  let ctx = newContext Tensor[T]
  let x = ctx.variable([2.T, 3, -4].toTensor, requires_grad = true)
  let output = x.product()
  check output.value.rank == 0
  check output.value.atContiguousIndex(0) == T(-24)
  output.backprop(newTensorWith[T]([], T(2)))
  check x.grad == [-24.T, -16, 12].toTensor

proc checkAxisProducts[T: SomeFloat]() =
  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable([[2.T, 3, 4], [5, 6, 7]].toTensor, requires_grad = true)
    let output = x.product(axis = 0, keepdim = false)
    check output.value.shape == [3]
    check output.value == [10.T, 18, 28].toTensor
    output.backprop([2.T, 3, 4].toTensor)
    check x.grad == [[10.T, 18, 28], [4, 9, 16]].toTensor

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable([[2.T, 3, 4], [5, 6, 7]].toTensor, requires_grad = true)
    let kept = x.product(axis = 1)
    check kept.value.shape == [2, 1]
    check kept.value == [[24.T], [210]].toTensor
    let output = x.product(axis = 1, keepdim = false)
    check output.value.shape == [2]
    check output.value == [24.T, 210].toTensor
    output.backprop([2.T, 3].toTensor)
    check x.grad == [[24.T, 16, 12], [126, 105, 90]].toTensor

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable([[2.T, 3, 4], [5, 6, 7]].toTensor, requires_grad = true)
    let output = x.product(axis = -1, keepdim = false)
    output.backprop([5.T, 7].toTensor)
    check x.grad == [[60.T, 40, 30], [294, 245, 210]].toTensor

  block:
    let ctx = newContext Tensor[T]
    let input = [[[2.T, 3, 4, 5, 6], [7.T, 8, 9, 10, 11]],
                 [[12.T, 13, 14, 15, 16], [17.T, 18, 19, 20, 21]]].toTensor
    let x = ctx.variable(input, requires_grad = true)
    let output = x.product(axis = 1)
    output.backprop([[[2.T, 3, 4, 5, 6]], [[-1.T, -2, -3, -4, -5]]].toTensor)
    check x.grad == [[[14.T, 24, 36, 50, 66], [4.T, 9, 16, 25, 36]],
                     [[-17.T, -36, -57, -80, -105], [-12.T, -26, -42, -60, -80]]].toTensor

proc checkNonFiniteProductGradients[T: SomeFloat]() =
  let inf = T(Inf)
  let nan = T(NaN)

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable([inf, T(2)].toTensor, requires_grad = true)
    let output = x.product()
    output.backprop()
    check x.grad[0].classify == fcNan
    check x.grad[1].classify == fcInf

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable([inf].toTensor, requires_grad = true)
    x.product().backprop()
    check x.grad[0].classify == fcNan

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable([T(0), inf, T(3)].toTensor, requires_grad = true)
    x.product().backprop()
    check x.grad[0].classify == fcInf
    check x.grad[1].classify == fcZero
    check x.grad[2].classify == fcNan

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable([T(0), nan, T(3)].toTensor, requires_grad = true)
    x.product().backprop()
    check x.grad[0].classify == fcNan
    check x.grad[1].classify == fcZero
    check x.grad[2].classify == fcNan

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable([T(0), T(0), inf].toTensor, requires_grad = true)
    x.product().backprop()
    for value in x.grad:
      check value.classify == fcZero

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable([T(0), T(0), inf].toTensor, requires_grad = true)
    x.product(axis = 0).backprop()
    check x.grad[0].classify == fcNan
    check x.grad[1].classify == fcNan
    check x.grad[2].classify == fcZero

proc checkProductViews[T: SomeFloat]() =
  block:
    let ctx = newContext Tensor[T]
    let owner = [[T(99), 2, 3, 98], [T(97), 4, 5, 96]].toTensor
    let view = owner[_, 1..2].transpose
    let x = ctx.variable(view, requires_grad = true)
    x.product().backprop()
    check x.grad == [[60.T, 30], [40, 24]].toTensor

  block:
    let ctx = newContext Tensor[T]
    let rows = [T(2), 3].toTensor.reshape(1, 2).broadcast(3, 2)
    let x = ctx.variable(rows, requires_grad = true)
    x.product().backprop()
    check x.grad == [[108.T, 72], [108, 72], [108, 72]].toTensor

proc checkProductScalarsAndEmpty[T: SomeFloat]() =
  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable(newTensorWith[T]([], T(0)), requires_grad = true)
    x.product().backprop(newTensorWith[T]([], T(4)))
    check x.grad.rank == 0
    check x.grad.atContiguousIndex(0) == T(4)

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable(newTensorWith[T]([], T(Inf)), requires_grad = true)
    x.product().backprop(newTensorWith[T]([], T(-2)))
    check x.grad.atContiguousIndex(0) == T(-2)

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable(newTensorWith[T]([], T(NaN)), requires_grad = true)
    x.product(axis = -1).backprop(newTensorWith[T]([], T(0)))
    check x.grad.atContiguousIndex(0) == T(0)

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable(newTensorWith[T]([], T(3)), requires_grad = true)
    let output = x.product(axis = 0)
    check output.value.rank == 0
    check output.value.atContiguousIndex(0) == T(3)
    output.backprop()
    check x.grad.atContiguousIndex(0) == T(1)

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable(newTensor[T](2, 0, 3), requires_grad = true)
    let full = x.product()
    check full.value.atContiguousIndex(0) == T(1)
    full.backprop()
    check x.grad.shape == [2, 0, 3]
    check x.grad.size == 0

  block:
    let ctx = newContext Tensor[T]
    let x = ctx.variable(newTensor[T](2, 0, 3), requires_grad = true)
    let output = x.product(axis = 1, keepdim = false)
    check output.value == [[T(1), 1, 1], [1, 1, 1]].toTensor
    output.backprop()
    check x.grad.shape == [2, 0, 3]
    check x.grad.size == 0

proc checkProductGraphModes[T: SomeFloat]() =
  let ctx = newContext Tensor[T]
  let x = ctx.variable([T(2), 3].toTensor, requires_grad = true)
  let output = x.product()
  output.backprop(newTensorWith[T]([], T(2)), retain_graph = true)
  output.backprop(newTensorWith[T]([], T(3)), retain_graph = true)
  check x.grad == [T(15), 10].toTensor

  ctx.no_grad_mode:
    check not x.product().requires_grad

proc main() =
  suite "Autograd product reductions":
    test "Full product VJP supports float32 and float64":
      checkFiniteProduct[float32]()
      checkFiniteProduct[float64]()

    test "Axis products preserve keepdim and accept positive and negative axes":
      checkAxisProducts[float32]()
      checkAxisProducts[float64]()

    test "Zero and non-finite product derivatives match reduction semantics":
      checkNonFiniteProductGradients[float32]()
      checkNonFiniteProductGradients[float64]()

    test "Product backward reads transpose, offset-slice and broadcast views":
      checkProductViews[float32]()
      checkProductViews[float64]()

    test "Product reductions support scalars and empty groups":
      checkProductScalarsAndEmpty[float32]()
      checkProductScalarsAndEmpty[float64]()

    test "Product backward accumulates retained VJPs and respects no-grad":
      checkProductGraphModes[float32]()
      checkProductGraphModes[float64]()

main()
