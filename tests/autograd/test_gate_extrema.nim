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

## Exercises tie counts, masked arithmetic, signed views and gradient identities.

import ../../src/arraymancer
import std/[unittest, math]

proc checkTiesAndMasks[F: SomeFloat]() =
  ## Negative and infinite seeds distinguish numeric masks from zero assignment.
  let values = [[F(1), F(1), F(3)], [F(3), F(3), F(2)]].toTensor
  for maximum in [false, true]:
    for keepdim in [false, true]:
      let context = newContext Tensor[F]
      let source = context.variable(values, requires_grad = true)
      let output = if maximum: source.amax(1, keepdim) else: source.amin(-1, keepdim)
      if keepdim: check output.value.shape == [2, 1]
      else: check output.value.shape == [2]
      output.backprop([F(2), F(-3)].toTensor.reshape(output.value.shape))
      if maximum:
        check source.grad == [[F(0), F(0), F(2)], [F(-1.5), F(-1.5), F(0)]].toTensor
        check source.grad[1, 2].classify == fcNegZero
      else:
        check source.grad == [[F(1), F(1), F(0)], [F(0), F(0), F(-3)]].toTensor
        check source.grad[1, 0].classify == fcNegZero
        check source.grad[1, 1].classify == fcNegZero
      check source.grad[0, if maximum: 0 else: 2].classify == fcZero

  block:
    let context = newContext Tensor[F]
    let source = context.variable(values, requires_grad = true)
    source.amax(1).backprop([F(Inf), F(-Inf)].toTensor)
    check source.grad[0, 0].classify == fcNan
    check source.grad[0, 1].classify == fcNan
    check source.grad[0, 2].classify == fcInf
    check source.grad[1, 0].classify == fcNegInf
    check source.grad[1, 1].classify == fcNegInf
    check source.grad[1, 2].classify == fcNan
  block:
    let context = newContext Tensor[F]
    let source = context.variable(values, requires_grad = true)
    source.amin(1).backprop(newTensorWith[F]([2], F(-0.0)))
    for index in 0 ..< source.grad.size:
      check source.grad.atContiguousIndex(index).classify == fcNegZero
  block:
    let context = newContext Tensor[F]
    let source = context.variable(
      [[F(1), F(NaN), F(-2)], [F(3), F(3), F(1)]].toTensor, requires_grad = true)
    source.amin(1).backprop([F(2), F(-3)].toTensor)
    for column in 0 ..< 3: check source.grad[0, column].classify == fcNan
    check source.grad[1, 0].classify == fcNegZero
    check source.grad[1, 1].classify == fcNegZero
    check source.grad[1, 2] == F(-3)
  block:
    let context = newContext Tensor[F]
    let source = context.variable(newTensorWith[F]([], F(NaN)), requires_grad = true)
    source.amax(-1, keepdim = true).backprop(newTensorWith[F]([], F(-3)))
    check source.grad.shape == []
    check source.grad.atContiguousIndex(0).classify == fcNan

proc checkViewsAndTiles[F: SomeFloat]() =
  ## Three separated axes exercise both carry levels; 33 columns cross a tile tail.
  for reversed in [false, true]:
    var values = newTensorUninit[F](2, 2, 2, 2, 2)
    for index in 0 ..< values.size: values.unsafe_raw_offset[index] = F(index)
    let view = if reversed:
      values[^1..0|-1, ^1..0|-1, ^1..0|-1, ^1..0|-1, ^1..0|-1]
    else: values
    let context = newContext Tensor[F]
    let source = context.variable(view, requires_grad = true)
    let output = source.amin([0, 2, 4])
    output.backprop([[F(1), F(-2)], [F(3), F(-4)]].toTensor)
    check source.grad.shape == [2, 2, 2, 2, 2]
    check source.grad.is_C_contiguous
    let chosen = if reversed: [21, 23, 29, 31] else: [0, 2, 8, 10]
    let weights = [F(1), F(-2), F(3), F(-4)]
    for index in 0 ..< values.size:
      var expected = F(0)
      for position in 0 ..< 4:
        if index == chosen[position]: expected = weights[position]
      check source.grad.atContiguousIndex(index) == expected
      if expected == F(0):
        let kind = if (index div 2) mod 2 == 1: fcNegZero else: fcZero
        check source.grad.atContiguousIndex(index).classify == kind
    source.grad[0, 0, 0, 0, 0] = F(123)
    check source.value[0, 0, 0, 0, 0] == F(if reversed: 31 else: 0)

  block:
    var values = newTensorWith[F]([2, 33], F(1))
    var seed = newTensorUninit[F](33)
    for column in 0 ..< 33:
      if column mod 2 == 1: values[1, column] = F(2)
      seed[column] = F(column mod 3 - 1)
    let context = newContext Tensor[F]
    let source = context.variable(values, requires_grad = true)
    source.amin(0).backprop(seed)
    for column in 0 ..< 33:
      let weight = seed[column] / F(if column mod 2 == 0: 2 else: 1)
      check source.grad[0, column] == weight
      check source.grad[1, column] == (if column mod 2 == 0: weight else: F(0))
      if column mod 2 == 1 and seed[column] < F(0):
        check source.grad[1, column].classify == fcNegZero

  block:
    let context = newContext Tensor[F]
    let source = context.variable([[F(1), F(3), F(2)]].toTensor.broadcast(4, 3),
      requires_grad = true)
    source.amax(0, keepdim = true).backprop([[F(2), F(-3), F(4)]].toTensor)
    for row in 0 ..< 4:
      check source.grad[row, 0] == F(0.5)
      check source.grad[row, 1] == F(-0.75)
      check source.grad[row, 2] == F(1)

proc checkAccumulation[F: SomeFloat]() =
  ## Fresh and reset intermediate buffers must preserve the first zero's sign.
  let context = newContext Tensor[F]
  let source = context.variable([F(1), F(2), F(3)].toTensor, requires_grad = true)
  let output = (source + source).sum()
  let seed = newTensorWith[F]([], F(-0.0))
  for iteration in 0 ..< 2:
    output.backprop(seed, retain_graph = true)
    for index in 0 ..< source.grad.size:
      check source.grad[index].classify == fcNegZero
  source.grad.setZero()
  output.backprop(seed)
  for index in 0 ..< source.grad.size:
    check source.grad[index].classify == fcZero

suite "CPU amin/amax seeded gradients":
  test "Float32 ties and nonfinite masks retain IEEE zero signs":
    checkTiesAndMasks[float32]()
  test "Float64 ties and nonfinite masks retain IEEE zero signs":
    checkTiesAndMasks[float64]()
  test "Float32 VJPs cover signed carry coordinates and column tile tails":
    checkViewsAndTiles[float32]()
  test "Float64 VJPs cover signed carry coordinates and column tile tails":
    checkViewsAndTiles[float64]()
  test "Float32 first and retained contributions preserve signed zeros":
    checkAccumulation[float32]()
  test "Float64 first and retained contributions preserve signed zeros":
    checkAccumulation[float64]()
  test "Empty gradients, scalar axes and no-grad calls preserve their contracts":
    let context = newContext Tensor[float64]
    let empty = context.variable(newTensor[float64](2, 0, 3), requires_grad = true)
    let reduced = empty.amin([0, 2], keepdim = true)
    check reduced.value.shape == [1, 0, 1]
    reduced.backprop(newTensor[float64](1, 0, 1))
    check empty.grad.shape == [2, 0, 3]
    check empty.grad.size == 0
    let scalar = context.variable(newTensorWith[float64]([], -0.0), requires_grad = true)
    let scalarOutput = scalar.amin([], keepdim = true)
    check scalarOutput.value.shape == []
    scalarOutput.backprop(newTensorWith[float64]([], -3.0))
    check scalar.grad.atContiguousIndex(0) == -3.0
    let tracked = context.variable([[1.0, 2.0], [3.0, 4.0]].toTensor, requires_grad = true)
    var untracked: Variable[Tensor[float64]]
    context.no_grad_mode:
      untracked = tracked.amax(0)
    check not untracked.requires_grad
    expect ValueError: untracked.backprop([1.0, 1.0].toTensor)
    expect ValueError: discard tracked.amin([0, -2])
    expect IndexDefect: discard tracked.amax(2)
