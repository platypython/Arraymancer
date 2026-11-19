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

## Exercises first-index scatter, inactive positive zeros and saved-index gates.
import ../../src/arraymancer
import std/[math, unittest]

proc checkFirstScatter[F: SomeFloat]() =
  ## Ties must not split seeds, and inactive nonfinite seeds must not multiply masks.
  let values = [[F(1), F(1), F(2)], [F(3), F(3), F(0)]].toTensor
  for maximum in [false, true]:
    for keepdim in [false, true]:
      let context = newContext Tensor[F]
      let source = context.variable(values, requires_grad = true)
      let pair = if maximum: source.maxWithIndices(1, keepdim)
                 else: source.minWithIndices(-1, keepdim)
      let chosen = if maximum: [2, 0] else: [0, 2]
      let weights = [F(2), F(-3)]
      if keepdim:
        check pair.values.value.shape == [2, 1]
        check pair.indices.shape == [2, 1]
      else:
        check pair.values.value.shape == [2]
        check pair.indices.shape == [2]
      pair.values.backprop(weights.toTensor.reshape(pair.values.value.shape))
      for row in 0 ..< 2:
        check pair.indices.atContiguousIndex(row) == int64(chosen[row])
        for column in 0 ..< 3:
          if column == chosen[row]: check source.grad[row, column] == weights[row]
          else: check source.grad[row, column].classify == fcZero
      check values == [[F(1), F(1), F(2)], [F(3), F(3), F(0)]].toTensor

  for maximum in [false, true]:
    let context = newContext Tensor[F]
    let source = context.variable(
      [[F(0), F(NaN), F(NaN)], [F(NaN), F(-1), F(2)]].toTensor,
      requires_grad = true)
    let pair = if maximum: source.maxWithIndices(1) else: source.minWithIndices(1)
    pair.values.backprop([F(Inf), F(NaN)].toTensor)
    check pair.indices == [1'i64, 0'i64].toTensor
    check source.grad[0, 1].classify == fcInf
    check source.grad[1, 0].classify == fcNan
    for flat in [0, 2, 4, 5]:
      check source.grad.atContiguousIndex(flat).classify == fcZero

  for maximum in [false, true]:
    let context = newContext Tensor[F]
    let source = context.variable(
      [[F(-0.0), F(0.0), F(-0.0)], [F(0.0), F(-0.0), F(0.0)]].toTensor,
      requires_grad = true)
    let pair = if maximum: source.maxWithIndices(1) else: source.minWithIndices(1)
    pair.values.backprop(newTensorWith[F]([2], F(-0.0)))
    for flat in 0 ..< 6:
      let expected = if flat mod 3 == 0: fcNegZero else: fcZero
      check source.grad.atContiguousIndex(flat).classify == expected

proc checkViewScatter[F: SomeFloat]() =
  ## A retained negative-stride seed and 33-column tile tail stress logical mapping.
  var values = newTensorUninit[F](2, 2, 33)
  for index in 0 ..< values.size:
    values.unsafe_raw_offset[index] = F((index div 33) mod 2)
  let view = values[^1..0|-1, ^1..0|-1, _]
  var seedStorage = newTensorUninit[F](2, 1, 33)
  for index in 0 ..< seedStorage.size:
    seedStorage.unsafe_raw_offset[index] = F(index + 1)
  let seed = seedStorage[^1..0|-1, _, ^1..0|-1]
  for maximum in [false, true]:
    let context = newContext Tensor[F]
    let source = context.variable(view, requires_grad = true)
    let pair = if maximum: source.maxWithIndices(1, keepdim = true)
               else: source.minWithIndices(-2, keepdim = true)
    pair.values.backprop(seed)
    check source.grad.shape == [2, 2, 33]
    check source.grad.is_C_contiguous
    let chosen = if maximum: 0 else: 1
    for outer in 0 ..< 2:
      for row in 0 ..< 2:
        for column in 0 ..< 33:
          if row == chosen:
            check source.grad[outer, row, column] == seed[outer, 0, column]
          else:
            check source.grad[outer, row, column].classify == fcZero
    source.grad[0, 0, 0] = F(123)
    check source.value[0, 0, 0] == F(1)
    check seed[0, 0, 0] == F(66)

  block:
    let context = newContext Tensor[F]
    let source = context.variable([[F(1), F(3), F(2)]].toTensor.broadcast(4, 3),
      requires_grad = true)
    source.maxWithIndices(0).values.backprop([F(2), F(-3), F(4)].toTensor)
    check source.grad[0, _] == [[F(2), F(-3), F(4)]].toTensor
    for row in 1 ..< 4:
      for column in 0 ..< 3:
        check source.grad[row, column].classify == fcZero

proc checkGraphContracts[F: SomeFloat]() =
  ## Saved indices remain usable across retained traversals and unrelated graphs.
  let context = newContext Tensor[F]
  let source = context.variable([F(1), F(1), F(2)].toTensor, requires_grad = true)
  let unrelated = context.variable([F(4), F(3)].toTensor, requires_grad = true)
  let other = unrelated.minWithIndices(0)
  let pair = source.minWithIndices(0)
  let negativeZero = newTensorWith[F]([], F(-0.0))
  pair.values.backprop(negativeZero, retain_graph = true)
  check source.grad[0].classify == fcNegZero
  check source.grad[1].classify == fcZero
  pair.values.backprop(newTensorWith[F]([], F(2)), retain_graph = true)
  pair.values.backprop(newTensorWith[F]([], F(3)))
  check source.grad == [F(5), F(0), F(0)].toTensor
  other.values.backprop(newTensorWith[F]([], F(-2)))
  check unrelated.grad == [F(0), F(-2)].toTensor
  expect ValueError: pair.values.backprop(negativeZero)

  let scalar = context.variable(newTensorWith[F]([], F(NaN)), requires_grad = true)
  let scalarPair = scalar.maxWithIndices(-1, keepdim = true)
  check scalarPair.indices.shape.len == 0
  scalarPair.values.backprop(negativeZero)
  check scalar.grad.shape.len == 0
  check scalar.grad.atContiguousIndex(0).classify == fcNegZero
  let empty = context.variable(newTensor[F](0, 3), requires_grad = true)
  empty.minWithIndices(1).values.backprop(newTensor[F](0))
  check empty.grad.shape == [0, 3]
  check empty.grad.size == 0

  var untracked: Variable[Tensor[F]]
  context.no_grad_mode:
    untracked = source.maxWithIndices(0).values
  check not untracked.requires_grad
  expect ValueError: untracked.backprop(negativeZero)
  let detached = context.variable([F(1), F(2)].toTensor)
  check not detached.minWithIndices(0).values.requires_grad
  let fresh = context.variable([F(1), F(2)].toTensor, requires_grad = true)
  let freshPair = fresh.maxWithIndices(0)
  expect ValueError: freshPair.values.backprop([F(1)].toTensor)
  freshPair.values.backprop(newTensorWith[F]([], F(3)))
  check fresh.grad == [F(0), F(3)].toTensor
  expect IndexDefect: discard fresh.minWithIndices(1)

suite "CPU paired extrema seeded gradients":
  test "Float32 ties, first NaNs and inactive seed signs":
    checkFirstScatter[float32]()
  test "Float64 ties, first NaNs and inactive seed signs":
    checkFirstScatter[float64]()
  test "Float32 signed/broadcast views retain dense gradient coordinates":
    checkViewScatter[float32]()
  test "Float64 signed/broadcast views retain dense gradient coordinates":
    checkViewScatter[float64]()
  test "Float32 scalar, empty, no-grad and retained graph contracts":
    checkGraphContracts[float32]()
  test "Float64 scalar, empty, no-grad and retained graph contracts":
    checkGraphContracts[float64]()
