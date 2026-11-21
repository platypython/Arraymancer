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

## Exercises ordered cumulative-extrema VJPs, source independence and tape edges.
import ../../src/arraymancer
import std/[unittest, math, complex]

proc checkLatestScatter[T: SomeFloat]() =
  ## Prefix ties route to the latest selected index, never an equality mask.
  let source = [[T(2), T(1), T(3)], [T(1), T(3), T(3)]].toTensor
  for maximum in [false, true]:
    let context = newContext Tensor[T]
    let input = context.variable(source, requires_grad = true)
    let output = if maximum: input.cummax(1) else: input.cummin(1)
    output.values.backprop(newTensorWith[T]([2, 3], T(2)))
    let expected = if maximum: [[T(4), T(0), T(2)], [T(2), T(2), T(2)]].toTensor
      else: [[T(2), T(4), T(0)], [T(6), T(0), T(0)]].toTensor
    check input.grad == expected
    check input.grad.is_C_contiguous

proc checkZeroAndNonfiniteScatter[T: SomeFloat]() =
  ## Positive-zero initialization differs from ordinary paired scalar copying.
  let context = newContext Tensor[T]
  let scalar = context.variable(newTensorWith[T]([], T(NaN)), requires_grad = true)
  let scalarOutput = scalar.cummin(-1)
  scalarOutput.values.backprop(newTensorWith[T]([], T(-0.0)))
  check scalar.grad.rank == 0
  check scalar.grad.atContiguousIndex(0).classify == fcZero
  var source = newTensorWith[T]([2, 3], T(NaN))
  source[0, 0] = T(2)
  source[0, 2] = T(Inf)
  source[1, 1] = T(NegInf)
  let input = context.variable(source, requires_grad = true)
  let output = input.cummax(1)
  output.values.backprop(newTensorWith[T]([2, 3], T(NaN)))
  for index in [0, 1, 3, 5]:
    check input.grad.atContiguousIndex(index).classify == fcNan
  for index in [2, 4]:
    check input.grad.atContiguousIndex(index).classify == fcZero

proc checkScatterRounding[T: SomeFloat]() =
  ## Sequential storage-width additions give one; widened/reassociated sums give two.
  let boundary = when T is float32: T(16777216) else: T(9007199254740992.0)
  let seed = [boundary, T(1), -boundary, T(1)].toTensor
  for maximum in [false, true]:
    let context = newContext Tensor[T]
    let source = if maximum: [T(3), T(2), T(1), T(0)].toTensor
      else: [T(0), T(1), T(2), T(3)].toTensor
    let input = context.variable(source, requires_grad = true)
    let output = if maximum: input.cummax() else: input.cummin()
    output.values.backprop(seed)
    check input.grad == [T(1), T(0), T(0), T(0)].toTensor

proc checkComplexTracking[F: SomeFloat]() =
  ## Tracking, not merely complex dtype, disables scalar/empty differentiation.
  type C = Complex[F]
  let context = newContext Tensor[C]
  let scalar = context.variable(newTensorWith[C]([], C(re: F(2), im: F(7))),
    requires_grad = true)
  expect IndexDefect: discard scalar.cummin(1)
  expect ValueError: discard scalar.cummax(-1)
  let empty = context.variable(newTensor[C](2, 0), requires_grad = true)
  expect IndexDefect: discard empty.cummax(2)
  expect ValueError: discard empty.cummin(1)
  var scalarOutput, emptyOutput: tuple[values: Variable[Tensor[C]], indices: Tensor[int64]]
  context.no_grad_mode:
    scalarOutput = scalar.cummax(-1)
    emptyOutput = empty.cummin(1)
  check scalar.is_grad_needed
  check not scalarOutput.values.requires_grad
  check scalarOutput.values.value.atContiguousIndex(0) == C(re: F(2), im: F(7))
  check scalarOutput.indices.atContiguousIndex(0) == 0'i64
  check not emptyOutput.values.requires_grad
  check emptyOutput.values.value.shape == [2, 0]
  let inactive = context.variable(scalar.value, requires_grad = false).cummin(0)
  check not inactive.values.requires_grad
  check inactive.values.value.atContiguousIndex(0) == C(re: F(2), im: F(7))

suite "CPU cumulative extrema seeded gradients":
  test "Latest selected indices receive every prefix contribution":
    checkLatestScatter[float32]()
    checkLatestScatter[float64]()

  test "Scalar negative-zero seeds add to positive zero and inactive NaN positions stay zero":
    checkZeroAndNonfiniteScatter[float32]()
    checkZeroAndNonfiniteScatter[float64]()

  test "Negative-stride source and seed views use logical coordinates without mutation":
    let context = newContext Tensor[float64]
    let owner = [[9.0, 1, 9], [0.0, 8, 8]].toTensor
    let seedOwner = [[0.0, 1, 2], [3.0, 4, 5]].toTensor
    let source = owner[1..0|-1, 2..0|-1]
    let seed = seedOwner[1..0|-1, 2..0|-1]
    let input = context.variable(source, requires_grad = true)
    input.cummax(-1).values.backprop(seed)
    check input.grad == [[5.0, 7, 0], [3.0, 0, 0]].toTensor
    check input.grad.is_C_contiguous
    check owner == [[9.0, 1, 9], [0.0, 8, 8]].toTensor
    check seedOwner == [[0.0, 1, 2], [3.0, 4, 5]].toTensor

  test "Zero-stride broadcast seeds remain logical input-shaped contributions":
    let context = newContext Tensor[float32]
    let source = [[1.0'f32, 3, 2]].toTensor.broadcast(4, 3)
    let seed = [[-2.0'f32, 0, 2]].toTensor.broadcast(4, 3)
    let input = context.variable(source, requires_grad = true)
    input.cummin(0).values.backprop(seed)
    check input.grad == [[-2.0'f32, 0, 2], [-2.0'f32, 0, 2],
      [-2.0'f32, 0, 2], [-2.0'f32, 0, 2]].toTensor
    check source == [[1.0'f32, 3, 2], [1.0'f32, 3, 2],
      [1.0'f32, 3, 2], [1.0'f32, 3, 2]].toTensor

  test "Thirty-third packed chain accumulates into the latest NaN destination":
    let context = newContext Tensor[float32]
    var source = newTensor[float32](4, 33)
    source[1, 32] = NaN.float32
    source[3, 32] = NaN.float32
    let input = context.variable(source, requires_grad = true)
    input.cummax(0).values.backprop(newTensorWith[float32]([4, 33], 1.0'f32))
    var expected = newTensorWith[float32]([4, 33], 1.0'f32)
    expected[1, 32] = 2
    expected[2, 32] = 0
    check input.grad == expected

  test "Float32 repeated destinations round every addition before the next seed":
    checkScatterRounding[float32]()

  test "Float64 repeated destinations round every addition before the next seed":
    checkScatterRounding[float64]()

  test "Retained VJPs accumulate without changing saved selected indices":
    let context = newContext Tensor[float64]
    let input = context.variable([[2.0, 1, 3], [1.0, 3, 3]].toTensor, requires_grad = true)
    let output = input.cummin(1)
    output.values.backprop(newTensorWith[float64]([2, 3], 2.0), retain_graph = true)
    check input.grad == [[2.0, 4, 0], [6.0, 0, 0]].toTensor
    output.values.backprop([[-2.0, -1, 0], [1.0, 2, -2]].toTensor)
    check input.grad == [[0.0, 3, 0], [7.0, 0, 0]].toTensor
    check output.indices == [[0'i64, 1, 1], [0'i64, 0, 0]].toTensor

  test "Wrong-rank seeds reject before consuming a valid graph":
    let context = newContext Tensor[float64]
    let input = context.variable([[2.0, 1, 3], [1.0, 3, 3]].toTensor, requires_grad = true)
    let output = input.cummin(1)
    expect ValueError: output.values.backprop(newTensorWith[float64]([6], 2.0))
    output.values.backprop(newTensorWith[float64]([2, 3], 2.0))
    check input.grad == [[2.0, 4, 0], [6.0, 0, 0]].toTensor

  test "Gradient storage does not alias inputs, forward values, saved indices or seeds":
    let context = newContext Tensor[float64]
    let source = [3.0, 2, 1].toTensor
    let input = context.variable(source, requires_grad = true)
    let seed = [1.0, 2, 3].toTensor
    let output = input.cummin()
    output.values.backprop(seed)
    check input.grad.get_offset_ptr != source.get_offset_ptr
    check input.grad.get_offset_ptr != seed.get_offset_ptr
    check input.grad.get_offset_ptr != output.values.value.get_offset_ptr
    check cast[pointer](input.grad.get_offset_ptr) != cast[pointer](output.indices.get_offset_ptr)
    input.grad[0] = -77
    check source == [3.0, 2, 1].toTensor
    check output.values.value == source
    check output.indices == [0'i64, 1, 2].toTensor
    check seed == [1.0, 2, 3].toTensor

  test "Complex differentiation rejection preserves no-grad and nontracking forwards":
    checkComplexTracking[float32]()
    checkComplexTracking[float64]()
