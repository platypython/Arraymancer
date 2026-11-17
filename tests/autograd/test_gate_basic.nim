# Copyright 2017-2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
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
import std / [unittest, sequtils, math]

proc main() =
  suite "Autograd of basic operations":
    test "Broadcast multiplication and division reduce both seeded gradients":
      let seed = [[1.0, 2, 3], [4.0, 5, 6]].toTensor
      for divide in [false, true]:
        let ctx = newContext Tensor[float64]
        let a = ctx.variable([[2.0], [4.0]].toTensor, requires_grad = true)
        let b = ctx.variable([1.0, 2, 4].toTensor, requires_grad = true)
        let output = if divide: a /. b else: a *. b
        output.backprop(seed)
        if divide:
          check a.grad == [[2.75], [8.0]].toTensor
          check b.grad == [-18.0, -6, -1.875].toTensor
        else:
          check a.grad == [[17.0], [38.0]].toTensor
          check b.grad == [18.0, 24, 30].toTensor

    test "Scalar broadcast gradients terminate and preserve rank zero":
      let ctx = newContext Tensor[float64]
      let scalar = ctx.variable(newTensorWith[float64]([], 2.0), requires_grad = true)
      let x = ctx.variable([[1.0, 2, 4], [1.0, 2, 4]].toTensor)
      let product = scalar *. x
      product.sum.backprop()
      check scalar.grad.rank == 0
      check scalar.grad.atContiguousIndex(0) == 14.0
      scalar.grad.setZero()
      let quotient = scalar /. x
      quotient.sum.backprop()
      check scalar.grad.rank == 0
      check scalar.grad.atContiguousIndex(0) == 3.5

    test "Empty broadcast products and quotients accumulate zero gradients":
      for divide in [false, true]:
        let ctx = newContext Tensor[float64]
        let a = ctx.variable(ones[float64](1, 3), requires_grad = true)
        let b = ctx.variable(zeros[float64](0, 3), requires_grad = true)
        let output = if divide: a /. b else: a *. b
        output.backprop(zeros[float64](0, 3))
        check a.grad == zeros[float64](1, 3)
        check b.grad.shape == [0, 3]
        check b.grad.size == 0

    test "Negative-axis reductions squeeze outputs and restore VJP dimensions":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([[1.0, 2, 3], [4.0, 5, 6]].toTensor.transpose,
                           requires_grad = true)
      let total = x.sum(axis = -1, keepdim = false)
      let average = x.mean(axis = -1, keepdim = false)
      check total.value == [5.0, 7, 9].toTensor
      check average.value == [2.5, 3.5, 4.5].toTensor
      total.backprop([2.0, 4, 6].toTensor)
      average.backprop([4.0, 8, 12].toTensor)
      check x.grad == [[4.0, 4], [8.0, 8], [12.0, 12]].toTensor

    test "Rank-one and scalar axis reductions preserve scalar semantics":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([2.0, 4].toTensor, requires_grad = true)
      let average = x.mean(axis = 0, keepdim = false)
      check average.value.rank == 0
      check average.value.atContiguousIndex(0) == 3.0
      let scalar = average.sum(axis = -1)
      check scalar.value.rank == 0
      scalar.backprop()
      check x.grad == [0.5, 0.5].toTensor
      expect ValueError:
        discard x.mean(axis = -2)
      expect ValueError:
        discard x.sum(axis = 1)

    test "Empty reduced axes produce zero sums and NaN means with correct shapes":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable(zeros[float64](0, 3), requires_grad = true)
      let total = x.sum(axis = 0, keepdim = false)
      let average = x.mean(axis = 0, keepdim = false)
      check total.value == zeros[float64](3)
      check average.value.shape == [3]
      for value in average.value: check value.classify == fcNan
      total.backprop()
      average.backprop()
      check x.grad.shape == [0, 3]
      check x.grad.size == 0
      let unreduced = x.sum(axis = 1)
      check unreduced.value.shape == [0, 1]
      unreduced.backprop()
      check x.grad.size == 0

    test "No-grad mode disables chunk, squeeze and unsqueeze backward edges":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([[[2.0], [3.0]]].toTensor, requires_grad = true)
      ctx.no_grad_mode:
        let squeezed = x.squeeze(0)
        let expanded = x.unsqueeze(1)
        let pieces = x.chunk(2, axis = 1)
        check squeezed.value == [[2.0], [3.0]].toTensor
        check expanded.value.shape == [1, 1, 2, 1]
        check not squeezed.requires_grad
        check not expanded.requires_grad
        for piece in pieces:
          check not piece.requires_grad
        expect ValueError:
          pieces[0].backprop()
        ctx.no_grad_mode:
          check not x.squeeze(0).requires_grad
        check not x.is_grad_needed
      check x.is_grad_needed
      x.sum.backprop()
      check x.grad == ones[float64](1, 2, 1)

    test "Ellipsis resolves trailing selectors and negative indices once":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable(toSeq(0..11).toTensor.asType(float64).reshape(2, 2, 3),
                           requires_grad = true)
      let selected = x[`...`, [-1, 0]]
      check selected.value == [[[2.0, 0], [5.0, 3]], [[8.0, 6], [11.0, 9]]].toTensor
      selected.sum.backprop()
      check x.grad == [[[1.0, 0, 1], [1.0, 0, 1]], [[1.0, 0, 1], [1.0, 0, 1]]].toTensor

    test "Slice bounds evaluate once and no-grad disables slice recording":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([1.0, 2, 3, 4].toTensor, requires_grad = true)
      var calls = 0
      proc start(): int =
        inc calls
        1
      let selected = x[start()..3]
      check calls == 1
      ctx.no_grad_mode:
        check not x[1..2].requires_grad
        check not x[1].requires_grad
        check not x[[0, 2]].requires_grad
      selected.sum.backprop()
      check x.grad == [0.0, 1, 1, 1].toTensor

    test "Overlapping signed-stride slices scatter seeded gradients":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([1.0, 2, 3, 4, 5].toTensor, requires_grad = true)
      let a = x[1..3]
      let b = x[4..0|-2]
      let output = a.sum + b.sum
      output.backprop(retain_graph = true)
      check x.grad == [1.0, 1, 2, 1, 1].toTensor
      output.backprop()
      check x.grad == [2.0, 2, 4, 2, 2].toTensor

    test "Point indexing returns a scalar and records a backward edge":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([[1.0, 2], [3.0, 4]].toTensor.transpose, requires_grad = true)
      let output = x[1, 0]
      check output.value.rank == 0
      check output.value.atContiguousIndex(0) == 2.0
      output.backprop([3.0].toTensor.reshape(newSeq[int]()))
      check x.grad == [[0.0, 0], [3.0, 0]].toTensor

    test "Repeated index selections accumulate and snapshot selectors":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([[1.0, 2, 3], [4.0, 5, 6]].toTensor, requires_grad = true)
      var indices = [2, 0, 2].toTensor
      let output = x[_, indices]
      indices[0] = 1
      check output.value == [[3.0, 1, 3], [6.0, 4, 6]].toTensor
      output.backprop([[1.0, 2, 3], [4.0, 5, 6]].toTensor)
      check x.grad == [[2.0, 0, 4], [5.0, 0, 10]].toTensor

    test "Full and axis masks scatter without aliasing gradients":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([[1.0, 2, 3], [4.0, 5, 6]].toTensor.transpose, requires_grad = true)
      let full = x[[[true, false], [false, true], [true, false]]]
      check full.value == [1.0, 5, 3].toTensor
      full.backprop([2.0, 3, 4].toTensor)
      check x.grad == [[2.0, 0], [0.0, 3], [4.0, 0]].toTensor
      let axis = x[[true, false, true], _]
      axis.sum.backprop()
      check x.grad == [[3.0, 1], [0.0, 3], [5.0, 1]].toTensor
      let empty = x[[false, false, false], _]
      check empty.value.shape == [0, 2]
      empty.backprop(zeros[float64](0, 2))
      check x.grad == [[3.0, 1], [0.0, 3], [5.0, 1]].toTensor

    test "Retained backward recomputes intermediates and accumulates only leaves":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([2.0, 3].toTensor, requires_grad = true)
      let squared = x *. x
      let output = squared *. squared
      output.backprop([1.0, 2].toTensor, retain_graph = true)
      check x.grad == [32.0, 216].toTensor
      output.backprop([3.0, 4].toTensor, retain_graph = true)
      check x.grad == [128.0, 648].toTensor
      check squared.grad == [24.0, 72].toTensor
      output.backprop()
      check x.grad == [160.0, 756].toTensor
      expect ValueError:
        output.backprop()
      check x.grad == [160.0, 756].toTensor

    test "Independent graphs survive out-of-order backward and leaf backward":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([2.0, 3].toTensor, requires_grad = true)
      let y = ctx.variable([4.0, 5].toTensor, requires_grad = true)
      let earlier = x *. x
      let later = y *. y
      x.backprop([1.0, 2].toTensor)
      earlier.backprop()
      check x.grad == [5.0, 8].toTensor
      check y.grad == [0.0, 0].toTensor
      later.backprop()
      check y.grad == [8.0, 10].toTensor

    test "Unrelated singular gates cannot contaminate gradients":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([2.0, 3].toTensor, requires_grad = true)
      let unused = x /. ctx.variable(zeros[float64](2))
      let output = x + x
      output.backprop()
      check x.grad == [2.0, 2].toTensor
      check unused.grad == zeros[float64](2)

    test "Shared freed ancestors fail before modifying gradients":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([2.0, 3].toTensor, requires_grad = true)
      let squared = x *. x
      let a = squared + squared
      let b = squared *. squared
      a.backprop(retain_graph = true)
      b.backprop()
      check x.grad == [40.0, 120].toTensor
      expect ValueError:
        a.backprop()
      check x.grad == [40.0, 120].toTensor
      check a.grad == ones[float64](2)

    test "Direct chunk backward zeros unused outputs on each retained traversal":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([1.0, 2, 3, 4].toTensor, requires_grad = true)
      let parts = x.chunk(2, axis = 0)
      parts[1].backprop([2.0, 3].toTensor, retain_graph = true)
      check x.grad == [0.0, 0, 2, 3].toTensor
      parts[0].backprop([5.0, 7].toTensor)
      check x.grad == [5.0, 7, 2, 3].toTensor
      expect ValueError:
        parts[1].backprop()

    test "Sum returns a scalar and propagates a scalar seed through a view":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([[1.0, 2, 3], [4.0, 5, 6]].toTensor.transpose,
                           requires_grad = true)
      let output = x.sum
      check output.value.rank == 0
      check output.value.size == 1
      check output.value.atContiguousIndex(0) == 21.0
      output.backprop([2.0].toTensor.reshape(newSeq[int]()))
      check x.grad == ones[float64](3, 2) * 2.0

    test "Axis mean VJP divides by the reduced axis on a three-dimensional view":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable(toSeq(1..24).toTensor.asType(float64).reshape(2, 3, 4),
                           requires_grad = true)
      let output = x.mean(axis = 1)
      let seed = ones[float64](2, 1, 4) * 6.0
      output.backprop(seed)
      check output.value.shape == [2, 1, 4]
      check output.value == [[[5.0, 6, 7, 8]], [[17.0, 18, 19, 20]]].toTensor
      check x.grad == ones[float64](2, 3, 4) * 2.0

    test "Explicit strided gradients weight a branched vector-Jacobian product":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([[1.0, 2, 3], [4.0, 5, 6]].toTensor,
                           requires_grad = true)
      let y = x *. x + x
      var seed = [[1.0, 4], [2.0, 5], [3.0, 6]].toTensor.transpose
      let original = seed.clone
      y.backprop(seed)
      check x.grad == [[3.0, 10, 21], [36.0, 55, 78]].toTensor
      check y.grad == original
      seed[0, 0] = 100
      check y.grad == original

    test "Seeded broadcast gradients reduce leading and singleton dimensions":
      let ctx = newContext Tensor[float64]
      let a = ctx.variable([[10.0], [20.0]].toTensor, requires_grad = true)
      let b = ctx.variable([1.0, 2, 3].toTensor, requires_grad = true)
      let output = a +. b
      output.backprop([[1.0, 2, 3], [4.0, 5, 6]].toTensor)
      check a.grad == [[6.0], [15.0]].toTensor
      check b.grad == [5.0, 7, 9].toTensor

    test "Invalid seed shape preserves the tape and existing gradients":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([[1.0, 2], [3.0, 4]].toTensor, requires_grad = true)
      let output = x + x
      expect ValueError:
        output.backprop([1.0, 2, 3, 4].toTensor)
      check x.grad == zeros[float64](2, 2)
      check output.grad == zeros[float64](2, 2)
      output.backprop([[1.0, 2], [3.0, 4]].toTensor)
      check x.grad == [[2.0, 4], [6.0, 8]].toTensor

    test "Explicit scalar seeds scale reductions and leaf seeds accumulate":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable([2.0, 4].toTensor, requires_grad = true)
      let output = x.mse_loss(zeros[float64](2))
      output.backprop([3.0].toTensor.reshape(newSeq[int]()))
      check x.grad == [6.0, 12].toTensor
      let seed = [2.0, -1].toTensor
      x.backprop(seed)
      x.backprop(seed)
      check x.grad == [10.0, 10].toTensor
      check seed == [2.0, -1].toTensor

    test "Empty output seeds and untraced outputs":
      let ctx = newContext Tensor[float64]
      let x = ctx.variable(zeros[float64](0, 3), requires_grad = true)
      let output = x + x
      output.backprop(zeros[float64](0, 3))
      check x.grad.shape == [0, 3]
      check x.grad.size == 0
      let constant = ctx.variable([1.0].toTensor)
      expect ValueError:
        constant.backprop([2.0].toTensor)

    test "Gradient of tensor addition":

      let a = toSeq(1..8).toTensor.reshape(2,4).asType(float32)
      let b = toSeq(11..18).toTensor.reshape(2,4).asType(float32)

      let ctx = newContext Tensor[float32]

      let va = ctx.variable(a, requires_grad = true)
      let vb = ctx.variable(b, requires_grad = true)

      let vc = va + vb

      vc.backprop()

      let onesTensor = ones[float32](2, 4)

      check: va.grad == onesTensor
      check: vb.grad == onesTensor

    test "Gradient of mean":

      let a = toSeq(1..8).toTensor.reshape(2,4).asType(float32)

      let ctx = newContext Tensor[float32]

      let va = ctx.variable(a, requires_grad = true)
      let m = va.mean()
      check m.value.rank == 0
      check m.value.size == 1
      check m.value.atContiguousIndex(0) == 4.5'f32

      m.backprop()

      let constantTensor = ones[float32](2, 4) / 8.0

      check: va.grad == constantTensor

    test "MSE returns a scalar and backpropagates the mean gradient":
      let
        prediction = [1'f32, 2, 3, 4].toTensor.reshape(2, 2)
        target = zeros[float32](2, 2)
        ctx = newContext Tensor[float32]
      let vprediction = ctx.variable(prediction, requires_grad = true)
      let loss = vprediction.mse_loss(target)

      check: loss.value.rank == 0
      check: loss.value.size == 1
      check: loss.value.atContiguousIndex(0) == 7.5'f32

      loss.backprop()

      check: loss.grad.rank == 0
      check: vprediction.grad == [0.5'f32, 1, 1.5, 2].toTensor.reshape(2, 2)


    test "Gradient of mean along one axis":

      let a = toSeq(1..8).toTensor.reshape(2,4).asType(float32)

      let ctx = newContext Tensor[float32]

      let va = ctx.variable(a, requires_grad = true)

      let m0 = va.mean(axis=0)
      m0.backprop()
      let constantTensor0 = ones[float32](2, 4) / 2.0
      check: va.grad == constantTensor0

      va.grad = zeros_like(va.grad)

      let m = va.mean(axis=1)
      m.backprop()
      let constantTensor1 = ones[float32](2, 4) / 4.0
      check: va.grad == constantTensor1

    test "Gradient of division":
      let a = toSeq(1..8).toTensor.reshape(2,4).asType(float32)
      let b = toSeq(11..18).toTensor.reshape(2,4).asType(float32)

      let ctx = newContext Tensor[float32]

      let va = ctx.variable(a, requires_grad = true)
      let vb = ctx.variable(b, requires_grad = true)

      let vc = va /. vb

      vc.backprop()

      let gradC = ones[float32](2, 4)

      check: va.grad == gradC /. b
      check: vb.grad == - gradC *. a /. b ^. 2

main()
GC_fullCollect()
