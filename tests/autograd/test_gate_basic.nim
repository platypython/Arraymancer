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
import std / [unittest, sequtils]

proc main() =
  suite "Autograd of basic operations":
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
