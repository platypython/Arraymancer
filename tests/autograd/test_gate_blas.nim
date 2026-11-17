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

# # Differentiating through matmul:
# # See http://cs231n.stanford.edu/vecDerivs.pdf
# # And: https://danieltakeshi.github.io/2017/01/21/understanding-higher-order-local-gradient-computation-for-backpropagation-in-deep-neural-networks/
# # And: http://cs231n.stanford.edu/handouts/linear-backprop.pdf

# # If base op is C = X * W
# ∂C/∂X = previous_gradient * W.transpose
# ∂C/∂W = X.transpose * previous_gradient

# # If base op is C = W * X (our case)
# ∂C/∂X = W.transpose * previous_gradient
# ∂C/∂W = previous_gradient * X.transpose

proc main() =
  suite "Autograd of BLAS operations":
    test "Vector dot and matrix-vector seeded backward restore operand ranks":
      let ctx = newContext Tensor[float64]
      let a = ctx.variable([1.0, 2, 3].toTensor, requires_grad = true)
      let b = ctx.variable([3.0, 2, 1].toTensor, requires_grad = true)
      let dot = a * b
      check dot.value.rank == 0
      check dot.value.atContiguousIndex(0) == 10.0
      dot.backprop(newTensorWith[float64]([], 2.0))
      check a.grad == [6.0, 4, 2].toTensor
      check b.grad == [2.0, 4, 6].toTensor
      let matrix = ctx.variable([[1.0, 3], [2.0, 4]].toTensor.transpose, requires_grad = true)
      let vector = ctx.variable([2.0, 3].toTensor, requires_grad = true)
      let output = matrix * vector
      check output.value == [8.0, 18].toTensor
      output.backprop([1.0, 2].toTensor)
      check matrix.grad == [[2.0, 3], [4.0, 6]].toTensor
      check vector.grad == [7.0, 10].toTensor

    test "Batched vectors remove only promoted axes and sum shared vector VJPs":
      let matrices = [[[1.0, 2], [3.0, 4]], [[5.0, 6], [7.0, 8]]].toTensor
      for leftVector in [false, true]:
        let ctx = newContext Tensor[float64]
        let a = ctx.variable(matrices, requires_grad = true)
        let b = ctx.variable([2.0, 3].toTensor, requires_grad = true)
        let output = if leftVector: b * a else: a * b
        output.backprop([[1.0, 2], [3.0, 4]].toTensor)
        if leftVector:
          check output.value == [[11.0, 16], [31.0, 36]].toTensor
          check b.grad == [44.0, 64].toTensor
          check a.grad == [[[2.0, 4], [3.0, 6]], [[6.0, 8], [9.0, 12]]].toTensor
        else:
          check output.value == [[8.0, 18], [28.0, 38]].toTensor
          check b.grad == [50.0, 60].toTensor
          check a.grad == [[[2.0, 3], [4.0, 6]], [[6.0, 9], [8.0, 12]]].toTensor

    test "Empty contractions and broadcast batches keep zero gradient shapes":
      let ctx = newContext Tensor[float64]
      let a = ctx.variable(zeros[float64](2, 0), requires_grad = true)
      let b = ctx.variable(zeros[float64](0, 3), requires_grad = true)
      let output = a * b
      check output.value == zeros[float64](2, 3)
      output.backprop()
      check a.grad.shape == [2, 0]
      check b.grad.shape == [0, 3]
      let c = ctx.variable(ones[float64](1, 2, 3), requires_grad = true)
      let d = ctx.variable(zeros[float64](0, 3, 4), requires_grad = true)
      let empty = c * d
      check empty.value.shape == [0, 2, 4]
      empty.backprop()
      check c.grad == zeros[float64](1, 2, 3)
      check d.grad.shape == [0, 3, 4]

    test "Complex vector VJPs conjugate operands, not the upstream gradient":
      let ctx = newContext Tensor[Complex[float64]]
      let a = ctx.variable([complex(1.0, 1.0), complex(2.0, -1.0)].toTensor, requires_grad = true)
      let b = ctx.variable([complex(3.0, -1.0), complex(-2.0, 2.0)].toTensor, requires_grad = true)
      let output = a * b
      check output.value.atContiguousIndex(0) == complex(2.0, 8.0)
      output.backprop(newTensorWith[Complex[float64]]([], complex(2.0, -1.0)))
      check a.grad == [complex(7.0, -1.0), complex(-6.0, -2.0)].toTensor
      check b.grad == [complex(1.0, -3.0), complex(5.0, 0.0)].toTensor

    test "Gradient of matrix multiplication":

      let W = toSeq(1..8).toTensor.reshape(2,4).asType(float32)
      let X = toSeq(11..22).toTensor.reshape(4,3).asType(float32)

      let ctx = newContext Tensor[float32]

      let w_ag = ctx.variable(W, requires_grad = true)
      let x_ag = ctx.variable(X, requires_grad = true)

      let C = w_ag * x_ag

      C.backprop

      let grad_C = ones[float32](2,3)
      check: w_ag.grad == grad_C * X.transpose
      check: x_ag.grad == W.transpose * grad_C

main()
GC_fullCollect()
