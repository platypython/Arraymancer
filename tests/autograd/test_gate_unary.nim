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

## Exercises float32 unary composition and disabled gradient tracking.
import std/unittest
import ../../src/arraymancer

suite "Elementary unary autograd":
  test "Float32 unary chains preserve scalar shape and seeded VJPs":
    let ctx = newContext Tensor[float32]
    let x = ctx.variable(newTensorWith[float32]([], 4'f32), requires_grad = true)
    let output = x.exp.ln.sqrt
    output.backprop(newTensorWith[float32]([], 3'f32))
    check output.value.rank == 0
    check abs(output.value.item - 2'f32) < 1e-6'f32
    check x.grad.rank == 0
    check abs(x.grad.item - 0.75'f32) < 1e-6'f32

  test "Unary no-grad outputs do not affect later differentiation":
    let ctx = newContext Tensor[float64]
    let x = ctx.variable([1.0, 4.0].toTensor, requires_grad = true)
    ctx.no_grad_mode:
      check not x.exp.requires_grad
      check not x.ln.requires_grad
      check not x.sqrt.requires_grad
    let constant = ctx.variable([1.0, 4.0].toTensor, requires_grad = false)
    check not constant.exp.requires_grad
    check not constant.ln.requires_grad
    check not constant.sqrt.requires_grad
    x.sqrt.sum.backprop()
    check x.grad.mean_absolute_error([0.5, 0.25].toTensor) < 1e-12
