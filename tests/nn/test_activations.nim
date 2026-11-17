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

## Checks classic activation edge cases and GELU numerical gradients.
import ../../src/arraymancer
import std / [unittest, math]

proc main() =
  suite "Classic activation IEEE behavior":
    test "ReLU preserves NaNs and signed zero while masking inactive NaN seeds":
      let input = [-0.0'f32, NaN.float32, -1'f32, 1'f32].toTensor
      let ctx = newContext Tensor[float32]
      let x = ctx.variable(input, requires_grad = true)
      let output = x.relu
      output.backprop([NaN.float32, 1'f32, NaN.float32, NaN.float32].toTensor)
      check output.value[0].classify == fcNegZero
      check output.value[1].classify == fcNan
      check output.value[2] == 0'f32
      check output.value[3] == 1'f32
      check x.grad[0] == 0'f32
      check x.grad[1] == 1'f32
      check x.grad[2] == 0'f32
      check x.grad[3].classify == fcNan
      var inplace = input.clone
      inplace.mrelu
      check inplace[0].classify == fcNegZero
      check inplace[1].classify == fcNan
      check inplace[2] == 0'f32
      check inplace[3] == 1'f32

    test "ReLU retains integer types and mutates transposed views in place":
      let original = [[low(int32), 0'i32, 2'i32],
                      [4'i32, -5'i32, high(int32)]].toTensor
      var view = original.transpose
      let expected = [[0'i32, 4'i32], [0'i32, 0'i32],
                      [2'i32, high(int32)]].toTensor
      check view.relu == expected
      check original[0, 0] == low(int32)
      view.mrelu
      check view == expected
      check original == expected.transpose
      let unsigned = [0'u64, high(uint64)].toTensor
      check unsigned.relu == [0'u64, high(uint64)].toTensor

  suite "GELU activation":
    test "GELU forward matches the closed form":
      let ctx = newContext Tensor[float64]
      let x = randomTensor[float64]([3, 4], 4.0) -. 2.0

      ctx.no_grad_mode:
        let y = gelu(ctx.variable(x)).value
        check: y.shape == @[3, 4]
        for i in 0 ..< 3:
          for j in 0 ..< 4:
            let e = 0.5 * x[i, j] * (1 + erf(x[i, j] / sqrt(2.0)))
            check: abs(y[i, j] - e) < 1e-12

    test "GELU backward with numerical gradient":
      let ctx = newContext Tensor[float64]
      let x = randomTensor[float64]([2, 5], 2.0) -. 1.0
      let grad_out = randomTensor[float64]([2, 5], 1.0)
      let vx = ctx.variable(x, requires_grad = true)
      let loss = (gelu(vx) *. ctx.variable(grad_out)).sum()
      loss.backprop()

      proc loss_fn(inp: Tensor[float64]): float64 =
        let c = newContext Tensor[float64]
        c.no_grad_mode:
          result = (gelu(c.variable(inp)).value *. grad_out).sum

      check: vx.grad.mean_relative_error(numerical_gradient(x, loss_fn)) < 1e-6

    test "QuickGELU forward matches x * sigmoid(1.702 x)":
      let ctx = newContext Tensor[float64]
      let x = randomTensor[float64]([3, 4], 4.0) -. 2.0

      ctx.no_grad_mode:
        let y = quickGelu(ctx.variable(x)).value
        for i in 0 ..< 3:
          for j in 0 ..< 4:
            let e = x[i, j] / (1.0 + exp(-1.702 * x[i, j]))
            check: abs(y[i, j] - e) < 1e-12

    test "QuickGELU backward with numerical gradient":
      let ctx = newContext Tensor[float64]
      let x = randomTensor[float64]([2, 5], 2.0) -. 1.0
      let grad_out = randomTensor[float64]([2, 5], 1.0)
      let vx = ctx.variable(x, requires_grad = true)
      let loss = (quickGelu(vx) *. ctx.variable(grad_out)).sum()
      loss.backprop()

      proc loss_fn(inp: Tensor[float64]): float64 =
        let c = newContext Tensor[float64]
        c.no_grad_mode:
          result = (quickGelu(c.variable(inp)).value *. grad_out).sum

      check: vx.grad.mean_relative_error(numerical_gradient(x, loss_fn)) < 1e-6

  echo ""

main()
