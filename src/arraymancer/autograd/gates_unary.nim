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

## Provides real floating-point exponential, logarithm and square-root VJPs.
import ../tensor, ./autograd_common

type
  UnaryKind = enum unaryExp, unaryLn, unarySqrt
  UnaryGate[TT; kind: static[UnaryKind]] {.final.} = ref object of Gate[TT]
    cache: TT

proc sqrtGradient[T: SomeFloat](gradient, output: Tensor[T]): Tensor[T] =
  # Fuse reciprocal scaling instead of allocating a denominator tensor.
  map2_inline(gradient, output, x / (T(2) * y))

when defined(metal) or defined(coreai):
  proc sqrtGradient(gradient, output: MetalTensor[float32]): MetalTensor[float32] =
    # Operation 13 fuses grad / (2 * output) in one native dispatch.
    result = newMetalTensorUninit[float32](gradient.shape, gradient.storage.device)
    dispatchMetalMath(gradient, output, result, 13)

proc unaryBackward[TT; kind: static[UnaryKind]](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gate = UnaryGate[TT, kind](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  when kind == unaryExp:
    result[0] = gradient *. gate.cache
  elif kind == unaryLn:
    result[0] = gradient /. gate.cache
  else:
    result[0] = sqrtGradient(gradient, gate.cache)

proc unary[TT](a: Variable[TT], value: TT, kind: static[UnaryKind]): Variable[TT] =
  ## Specialize the callback so backward needs no runtime operation tag.
  new result
  result.context = a.context
  result.value = value
  if a.is_grad_needed:
    var gate: UnaryGate[TT, kind]
    new gate
    when kind == unaryLn: gate.cache = a.value
    else: gate.cache = result.value
    result.grad = zeros_like(result.value)
    result.requires_grad = true
    const name = ["Exp", "Ln", "Sqrt"][ord(kind)]
    register_node(name, gate, unaryBackward[TT, kind], result, a)

proc exp*[T: SomeFloat](a: Variable[Tensor[T]]): Variable[Tensor[T]] =
  ## Element-wise real exponential with a shape-preserving VJP.
  unary(a, a.value.exp, unaryExp)

proc ln*[T: SomeFloat](a: Variable[Tensor[T]]): Variable[Tensor[T]] =
  ## Element-wise real natural logarithm; backward divides by the input.
  unary(a, a.value.ln, unaryLn)

proc sqrt*[T: SomeFloat](a: Variable[Tensor[T]]): Variable[Tensor[T]] =
  ## Element-wise real square root; zero/domain derivatives retain IEEE results.
  unary(a, a.value.sqrt, unarySqrt)

when defined(metal) or defined(coreai):
  proc exp*(a: Variable[MetalTensor[float32]]): Variable[MetalTensor[float32]] =
    ## GPU-resident float32 exponential and VJP, without host readback.
    unary(a, a.value.exp, unaryExp)

  proc ln*(a: Variable[MetalTensor[float32]]): Variable[MetalTensor[float32]] =
    ## GPU-resident float32 natural logarithm and VJP.
    unary(a, a.value.ln, unaryLn)

  proc sqrt*(a: Variable[MetalTensor[float32]]): Variable[MetalTensor[float32]] =
    ## GPU-resident float32 square root with one fused backward dispatch.
    unary(a, a.value.sqrt, unarySqrt)
