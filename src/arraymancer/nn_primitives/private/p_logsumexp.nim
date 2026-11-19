# Copyright 2017 the Arraymancer contributors
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

import  ../../tensor,
        math

## Provides streaming normalizers for softmax cross-entropy kernels.

# - Classic version: log ∑i exp(xi) = α + log ∑i exp(xi−α)
# with α = max(xi) for xi in x

# - Streaming version: from http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
# which is similar to Welford algorithm for streaming mean and variance in statistics

# Benchmark shows that the streaming version is 25% faster
# from batches of 256x3 classes to 256x1000 classes
# (see project_root/benchmark_implementation)

proc streaming_max_sumexp*[T](t: Tensor[T]): tuple[max:T, sumexp: T] {.noSideEffect, inline.}=
  result.max = -Inf.T   # will store the streaming max of the tensor  # TODO: replace by low(T) when 0.18 for https://github.com/nim-lang/Nim/commit/badba83d38371726bafba5870d5fb927eb453e41
  result.sumexp = 0.T   # will store the streaming sum of exp of the tensor

  for x in t:
    if x <= result.max:
      result.sumexp += exp(x - result.max)
    else:
      result.sumexp = result.sumexp * exp(result.max - x) + 1
      result.max = x

proc streaming_max_sumexp*[T](t: Tensor[T], axis: int): Tensor[tuple[max:T, sumexp: T]] {.noinit.}=
  # Only 2D tensor are supported for now. (i.e. no 3D Softmax)
  assert axis in {0, 1} # TODO proper error message/check

  result = newTensorUninit[tuple[max:T, sumexp: T]](t.shape[axis])

  for i in `||`(0, t.shape[axis]-1, "simd"):
    result.data[i] = t.atAxisIndex(axis, i).streaming_max_sumexp

  # Reexpand the tensor to be consistent with fold_axis/reduce_axis
  if axis == 0:
    result = result.unsqueeze(1)
  else:
    result = result.unsqueeze(0)


proc stable_softmax*[T](x, max, sumexp: T): T {.noSideEffect, inline.}=
  # Numerically stable streaming softmax helper
  result = exp(x - max) / sumexp

