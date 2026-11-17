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

import ../../tensor

proc reduce_broadcast_dims*[T](grad: Tensor[T], target_shape: Metadata): Tensor[T] =
  ## Sum a broadcasted gradient back to `target_shape`.
  if grad.shape == target_shape:
    return grad
  if grad.size == 0:
    return zeros[T](target_shape)
  result = grad
  while result.rank > target_shape.len:
    var shape = result.shape
    for i in 0 ..< shape.len - 1:
      shape[i] = shape[i + 1]
    dec shape.len
    result = result.sum(axis = 0).reshape(shape)
  for i in 0 ..< target_shape.len:
    if target_shape[i] == 1 and result.shape[i] > 1:
      result = result.sum(axis = i)
  if result.shape != target_shape:
    result = result.reshape(target_shape)

when defined(metal) or defined(coreai):
  proc reduce_broadcast_dims*(grad: MetalTensor[float32]; target_shape: Metadata): MetalTensor[float32] =
    ## Reduce on-device, including empty broadcast axes and rank-zero targets.
    result = grad
    if result.rank < target_shape.len:
      raise newException(MetalError, "Broadcast gradient rank is smaller than its target")
    while result.rank > target_shape.len:
      result = result.sum(axis = 0).squeeze(0)
    for axis in 0 ..< target_shape.len:
      if target_shape[axis] == 1 and result.shape[axis] != 1:
        result = result.sum(axis = axis)
    if result.shape != target_shape:
      result = result.reshape(target_shape)
