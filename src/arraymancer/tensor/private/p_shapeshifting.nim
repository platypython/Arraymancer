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

import  ../../laser/tensor/initialization,
        ../data_structure, ../higher_order_applymap,
        ../init_cpu,
        ./p_checks,
        nimblas

import std / sequtils

proc contiguousImpl*[T](t: Tensor[T], layout: OrderType, result: var Tensor[T]) =
  if layout == rowMajor:
    result = t.map_inline(x)
  else: # colMajor
    var size: int
    initTensorMetadata(result, size, t.shape, colMajor)
    allocCpuStorage(result.storage, size)
    apply2_inline(result, t):
      y

proc reshape_no_copy*(t: AnyTensor, new_shape: varargs[int]|Metadata|seq[int], result: var AnyTensor, layout: OrderType) {.noSideEffect.}=
  result.shape.copyFrom(new_shape)
  shape_to_strides(result.shape, layout, result.strides)
  result.offset = t.offset

proc reshape_with_copy*[T](t: Tensor[T], new_shape: varargs[int]|Metadata|seq[int], result: var Tensor[T]) =
  contiguousImpl(t, rowMajor, result)
  reshape_no_copy(t, new_shape, result, rowMajor)
  result.offset = 0 # Offset needs to be reset! Copy is a new tensor of `new_shape`

proc infer_shape*(t: Tensor, new_shape: varargs[int]): seq[int] {.noinit.} =
  ## Replace the single -1 value on `new_shape` with the value that
  ## makes the size the same as that of the input tensor
  result = new_shape.toSeq
  var auto_axis = -1
  var auto_axis_count = 0
  for n in 0 .. result.high:
    if result[n] == -1:
      auto_axis_count += 1
      auto_axis = n
      break
  if auto_axis_count > 1:
    raise newException(ValueError, "Only one dimension can be inferred by inferShape")
  elif auto_axis_count == 0:
    when compileOption("boundChecks"):
      raise newException(ValueError, "At least one dimension must be inferred by inferShape")
  else:
    result[auto_axis] = t.size div result.filterIt(it != -1).prod

proc reshapeImpl*(t: AnyTensor, new_shape: varargs[int]|Metadata|seq[int],
    result: var AnyTensor, infer: static bool) =
  when infer:
    let new_shape = t.infer_shape(new_shape)

  when compileOption("boundChecks"):
    check_reshape(t, new_shape)

  if t.is_C_contiguous:
    reshape_no_copy(t, new_shape, result, rowMajor)
    result.storage = t.storage
  else:
    reshape_with_copy(t, new_shape, result)

proc broadcastImpl*(t: var AnyTensor, shape: varargs[int]|Metadata) {.noSideEffect.}=
  when compileOption("boundChecks"):
    assert t.rank == shape.len

  for i in 0..<t.rank:
    if t.shape[i] == 1:
      if shape[i] != 1:
        t.shape[i] = shape[i]
        t.strides[i] = 0
    elif t.shape[i] != shape[i]:
      # TODO: $varargs is not supported on stable
      var strShape = "["
      for i in shape:
        if i != 0: strShape.add ','
        strShape.add ' '
        strShape.add $i
      strShape.add ']'
      raise newException(ValueError, "The broadcasted size of the tensor " & strShape &
        ", must match existing size " & $t.shape & " for non-singleton dimension")

proc broadcast2Impl*[T](a, b: AnyTensor[T], result: var tuple[a, b: AnyTensor[T]]) {.noSideEffect.}=
  let rank = max(a.rank, b.rank)

  var shapeA, stridesA, shapeB, stridesB = Metadata(len: rank) # initialized with 0
  let leadingA = rank - a.rank
  let leadingB = rank - b.rank

  for i in 0..<rank:
    # Missing leading dimensions behave like singleton axes with zero stride.
    let axisA = i - leadingA
    let axisB = i - leadingB
    let shape_A_iter = if axisA >= 0: a.shape[axisA] else: 1
    let shape_B_iter = if axisB >= 0: b.shape[axisB] else: 1

    if shape_A_iter == shape_B_iter:
      shapeA[i] = shape_A_iter
      shapeB[i] = shape_A_iter

      stridesA[i] = if axisA >= 0: a.strides[axisA] else: 0
      stridesB[i] = if axisB >= 0: b.strides[axisB] else: 0

    elif shape_A_iter == 1:
      shapeA[i] = shape_B_iter
      shapeB[i] = shape_B_iter

      # stridesA[i] is already 0
      stridesB[i] = if axisB >= 0: b.strides[axisB] else: 0
    elif shape_B_iter == 1:
      shapeA[i] = shape_A_iter
      shapeB[i] = shape_A_iter

      stridesA[i] = if axisA >= 0: a.strides[axisA] else: 0
      # stridesB[i] is already 0
    else:
      raise newException(ValueError, "Broadcasting error: non-singleton dimensions must be the same in both tensors. Tensors' shapes were: " &
        $a.shape & " and " & $b.shape)

  result.a.shape = shapeA
  result.a.strides = stridesA
  result.a.offset = a.offset

  result.b.shape = shapeB
  result.b.strides = stridesB
  result.b.offset = b.offset


proc exch_dim*[T](t: Tensor[T], dim1, dim2: int): Tensor[T] {.noinit,noSideEffect.}=
  if dim1 == dim2:
    return t

  result = t # copy or no-copy is managed in the caller of exch_dim or permuteImpl
  swap(result.strides[dim1], result.strides[dim2])
  swap(result.shape[dim1], result.shape[dim2])

proc permuteImpl*[T](result: var Tensor[T], dims: varargs[int]) {.noSideEffect.} =
  var perm = dims.toMetadata
  for i, p in perm:
    if p != i and p != -1:
      var j = i
      while true:
        result = exch_dim(result, j, perm[j])
        (perm[j], j) = (-1, perm[j])
        if perm[j] == i:
          break
      perm[j] = -1

proc squeezeImpl*(t: var AnyTensor) {.noSideEffect.} =
  var idx_real_dim = 0

  for i in 0 ..< t.rank:
    if t.shape[i] != 1:
      if i != idx_real_dim:
        t.shape[idx_real_dim] = t.shape[i]
        t.strides[idx_real_dim] = t.strides[i]
      inc idx_real_dim

  t.shape.len = idx_real_dim
  t.strides.len = idx_real_dim

proc squeezeImpl*(t: var AnyTensor, axis: int) {.noSideEffect.} =
  check_squeezeAxis(t, axis)
  if t.rank == 0: return
  let axis = if axis < 0: axis + t.rank else: axis
  if t.shape[axis] == 1:
    t.shape.delete(axis)
    t.strides.delete(axis)

proc unsqueezeImpl*(t: var AnyTensor, axis: int) {.noSideEffect.} =
  check_unsqueezeAxis(t, axis)
  let axis = if axis < 0: axis + t.rank + 1 else: axis

  # Keep inserted singleton strides consistent with reshape and PyTorch.
  var stride: int
  if axis >= t.rank:
    stride = 1
  else:
    stride = t.shape[axis]*t.strides[axis]

  t.shape.insert(1, axis)
  t.strides.insert(stride, axis)
