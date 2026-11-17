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

when defined(blis):
  import ./backend/blis

import  ./private/p_checks,
        ./private/p_operator_blas_l2l3,
        ./private/p_matmul,
        ./fallback/naive_l2_gemv,
        ./data_structure,
        ./init_cpu

proc gemv*[T: SomeFloat|Complex](
          alpha: T,
          A: Tensor[T],
          x: Tensor[T],
          beta: T,
          y: var Tensor[T]) {.inline.}=
  ## General Matrix-Vector multiplication:
  ## y <- alpha * A * x + beta * y
  when compileOption("boundChecks"):
    check_matvec(A,x)
    # TODO: check y + tests
  when declared(blis):
    # OpenBLAS / MKL are still faster than BLIS in the contiguous case
    # For matrix vector, the vector can be non-contiguous for MKL / OpenBLAS
    if not A.isContiguous:
      blisMV_y_eq_aAx_p_by(alpha, A, x, beta, y)
      return

  blasMV_y_eq_aAx_p_by(alpha, A, x, beta, y)

proc gemv*[T: SomeInteger](
          alpha: T,
          A: Tensor[T],
          x: Tensor[T],
          beta: T,
          y: var Tensor[T]) {.inline.}=
  ## General Matrix-Vector multiplication:
  ## y <- alpha * A * x + beta * y
  when compileOption("boundChecks"):
    check_matvec(A,x)
    # TODO: check y + tests

  naive_gemv_fallback(alpha, A, x, beta, y)

proc gemm*[T: SomeFloat|Complex](
  alpha: T, A, B: Tensor[T],
  beta: T, C: var Tensor[T]) {.inline.}=
  # Matrix: C = alpha A matmul B + beta C
  when compileOption("boundChecks"):
    check_matmat(A,B)
    # TODO: check c + tests

  when declared(blis):
    if not A.isContiguous or not B.isContiguous or not C.isContiguous:
      blisMM_C_eq_aAB_p_bC(alpha, A, B, beta, C)
      return

  blasMM_C_eq_aAB_p_bC(alpha, A, B, beta, C)

proc gemm*[T: SomeInteger](
  alpha: T, A, B: Tensor[T],
  beta: T, C: var Tensor[T]) {.inline.}=
  # Matrix: C = alpha A matmul B + beta C
  when compileOption("boundChecks"):
    check_matmat(A,B)
    # TODO: check c + tests

  fallbackMM_C_eq_aAB_p_bC(alpha, A, B, beta, C)

proc gemm*[T: SomeNumber](
  A, B: Tensor[T],
  C: var Tensor[T]) {.deprecated: "Use explicit gemm(1, A, B, 0, C) instead".}=
  gemm(1.T, A, B, 0.T, C)

proc matmulImpl[T](alpha: T, a, b: Tensor[T], beta: T): Tensor[T] {.noinit.} =
  ## Promote vectors as views, broadcast batches and write each matrix into one output allocation.
  if a.rank < 1 or b.rank < 1:
    raise newException(ValueError, "Matrix multiplication requires operands of rank >= 1")
  if a.rank == 2 and b.rank == 2:
    let M = a.shape[0]
    let K = a.shape[1]
    let N = b.shape[1]
    if K != b.shape[0]:
      raise newException(ValueError, "Matrix inner dimensions must agree for multiplication")
    result = if K == 0: zeros[T](M, N) else: newTensorUninit[T](M, N)
    if M != 0 and N != 0 and K != 0:
      gemm(alpha, a, b, beta, result)
    return
  if a.rank == 1 and b.rank == 2 and a.strides[0] == 1 and b.isContiguous:
    let K = a.shape[0]
    let N = b.shape[1]
    if K != b.shape[0]:
      raise newException(ValueError, "Matrix inner dimensions must agree for multiplication")
    result = if K == 0: zeros[T](N) else: newTensorUninit[T](N)
    if N != 0 and K != 0:
      var matrix = b
      swap(matrix.shape[0], matrix.shape[1])
      swap(matrix.strides[0], matrix.strides[1])
      gemv(alpha, matrix, a, beta, result)
    return
  let left = promoteMatmulOperand(a, false)
  let right = promoteMatmulOperand(b, true)
  let M = left.shape[left.rank - 2]
  let K = left.shape[left.rank - 1]
  let N = right.shape[right.rank - 1]
  if K != right.shape[right.rank - 2]:
    raise newException(ValueError, "Matrix inner dimensions must agree for multiplication")

  if a.rank == 2 and b.rank == 1:
    result = if K == 0: zeros[T](M) else: newTensorUninit[T](M)
    if M != 0 and K != 0:
      gemv(alpha, a, b, beta, result)
    return

  let a_batch_rank = left.rank - 2
  let b_batch_rank = right.rank - 2
  let out_batch_rank = max(a_batch_rank, b_batch_rank)
  var batchShape: Metadata
  batchShape.len = out_batch_rank
  for i in 0 ..< out_batch_rank:
    let ai = i - (out_batch_rank - a_batch_rank)
    let bi = i - (out_batch_rank - b_batch_rank)
    let ad = if ai < 0: 1 else: left.shape[ai]
    let bd = if bi < 0: 1 else: right.shape[bi]
    if ad == bd: batchShape[i] = ad
    elif ad == 1: batchShape[i] = bd
    elif bd == 1: batchShape[i] = ad
    else: raise newException(ValueError, "Incompatible matrix batch dimensions")

  var outShape = batchShape
  outShape.add(M)
  outShape.add(N)
  result = if K == 0: zeros[T](outShape) else: newTensorUninit[T](outShape)
  var batches = if result.size == 0 or K == 0: 0 else: 1
  for dim in batchShape: batches *= dim
  for batch in 0 ..< batches:
    var curr = batch
    var aOff = left.offset
    var bOff = right.offset
    var outOff = result.offset
    for i in countdown(out_batch_rank - 1, 0):
      let coord = curr mod batchShape[i]
      curr = curr div batchShape[i]
      outOff += coord * result.strides[i]
      let ai = i - (out_batch_rank - a_batch_rank)
      let bi = i - (out_batch_rank - b_batch_rank)
      if ai >= 0 and left.shape[ai] != 1: aOff += coord * left.strides[ai]
      if bi >= 0 and right.shape[bi] != 1: bOff += coord * right.strides[bi]

    var aSlice = left
    aSlice.shape = [M, K].toMetadata
    aSlice.strides = [left.strides[left.rank - 2], left.strides[left.rank - 1]].toMetadata
    aSlice.offset = aOff
    var bSlice = right
    bSlice.shape = [K, N].toMetadata
    bSlice.strides = [right.strides[right.rank - 2], right.strides[right.rank - 1]].toMetadata
    bSlice.offset = bOff
    var outSlice = result
    outSlice.shape = [M, N].toMetadata
    outSlice.strides = [result.strides[result.rank - 2], result.strides[result.rank - 1]].toMetadata
    outSlice.offset = outOff
    gemm(alpha, aSlice, bSlice, beta, outSlice)

  if a.rank == 1:
    result.shape.delete(result.rank - 2)
    result.strides.delete(result.strides.len - 2)
  if b.rank == 1:
    result.shape.delete(result.rank - 1)
    result.strides.delete(result.strides.len - 1)

proc `*`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Matrix product with trailing-axis batch broadcasting.
  ## A left vector is promoted to a row and a right vector to a column; promoted
  ## axes are removed from the result. Two vectors therefore return a rank-zero
  ## dot product. Scalars and incompatible inner/batch dimensions raise ValueError.
  ## Empty contractions produce zeros. The output owns one new allocation.
  matmulImpl(1.T, a, b, 0.T)

proc `*`*[T: Complex[float32] or Complex[float64]](
      a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Complex matrix product with the same vector and batch rules as real tensors.
  ## Products do not conjugate either operand.
  type F = T.T
  matmulImpl(complex(1.F, 0.F), a, b, complex(0.F, 0.F))
