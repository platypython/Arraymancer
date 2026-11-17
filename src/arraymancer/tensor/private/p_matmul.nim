# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides metadata-only vector promotion for matrix products.

func promoteMatmulOperand*[TT](t: TT, column: bool): TT =
  ## A vector borrows a singleton row/column axis; its storage and element stride stay shared.
  result = t
  if t.shape.len == 1:
    result.shape.len = 2
    result.strides.len = 2
    if column:
      result.shape[1] = 1
      result.strides[1] = 0
    else:
      result.shape[1] = t.shape[0]
      result.shape[0] = 1
      result.strides[1] = t.strides[0]
      result.strides[0] = 0

func promoteMatmulGradient*[TT](t: TT, leftVector, rightVector: bool): TT =
  ## Restore only vector-promotion axes, without changing batch axes or copying a seed.
  result = t
  if leftVector:
    let axis = if rightVector: t.shape.len else: t.shape.len - 1
    inc result.shape.len
    inc result.strides.len
    for i in countdown(result.shape.len - 1, axis + 1):
      result.shape[i] = result.shape[i - 1]
      result.strides[i] = result.strides[i - 1]
    result.shape[axis] = 1
    result.strides[axis] = 0
  if rightVector:
    result.shape.add(1)
    result.strides.add(0)
