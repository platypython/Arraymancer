# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides deterministic GPU reductions with GPU-only partial buffers.
import ./[metal_types, init_metal, shapeshifting_metal, operators_metal, data_structure]

proc reduce(t: MetalTensor[float32]; normalize: bool): MetalTensor[float32] =
  ## Normalize in the final GPU pass to avoid allocating a second scalar.
  validateMetalView(t)
  var input = t
  while true:
    let groups = if input.size <= 65536: 1 else:
      input.size div 256 + int(input.size mod 256 != 0)
    var shape: Metadata
    if groups != 1:
      shape.len = 1
      shape[0] = groups
    let output = newMetalTensorUninit[float32](shape, t.storage.device)
    dispatchMetalMath(input, default(MetalTensor[float32]), output, 21,
      scalar = float32(t.size), scalarInput = normalize and groups == 1)
    if groups == 1: return output
    input = output

proc sum*(t: MetalTensor[float32]): MetalTensor[float32] =
  ## Full reduction returns a native scalar. Empty input sums to zero.
  reduce(t, false)

proc mean*(t: MetalTensor[float32]): MetalTensor[float32] =
  ## Empty mean is a native scalar NaN (GPU zero divided by zero).
  reduce(t, true)

proc sum*(t: MetalTensor[float32]; axis: int): MetalTensor[float32] =
  ## Keep the reduced dimension at length one, including empty-axis sums.
  validateMetalView(t)
  if axis < 0 or axis >= t.rank: raise newException(MetalError, "Metal sum axis outside rank")
  var shape = t.shape
  shape[axis] = 1
  result = newMetalTensorUninit[float32](shape, t.storage.device)
  dispatchMetalMath(t, default(MetalTensor[float32]), result, 22, axis = axis)
