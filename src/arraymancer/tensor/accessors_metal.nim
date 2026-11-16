# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides checked single-element GPU mutation of nonoverlapping native views.
import ./[metal_types, init_metal, shapeshifting_metal]

proc setElement*[T](t: var MetalTensor[T]; indices: openArray[int]; value: T) =
  ## Mutate the shared owner with a native command, never a host-buffer replacement.
  requireMetalMutable(t)
  if indices.len != t.rank: raise newException(MetalError, "Metal index rank mismatch")
  var offset = t.offset
  for axis, index in indices:
    if index < 0 or index >= t.shape[axis]: raise newException(MetalError, "Metal index outside shape")
    offset += index * t.strides[axis]
  var output = descriptor(t)
  output.rank = 0
  output.offset = int64(offset)
  checkMetal metalApi().setElement(t.storage.owner.handle, output,
    unsafeAddr value, arraymancerAppleLibDir.cstring)

proc `[]=`*[T](t: var MetalTensor[T]; i: int; value: T) = t.setElement([i], value)
proc `[]=`*[T](t: var MetalTensor[T]; i, j: int; value: T) = t.setElement([i, j], value)
