# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Initializes fresh Metal result slots without an owning tensor temporary.
import ../[metal_types, data_structure]

template initMetalTensorUninit*[T: KnownSupportsCopyMem](
    output: var MetalTensor[T]; shapeValue: Metadata; deviceValue: MetalDevice) =
  ## Resolve the device expression only after dtype and shape validation.
  ## Callers pass stable metadata, a resolved device and a fresh result slot.
  block:
    discard metalDtype[T]()
    let count = checkedMetalSize(shapeValue)
    if count > high(int) div sizeof(T):
      raise newException(MetalError, "Metal byte size overflow")
    let dev = deviceValue
    let bytes = uint64(count * sizeof(T))
    var handle: pointer
    checkMetal metalApi().buffer(dev.owner.handle, bytes, addr handle)
    output.storage = MetalStorage(owner: ownMetal(handle), device: dev, capacity: bytes)
    output.shape = shapeValue
    output.offset = 0
    shape_to_strides(shapeValue, rowMajor, output.strides)
