# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides shared native owners and Metal-only metadata; never a CPU data pointer.
import ./backend/metal_abi
import ../laser/[dynamic_stack_arrays]
import ../laser/tensor/datatypes
import std/complex
export metal_abi, dynamic_stack_arrays

type
  MetalFloat16* = distinct uint16
  MetalBFloat16* = distinct uint16
  MetalComplexFloat16* = object
    re*, im*: MetalFloat16
  MetalOwnerObj = object
    handle*: pointer
    release: proc(handle: pointer) {.cdecl, raises: [].}
  MetalOwner* = ref MetalOwnerObj
  MetalDevice* = ref object
    owner*: MetalOwner
    registryId*: uint64
  MetalStream* = ref object
    owner*: MetalOwner
    device*: MetalDevice
  MetalStorage* = ref object
    owner*: MetalOwner
    device*: MetalDevice
    capacity*: uint64
  MetalTensor*[T] = object
    ## Assignment shares storage; metadata remains independent.
    shape*, strides*: Metadata
    offset*: int
    storage*: MetalStorage

proc dispose(owner: var MetalOwnerObj) =
  if owner.handle != nil:
    owner.release(owner.handle)
    owner.handle = nil
when defined(gcDestructors):
  proc `=destroy`(owner: var MetalOwnerObj) = dispose(owner)
else:
  proc finalize(owner: MetalOwner) = dispose(owner[])

proc ownMetal*(handle: pointer): MetalOwner =
  if handle == nil: raise newException(MetalError, "Native Metal returned a nil handle")
  when defined(gcDestructors): new(result)
  else: new(result, finalize)
  result.handle = handle
  result.release = metalApi().release

func checkedMetalSize*(shape: Metadata): int =
  ## Check suffix products too: empty tensors still require representable strides.
  var suffix = 1
  for axis in countdown(shape.len - 1, 0):
    let dim = shape[axis]
    if dim < 0: raise newException(MetalError, "Negative Metal dimension")
    if dim != 0 and suffix > high(int) div dim:
      raise newException(MetalError, "Metal shape overflow")
    suffix *= dim
  discard suffix
  shape.shapeProduct

func rank*[T](t: MetalTensor[T]): int = t.shape.len
func size*[T](t: MetalTensor[T]): int =
  if t.storage == nil: 0 else: checkedMetalSize(t.shape)
func is_C_contiguous*[T](t: MetalTensor[T]): bool =
  var stride = 1
  for axis in countdown(t.rank - 1, 0):
    if t.shape[axis] != 1 and t.strides[axis] != stride: return false
    stride *= t.shape[axis]
  true

func fromBits*(typ: typedesc[MetalFloat16]; bits: uint16): MetalFloat16 = MetalFloat16(bits)
func fromBits*(typ: typedesc[MetalBFloat16]; bits: uint16): MetalBFloat16 = MetalBFloat16(bits)
func toBits*(value: MetalFloat16): uint16 = uint16(value)
func toBits*(value: MetalBFloat16): uint16 = uint16(value)

proc metalDtype*[T](): int32 =
  ## Native storage tags are distinct from Torch's complex total-bit-width names.
  when T is float32: 1
  elif T is MetalFloat16: 2
  elif T is MetalBFloat16: 3
  elif T is int8: 4
  elif T is int16: 5
  elif T is int32: 6
  elif T is int64: 7
  elif T is uint8: 8
  elif T is uint16: 9
  elif T is uint32: 10
  elif T is uint64: 11
  elif T is bool: 12
  elif T is MetalComplexFloat16: 13
  elif T is Complex32: 14
  else: raise newException(MetalDtypeError, "Unsupported Metal storage dtype")
