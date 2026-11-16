# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides replayable GPU random generation using native Philox state.
import ./[metal_types, init_metal]
import ../laser/tensor/datatypes

export metal_types

const MetalRngStateVersion* = 1'u32

proc requireGenerator(generator: MetalGenerator) =
  if generator == nil or generator.owner == nil or generator.owner.handle == nil:
    raise newException(MetalRngStateError, "Metal generator handle is nil")
  if generator.device == nil or generator.device.owner == nil or
      generator.device.owner.handle == nil:
    raise newException(MetalRngStateError, "Metal generator device is nil")

proc makeMetalGenerator(seed: uint64; device: MetalDevice; mode: int32): MetalGenerator =
  let dev = if device == nil: metalDevice() else: device
  if dev.owner == nil or dev.owner.handle == nil:
    raise newException(MetalRngStateError, "Metal generator requires a valid device")
  var handle: pointer
  checkMetal metalApi().generator(dev.owner.handle, seed, mode, addr handle)
  result = MetalGenerator(owner: ownMetal(handle), device: dev)

proc newMetalGenerator*(seed: uint64; device: MetalDevice = nil): MetalGenerator =
  ## Creates an explicitly seeded generator with its next-block counter at zero.
  makeMetalGenerator(seed, device, 0)

proc defaultMetalGenerator*(device: MetalDevice = nil): MetalGenerator =
  ## Returns the retained native default generator for a device, initially OS-seeded.
  makeMetalGenerator(0'u64, device, 1)

proc manualSeed*(generator: MetalGenerator; seed: uint64) =
  ## Reseeds deterministically and resets the Philox next-block counter to zero.
  requireGenerator(generator)
  var value = seed
  checkMetal metalApi().rngSeed(generator.owner.handle, addr value, 0)

proc seed*(generator: MetalGenerator): uint64 =
  ## Reseeds from OS entropy and returns the actual seed chosen by the native bridge.
  requireGenerator(generator)
  checkMetal metalApi().rngSeed(generator.owner.handle, addr result, 1)

proc getRngState*(generator: MetalGenerator): MetalRngState =
  ## Snapshots the reservation state immediately; it does not wait for pending GPU work.
  requireGenerator(generator)
  var state: AMRngState
  checkMetal metalApi().rngState(generator.owner.handle, addr state, 0)
  if state.version != MetalRngStateVersion or state.reserved != 0:
    raise newException(MetalRngStateError, "Native Metal RNG returned an incompatible state")
  result.version = state.version
  result.seed = state.seed
  result.counter = state.counter

proc setRngState*(generator: MetalGenerator; state: MetalRngState) =
  ## Restores a version-1 seed and little-endian 128-bit next-block counter.
  requireGenerator(generator)
  if state.version != MetalRngStateVersion:
    raise newException(MetalRngStateError, "Unsupported Metal RNG state version")
  var nativeState: AMRngState
  nativeState.version = state.version
  nativeState.reserved = 0
  nativeState.seed = state.seed
  nativeState.counter = state.counter
  checkMetal metalApi().rngState(generator.owner.handle, addr nativeState, 1)

proc randMetal*(shape: Metadata; generator: MetalGenerator): MetalTensor[float32] =
  ## Generates uniform [0,1) float32 values on the GPU, reserving ceil(size/4)
  ## Philox blocks at enqueue time. Zero-element tensors consume zero blocks; scalars consume one.
  requireGenerator(generator)
  result = newMetalTensorUninit[float32](shape, generator.device)
  checkMetal metalApi().rand(generator.owner.handle, result.storage.owner.handle,
    descriptor(result), arraymancerAppleLibDir.cstring)

proc randMetal*(shape: varargs[int]; generator: MetalGenerator): MetalTensor[float32] =
  if shape.len > min(LASER_MAXRANK, 8):
    raise newException(MetalError, "Excessive Metal rank")
  var metadata: Metadata
  metadata.len = shape.len
  for i, dim in shape: metadata[i] = dim
  randMetal(metadata, generator)
