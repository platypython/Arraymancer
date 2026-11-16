# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises native storage, explicit placement and unavailable-library behavior.
import std/unittest
import ../../src/arraymancer

when defined(missingMetalLibrary):
  let input = [1'f32, 2, 3, 4, 5, 6].toTensor
  var failed = false
  try: discard input.metal
  except MetalUnavailableError: failed = true
  doAssert failed
  doAssert input == [1'f32, 2, 3, 4, 5, 6].toTensor
else:
  suite "Native Metal storage":
    test "Round trip preserves values and original storage":
      let input = [1'f32, 2, 3, 4, 5, 6].toTensor.reshape(2, 3)
      let dev = metalDevice()
      let native = input.metal(dev)
      check native.storage.device.registryId == dev.registryId
      check dev.registryId != 0
      check native.cpu == input
      check input == [1'f32, 2, 3, 4, 5, 6].toTensor.reshape(2, 3)
      check metalIsAvailable()
      check metalDeviceCount() >= 1
    test "Scalars empties and sentinels differ":
      let scalar = newTensorWith[float32]([], 7'f32).metal
      check scalar.size == 1
      check scalar.cpu.size == 1
      var sentinel: MetalTensor[float32]
      check sentinel.size == 0
      let empty = newMetalTensorUninit[float32](2, 0, 3)
      check empty.size == 0
      check empty.cpu.shape == [2, 0, 3].toMetadata
    test "Invalid requests fail explicitly":
      expect MetalUnavailableError: discard metalDevice(-1)
      expect MetalUnavailableError: discard metalDevice(high(int))
      expect MetalDtypeError: discard newMetalTensorUninit[float64](2)
      expect MetalError: discard newMetalTensorUninit[float32](-1)
      expect MetalError: discard newMetalTensorUninit[float32](high(int), 2)
      expect MetalError: discard newMetalTensorUninit[float32](newSeq[int](100))
