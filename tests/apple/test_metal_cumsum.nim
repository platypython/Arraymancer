# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises native Metal32 cumulative scans and their reverse-mode adjoints.
import std/[math, unittest]
import ../../src/arraymancer

suite "Native Metal float32 cumulative scans":
  test "Packed subgroup and 128-value tiles carry through tails and long rows":
    for count in [1, 31, 32, 33, 127, 128, 129]:
      let input = ones[float32](count).metal
      let forward = input.cumsum.cpu
      let reverse = input.cumsumGradient(0).cpu
      for index in 0 ..< count:
        check forward[index] == float32(index + 1)
        check reverse[index] == float32(count - index)

    const count = 65539
    let longInput = ones[float32](count).metal
    let forward = longInput.cumsum.cpu
    let reverse = longInput.cumsumGradient(0).cpu
    for index in [0, 31, 32, 127, 128, 65535, 65536, count - 1]:
      check forward[index] == float32(index + 1)
      check reverse[index] == float32(count - index)

  test "Column tiles scan across block carries and preserve 3D axes":
    var host = newTensorUninit[float32](2, 131, 9)
    for batch in 0 ..< 2:
      for axis in 0 ..< 131:
        for inner in 0 ..< 9:
          host[batch, axis, inner] = float32(batch * 1000 + axis * 3 + inner)
    let input = host.metal
    for axis in [0, 1, -1]:
      check input.cumsum(axis).cpu == host.cumsum(axis)
    var expectedReverse = newTensorUninit[float32](2, 131, 9)
    for batch in 0 ..< 2:
      for inner in 0 ..< 9:
        var suffix = 0'f32
        for axis in countdown(130, 0):
          suffix += host[batch, axis, inner]
          expectedReverse[batch, axis, inner] = suffix
    check input.cumsumGradient(1).cpu == expectedReverse

  test "Offset, transposed, negative-stride and broadcast views scan natively":
    let owner = [[0'f32, 1, 2, 3], [4'f32, 5, 6, 7],
                 [8'f32, 9, 10, 11]].toTensor.metal
    let offset = owner.metalView([2, 3].toMetadata, [4, 1].toMetadata, 1)
    check offset.cumsum(1).cpu == [[1'f32, 3, 6], [5'f32, 11, 18]].toTensor
    let transposed = owner.transpose
    check transposed.cumsum(1).cpu == owner.cpu.transpose.cumsum(1)

    let packed = [1'f32, 2, 3, 4].toTensor.metal
    let reversed = packed.metalView([4].toMetadata, [-1].toMetadata, 3)
    check reversed.cumsum().cpu == [4'f32, 7, 9, 10].toTensor
    let broadcast = [1'f32, 2, 3].toTensor.metal.broadcast(4, 3)
    check broadcast.cumsum(0).cpu == [[1'f32, 2, 3], [2'f32, 4, 6],
      [3'f32, 6, 9], [4'f32, 8, 12]].toTensor

  test "Reverse inclusive scan is the seeded cumsum adjoint":
    let seed = [2'f32, -1, 3].toTensor.metal
    check seed.cumsumGradient(0).cpu == [4'f32, 2, 3].toTensor
    let matrix = [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor.metal
    check matrix.cumsumGradient(-1).cpu == [[6'f32, 5, 3], [15'f32, 11, 6]].toTensor

  test "Scalar bits, signed-zero vector semantics, and nonfinite values":
    let scalarInput = newTensorWith[float32]([], -0'f32).metal
    let scalarOutput = scalarInput.cumsum(-1)
    check scalarOutput.storage != scalarInput.storage
    let scalar = scalarOutput.cpu
    check cast[ptr uint32](scalar.get_offset_ptr)[] == 0x80000000'u32

    check [1e8'f32, 1, -1e8].toTensor.metal.cumsum.cpu ==
      [1e8'f32, 1e8'f32, 0'f32].toTensor

    let zeros = [-0'f32, -0'f32, 0'f32].toTensor.metal.cumsum.cpu
    let zeroBits = cast[ptr UncheckedArray[uint32]](zeros.get_offset_ptr)
    for index in 0 ..< zeros.size:
      check zeroBits[index] == 0'u32

    let nonfinite = [Inf.float32, -Inf.float32, 2'f32].toTensor.metal.cumsum.cpu
    check nonfinite[0].classify == fcInf
    check nonfinite[1].classify == fcNan
    check nonfinite[2].classify == fcNan

  test "Empty outputs and release-safe axis errors":
    let empty = newMetalTensorUninit[float32](2, 0, 3)
    let output = empty.cumsum(1)
    check output.shape == [2, 0, 3].toMetadata
    check output.size == 0
    check output.storage != empty.storage
    let input = [1'f32, 2, 3].toTensor.metal
    expect MetalError: discard input.cumsum(-2)
    expect MetalError: discard input.cumsum(1)
    let scalar = newTensorWith[float32]([], 1'f32).metal
    expect MetalError: discard scalar.cumsum(1)
