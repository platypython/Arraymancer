# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises arithmetic, layouts, broadcast errors and deterministic GPU reductions.
import std/[unittest, math]
import ../../src/arraymancer

suite "Native Metal float32 math":
  test "Unary kernels preserve negative strides, scalars and empty shapes":
    let packed = [0.25'f32, 1, 4, 9].toTensor.metal
    let reverse = packed.metalView([4].toMetadata, [-1].toMetadata, 3)
    let expected = [9'f32, 4, 1, 0.25].toTensor
    check reverse.sqrt.cpu == [3'f32, 2, 1, 0.5].toTensor
    check reverse.exp.cpu.mean_relative_error(expected.exp) < 2e-6
    check reverse.ln.cpu.mean_relative_error(expected.ln) < 2e-6
    let scalar = newTensorWith[float32]([], 4'f32).metal
    check scalar.sqrt.cpu == newTensorWith[float32]([], 2'f32)
    let empty = newMetalTensorUninit[float32](2, 0, 3)
    for output in [empty.exp, empty.ln, empty.sqrt]:
      check output.shape == [2, 0, 3].toMetadata
      check output.size == 0
    expect MetalDtypeError: discard [1'i32].toTensor.metal.exp

  test "Trailing-axis broadcasting and matmul preserve inputs":
    let a = [[0'f32], [10'f32]].toTensor.metal
    let b = [[0'f32, 1, 2]].toTensor.metal
    check (a +. b).cpu == [[0'f32, 1, 2], [10'f32, 11, 12]].toTensor
    let matrix = [[0'f32, 1, 2], [3'f32, 4, 5]].toTensor.metal
    let vector = [10'f32, 20, 30].toTensor.metal
    check (matrix +. vector).cpu == [[10'f32, 21, 32], [13'f32, 24, 35]].toTensor
    check (a.cpu == [[0'f32], [10'f32]].toTensor)
    let left = [[1'f32, 2], [3'f32, 4]].toTensor.metal
    let right = [[5'f32, 6], [7'f32, 8]].toTensor.metal
    check (left * right).cpu == [[19'f32, 22], [43'f32, 50]].toTensor
    check left.cpu == [[1'f32, 2], [3'f32, 4]].toTensor
    expect MetalError: discard matrix +. [1'f32, 2].toTensor.metal
    check (vector * vector).cpu == newTensorWith[float32]([], 1400'f32)
  test "Signed-stride arithmetic and tile edges stay native":
    let packed = [1'f32, 2, 3, 4].toTensor.metal
    let reverse = packed.metalView([4].toMetadata, [-1].toMetadata, 3)
    check (reverse * 2'f32).cpu == [8'f32, 6, 4, 2].toTensor
    let transposed = [[1'f32, 2], [3'f32, 4]].toTensor.metal.transpose
    check (transposed +. transposed).cpu == [[2'f32, 6], [4'f32, 8]].toTensor
    let left = newTensorWith[float32]([17, 19], 1'f32).metal
    let right = newTensorWith[float32]([19, 18], 1'f32).metal
    check (left * right).cpu == newTensorWith[float32]([17, 18], 19'f32)
    check (newMetalTensorUninit[float32](2, 0) * newMetalTensorUninit[float32](0, 3)).cpu == zeros[float32](2, 3)
  test "Device scalars empties and multi-group sums":
    let x = [[1'f32, 2], [3'f32, 4]].toTensor.metal
    let total = x.sum
    check total.rank == 0
    check total.storage.device.registryId == x.storage.device.registryId
    check total.cpu == newTensorWith[float32]([], 10'f32)
    check x.mean.cpu == newTensorWith[float32]([], 2.5'f32)
    check x.sum(axis = 0).cpu == [[4'f32, 6]].toTensor
    check x.sum(axis = 1).cpu == [[3'f32], [7'f32]].toTensor
    let many = newTensorWith[float32]([1025], 1'f32).metal
    check many.sum.cpu == newTensorWith[float32]([], 1025'f32)
    let empty = newMetalTensorUninit[float32](2, 0, 3)
    check empty.sum.cpu == newTensorWith[float32]([], 0'f32)
    let emptyMean = empty.mean.cpu
    check cast[ptr float32](emptyMean.get_offset_ptr)[].classify == fcNan
    check (empty + 1'f32).size == 0
    check empty.sum(axis = 1).cpu == zeros[float32](2, 1, 3)
    let scalar = newTensorWith[float32]([], 4'f32).metal
    check scalar.mean.cpu == newTensorWith[float32]([], 4'f32)
  test "In-place alias hazards materialize reads and preserve storage":
    var x = [[1'f32, 2], [3'f32, 4]].toTensor.metal
    let owner = x.storage
    x += x.transpose
    check x.storage == owner
    check x.cpu == [[2'f32, 5], [5'f32, 8]].toTensor
    let grad = [[1'f32, 2], [3'f32, 4]].toTensor.metal
    x.scaledSubtract(grad, 0.5'f32)
    check x.cpu == [[1.5'f32, 4], [3.5'f32, 6]].toTensor
    check grad.cpu == [[1'f32, 2], [3'f32, 4]].toTensor
    x.fill(0'f32)
    check x.cpu == zeros[float32](2, 2)
    check grad.ones_like.cpu == ones[float32](2, 2)
  test "Scalar and elementwise operations reject nonarithmetic dtypes":
    let x = [2'f32, 4].toTensor.metal
    let y = [1'f32, 2].toTensor.metal
    check (x -. y).cpu == [1'f32, 2].toTensor
    check (x *. y).cpu == [2'f32, 8].toTensor
    check (x /. y).cpu == [2'f32, 2].toTensor
    check (8'f32 /. x).cpu == [4'f32, 2].toTensor
    check (8'f32 -. x).cpu == [6'f32, 4].toTensor
    check (-x).cpu == [-2'f32, -4].toTensor
    expect MetalDtypeError: discard [1'i32, 2].toTensor.metal + 1'f32
