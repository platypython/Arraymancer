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
  test "Batched products preserve broadcasts, signed strides and vector axes":
    let left = [[[1'f32, 2], [3'f32, 4]],
                [[5'f32, 6], [7'f32, 8]]].toTensor.metal
    let right = [[[1'f32, 0], [0'f32, 2]]].toTensor.metal
    check (left * right).cpu == [[[1'f32, 4], [3'f32, 8]],
                                [[5'f32, 12], [7'f32, 16]]].toTensor
    let reversed = left.metalView([2, 2, 2].toMetadata, [4, 2, -1].toMetadata, 1)
    check (reversed * right).cpu == [[[2'f32, 2], [4'f32, 6]],
                                    [[6'f32, 10], [8'f32, 14]]].toTensor
    let highRank = newTensorWith[float32]([2, 1, 2, 2], 1'f32).metal
    let batches = newTensorWith[float32]([3, 2, 2], 2'f32).metal
    check (highRank * batches).cpu == newTensorWith[float32]([2, 3, 2, 2], 4'f32)
    let matrix = [0'f32, 1, 2, 0, 3, 4].toTensor.metal.metalView(
      [2, 2].toMetadata, [3, 1].toMetadata, 1)
    let vector = [5'f32, 6].toTensor.metal
    check (matrix * vector).cpu == [17'f32, 39].toTensor
    check (vector * matrix).cpu == [23'f32, 34].toTensor
    check (matrix.transpose * vector).cpu == [23'f32, 34].toTensor
    check (vector * matrix.transpose).cpu == [17'f32, 39].toTensor
    let reversedVector = vector.metalView([2].toMetadata, [-1].toMetadata, 1)
    check (matrix * reversedVector).cpu == [16'f32, 38].toTensor
    check (reversedVector * matrix).cpu == [21'f32, 32].toTensor
    check (reversedVector * reversedVector).cpu == newTensorWith[float32]([], 61'f32)
  test "Row-vector products cover split sums, column tails and signed batches":
    var source = newTensorUninit[float32](2, 9, 35)
    for batch in 0 ..< 2:
      for inner in 0 ..< 9:
        for column in 0 ..< 35:
          source[batch, inner, column] = float32(100 * batch + 3 * inner + column)
    let right = source.metal.metalView(
      [2, 9, 33].toMetadata, [315, 35, -1].toMetadata, 33)
    let left = [1'f32, 2, 3, 4, 5, 6, 7, 8, 9].toTensor.metal
    var expected = zeros[float32](2, 33)
    for batch in 0 ..< 2:
      for column in 0 ..< 33:
        for inner in 0 ..< 9:
          expected[batch, column] += float32(inner + 1) *
            float32(100 * batch + 3 * inner + 33 - column)
    let actual = (left * right).cpu
    check actual.shape == [2, 33].toMetadata
    check actual == expected
    check (newMetalTensorUninit[float32](0) *
      newMetalTensorUninit[float32](2, 0, 33)).cpu == zeros[float32](2, 33)
  test "Packed row-vector products preserve offset owners and partial SIMD grids":
    var matrixData = newTensorUninit[float32](9 * 36 + 8)
    for index in 0 ..< matrixData.size: matrixData[index] = float32(index mod 17 - 8)
    let matrix = matrixData.metal.metalView([9, 36].toMetadata, [36, 1].toMetadata, 4)
    let vectorData = [0'f32, 1, 2, 3, 4, 5, 6, 7, 8, 9].toTensor
    let vector = vectorData.metal.metalView([1, 9].toMetadata, [9, 1].toMetadata, 1)
    let outputOwner = newTensorWith[float32]([44], -99'f32).metal
    let output = outputOwner.metalView([1, 36].toMetadata, [36, 1].toMetadata, 4)
    var expected = newTensorWith[float32]([44], -99'f32)
    for column in 0 ..< 36:
      expected[4 + column] = 0'f32
      for inner in 0 ..< 9:
        expected[4 + column] += vectorData[1 + inner] * matrixData[4 + inner * 36 + column]
    dispatchMetalMath(vector, matrix, output, 20)
    check outputOwner.cpu == expected
    check (newMetalTensorUninit[float32](0) *
      newMetalTensorUninit[float32](0, 36)).cpu == zeros[float32](36)
  test "Queued row products keep task addresses and dependent contractions distinct":
    let vectors = [newTensorWith[float32]([5], 1'f32).metal,
                   newTensorWith[float32]([9], 2'f32).metal]
    let matrices = [newTensorWith[float32]([5, 132], 1'f32).metal,
                    newTensorWith[float32]([9, 132], 1'f32).metal]
    let consumer = newTensorWith[float32]([132, 132], 1'f32).metal
    var outputs: seq[MetalTensor[float32]]
    var dependent: MetalTensor[float32]
    for index in 0 ..< 40:
      outputs.add(vectors[index mod 2] * matrices[index mod 2])
      if index == 7:
        dependent = outputs[^1] * consumer
    for index, output in outputs:
      let expected = if index mod 2 == 0: 5'f32 else: 18'f32
      check output.cpu == newTensorWith[float32]([132], expected)
    check dependent.cpu == newTensorWith[float32]([132], 2376'f32)
  test "Packed dot products preserve offset views and vectorized tails":
    var leftData = newTensorUninit[float32](1040)
    var rightData = newTensorUninit[float32](1040)
    for index in 0 ..< 1040:
      leftData[index] = float32(index mod 13 - 6)
      rightData[index] = float32(index mod 5 - 2)
    let leftOwner = leftData.metal
    let rightOwner = rightData.metal
    for offset in [1, 4]:
      let left = leftOwner.metalView([1031].toMetadata, [1].toMetadata, offset)
      let right = rightOwner.metalView([1031].toMetadata, [1].toMetadata, offset)
      var expected = 0'f32
      for index in 0 ..< 1031:
        expected += leftData[offset + index] * rightData[offset + index]
      check (left * right).cpu == newTensorWith[float32]([], expected)
    let empty = newMetalTensorUninit[float32](0)
    check (empty * empty).cpu == newTensorWith[float32]([], 0'f32)
    let batches = leftOwner.metalView([2, 1, 7].toMetadata, [10, 0, -1].toMetadata, 8)
    let shared = rightOwner.metalView([1, 7, 1].toMetadata, [0, 1, 0].toMetadata, 4)
    var batchedExpected = zeros[float32](2, 1, 1)
    for batch in 0 ..< 2:
      for index in 0 ..< 7:
        batchedExpected[batch, 0, 0] += leftData[8 + batch * 10 - index] * rightData[4 + index]
    check (batches * shared).cpu == batchedExpected
    let left = leftOwner.metalView([1, 7].toMetadata, [0, 1].toMetadata, 4)
    let right = rightOwner.metalView([7, 1].toMetadata, [-1, 0].toMetadata, 10)
    let outputOwner = newTensorWith[float32]([3], -99'f32).metal
    let output = outputOwner.metalView([1, 1].toMetadata, [0, 0].toMetadata, 1)
    var expected = 0'f32
    for index in 0 ..< 7: expected += leftData[4 + index] * rightData[10 - index]
    dispatchMetalMath(left, right, output, 20)
    check outputOwner.cpu == [-99'f32, expected, -99'f32].toTensor
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
    for count in [65536, 65537]:
      let threshold = newTensorWith[float32]([count], 1'f32).metal
      check threshold.sum.cpu == newTensorWith[float32]([], float32(count))
      check threshold.mean.cpu == newTensorWith[float32]([], 1'f32)
      let reverse = threshold.metalView([count].toMetadata, [-1].toMetadata, count - 1)
      check reverse.sum.cpu == newTensorWith[float32]([], float32(count))
    let empty = newMetalTensorUninit[float32](2, 0, 3)
    check empty.sum.cpu == newTensorWith[float32]([], 0'f32)
    let emptyMean = empty.mean.cpu
    check cast[ptr float32](emptyMean.get_offset_ptr)[].classify == fcNan
    check (empty + 1'f32).size == 0
    check empty.sum(axis = 1).cpu == zeros[float32](2, 1, 3)
    let scalar = newTensorWith[float32]([], 4'f32).metal
    check scalar.mean.cpu == newTensorWith[float32]([], 4'f32)
  test "Axis reductions preserve signed layouts, scalar axes and empty identities":
    let owner = [99'f32, 1, 2, 3, 4, 5, 6, 99].toTensor.metal
    let x = owner.metalView([2, 3].toMetadata, [3, -1].toMetadata, 3)
    check x.sum(-1).cpu == [[6'f32], [15'f32]].toTensor
    check x.mean(-2).cpu == [[4.5'f32, 3.5, 2.5]].toTensor
    check x.prod(-1).cpu == [[6'f32], [120'f32]].toTensor
    check x.transpose.prod(-1).cpu == [[18'f32], [10'f32], [4'f32]].toTensor
    check x.prod.cpu == newTensorWith[float32]([], 720'f32)
    check owner.cpu == [99'f32, 1, 2, 3, 4, 5, 6, 99].toTensor
    let broadcast = [2'f32, 3].toTensor.metal.metalView(
      [2, 3].toMetadata, [1, 0].toMetadata, 0)
    check broadcast.prod(-1).cpu == [[8'f32], [27'f32]].toTensor
    let empty = newMetalTensorUninit[float32](2, 0, 3)
    check empty.prod.cpu == newTensorWith[float32]([], 1'f32)
    check empty.prod(-2).cpu == ones[float32](2, 1, 3)
    let emptyMean = empty.mean(-2).cpu
    for value in emptyMean: check value.classify == fcNan
    for output in [empty.sum(-1), empty.mean(-1), empty.prod(-1)]:
      check output.shape == [2, 0, 1].toMetadata
      check output.size == 0
    let scalar = newTensorWith[float32]([], -4'f32).metal
    for axis in [-1, 0]:
      for output in [scalar.sum(axis), scalar.mean(axis), scalar.prod(axis)]:
        check output.cpu == newTensorWith[float32]([], -4'f32)
        check output.storage != scalar.storage
    for axis in [-3, 2]:
      expect MetalError: discard x.mean(axis)
      expect MetalError: discard x.prod(axis)
      expect MetalError: discard x.sum(axis)
    expect MetalError: discard scalar.prod(1)
  test "Product groups preserve tails, zeros and nonfinite inputs":
    for count in [255, 256, 257, 65536, 65537]:
      var data = ones[float32](count + 2)
      data[1] = -2'f32
      data[count] = 3'f32
      let input = data.metal.metalView([count].toMetadata, [-1].toMetadata, count)
      check input.prod.cpu == newTensorWith[float32]([], -6'f32)
    let zero = [2'f32, -0'f32].toTensor.metal.prod.cpu
    check cast[ptr float32](zero.get_offset_ptr)[].signbit
    for data in [[NaN.float32, 1'f32], [Inf.float32, 0'f32]]:
      let actual = data.toTensor.metal.prod.cpu
      check cast[ptr float32](actual.get_offset_ptr)[].classify == fcNan
    var wideData = ones[float32](2, 5001)
    wideData[0, 5000] = 2'f32
    wideData[1, 5000] = 3'f32
    let wide = wideData.metal
    check wide.prod(-1).cpu == [[2'f32], [3'f32]].toTensor
    check wide.mean(-1).cpu.mean_relative_error(wideData.mean(1)) < 2e-6
    var tallData = ones[float32](513, 3)
    tallData[512, 0] = 2'f32
    tallData[512, 1] = 3'f32
    tallData[512, 2] = 4'f32
    let tall = tallData.metal
    check tall.prod(0).cpu == [[2'f32, 3, 4]].toTensor
    check tall.mean(0).cpu.mean_relative_error(tallData.mean(0)) < 2e-6
  test "Reduction packets preserve mixed shapes, guards and dependent reads":
    for operation in [22'i32, 26, 27]:
      var inputs: seq[MetalTensor[float32]]
      var owners, outputs: seq[MetalTensor[float32]]
      for index in 0 ..< 40:
        let rows = 3 + index mod 2
        inputs.add newTensorWith[float32]([rows, 7], float32(1 + index mod 2)).metal
        owners.add newTensorWith[float32]([rows + 2], -99'f32).metal
        outputs.add owners[^1].metalView([rows, 1].toMetadata, [1, 1].toMetadata, 1)
      var dependent: MetalTensor[float32]
      for index in 0 ..< 40:
        dispatchMetalMath(inputs[index], default(MetalTensor[float32]), outputs[index],
          operation, axis = 1)
        if index == 7: dependent = outputs[index] + 1'f32
      inputs.setLen(0)
      for index in 0 ..< 40:
        let item = float32(1 + index mod 2)
        let expected = if operation == 22: item * 7'f32
                       elif operation == 26: pow(item, 7'f32)
                       else: item
        let actual = owners[index].cpu
        check actual[0] == -99'f32
        check actual[actual.size - 1] == -99'f32
        check outputs[index].cpu == newTensorWith[float32]([3 + index mod 2, 1], expected)
      let expected = if operation == 22: 15'f32
                     elif operation == 26: 129'f32
                     else: 3'f32
      check dependent.cpu == newTensorWith[float32]([4, 1], expected)
  test "Queued snapshots survive later input mutation and dependent writes":
    var state = ones[float32](257).metal
    let before = state + 2'f32
    state.fill(7'f32)
    let after = state * 3'f32
    state.fill(-4'f32)
    state += after
    check before.cpu == newTensorWith[float32]([257], 3'f32)
    check after.cpu == newTensorWith[float32]([257], 21'f32)
    check state.cpu == newTensorWith[float32]([257], 17'f32)
  test "Reused destinations retain intermediate reads across batch boundaries":
    var destination = ones[float32](257).metal + 1'f32
    check destination.cpu == newTensorWith[float32]([257], 2'f32)
    var snapshots: seq[MetalTensor[float32]]
    for index in 0 ..< 80:
      destination.fill(float32(index))
      snapshots.add(destination + 1'f32)
    for index, snapshot in snapshots:
      check snapshot.cpu == newTensorWith[float32]([257], float32(index + 1))
  test "Packed unary outputs bound aligned tails and unaligned offset views":
    var source = newTensorUninit[float32](12)
    for index in 0 ..< 12: source[index] = float32(index + 1) * 0.125'f32
    let inputOwner = source.metal
    for offset in [1, 4]:
      let input = inputOwner.metalView([7].toMetadata, [1].toMetadata, offset)
      for operation in [10'i32, 11, 12]:
        let outputOwner = newTensorWith[float32]([12], -99'f32).metal
        let output = outputOwner.metalView([7].toMetadata, [1].toMetadata, offset)
        dispatchMetalMath(input, default(MetalTensor[float32]), output,
          operation, scalarInput = true)
        var expected = newTensorUninit[float32](7)
        for index in 0 ..< 7: expected[index] = source[offset + index]
        case operation
        of 10: expected = expected.exp
        of 11: expected = expected.ln
        else: expected = expected.sqrt
        check output.cpu.mean_relative_error(expected) < 2e-6
        let entireOutput = outputOwner.cpu
        for index in 0 ..< 12:
          if index < offset or index >= offset + 7:
            check entireOutput[index] == -99'f32
  test "Queued unary packets preserve mixed grid widths and dependent results":
    let inputOwner = newTensorWith[float32]([1036], 4'f32).metal
    let widths = [3, 3, 5, 5, 1020, 1020, 1024, 1024, 1031, 1031]
    var outputs: seq[MetalTensor[float32]]
    var dependent: MetalTensor[float32]
    # Upload sentinels before queuing so missing tail writes cannot read old results.
    for index in 0 ..< 40:
      outputs.add newTensorWith[float32]([widths[index mod widths.len]], -99'f32).metal
    for index in 0 ..< outputs.len:
      let count = widths[index mod widths.len]
      let input = inputOwner.metalView([count].toMetadata, [1].toMetadata, 4)
      dispatchMetalMath(input, default(MetalTensor[float32]), outputs[index],
        12, scalarInput = true)
      if index == 7: dependent = outputs[index].sqrt
    for index in 0 ..< outputs.len:
      let count = widths[index mod widths.len]
      check outputs[index].shape == [count].toMetadata
      check outputs[index].cpu == newTensorWith[float32]([count], 2'f32)
    check dependent.cpu.mean_relative_error(
      newTensorWith[float32]([1024], 2'f32).sqrt) < 2e-6
  test "Unary packets preserve repeated destinations and intermediate reads":
    let inputs = [newTensorWith[float32]([1031], 4'f32).metal,
                  newTensorWith[float32]([1031], 9'f32).metal]
    let owner = newTensorWith[float32]([1036], -99'f32).metal
    let destination = owner.metalView([1031].toMetadata, [1].toMetadata, 4)
    var snapshots: seq[MetalTensor[float32]]
    for index in 0 ..< 40:
      dispatchMetalMath(inputs[index mod 2], default(MetalTensor[float32]),
        destination, 12, scalarInput = true)
      if index in [7, 16, 31]: snapshots.add destination.sqrt
    let actual = owner.cpu
    for index in 0 ..< 4: check actual[index] == -99'f32
    for index in 4 ..< 1035: check actual[index] == 3'f32
    check actual[1035] == -99'f32
    for index, value in [3'f32, 2, 3]:
      check snapshots[index].cpu.mean_relative_error(
        newTensorWith[float32]([1031], value).sqrt) < 2e-6
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
