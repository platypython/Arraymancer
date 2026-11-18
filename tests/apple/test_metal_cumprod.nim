# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises native Metal32 cumulative products and their reverse-mode adjoints.
import std/[math, unittest]
import ../../src/arraymancer

suite "Native Metal float32 cumulative products":
  test "Row tails and long scans carry exact bounded products":
    for count in [1, 31, 32, 33, 127, 128, 129]:
      let input = ones[float32](count).metal
      let forward = input.cumprod().cpu
      let prefixes = input.cumprod()
      let reverse = cumprodGradient(input, prefixes, ones[float32](count).metal, 0).cpu
      for index in 0 ..< count:
        check forward[index] == 1'f32
        check reverse[index] == float32(count - index)

    const count = 65539
    let input = ones[float32](count).metal
    let forward = input.cumprod().cpu
    let prefixes = input.cumprod()
    let reverse = cumprodGradient(input, prefixes, ones[float32](count).metal, 0).cpu
    for index in [0, 31, 32, 127, 128, 65535, 65536, count - 1]:
      check forward[index] == 1'f32
      check reverse[index] == float32(count - index)

  test "Queued packed scans retain each shape, axis and dropped input":
    var outputs: seq[MetalTensor[float32]]
    var expected: seq[Tensor[float32]]
    for action in 0 ..< 40:
      let count = [17, 31, 33, 127, 128][action mod 5]
      let axis = action mod 2
      let shape = if axis == 1: [2, count] else: [count, 2]
      var host = newTensorWith[float32](shape, 1'f32)
      for index in 0 ..< host.size:
        host.unsafe_raw_offset[index] = if (index + action) mod 3 == 0: -1'f32 else: 1'f32
      expected.add host.cumprod(axis)
      outputs.add host.metal.cumprod(axis)
    for action in 0 ..< outputs.len:
      check outputs[action].shape == expected[action].shape
      check outputs[action].cpu == expected[action]

  test "Prefetched blocks preserve sequential overflow, underflow and signed tails":
    var overflow = newTensorWith[float32]([19], 1'f32)
    overflow[7] = 1e20'f32
    overflow[8] = 1e20'f32
    overflow[9] = 1e-20'f32
    let overflowPrefixes = overflow.metal.cumprod().cpu
    for index in 0 ..< 7: check overflowPrefixes[index] == 1'f32
    check overflowPrefixes[7] == 1e20'f32
    for index in 8 ..< 19: check overflowPrefixes[index].classify == fcInf
    var underflow = newTensorWith[float32]([19], 1'f32)
    underflow[7] = 1e-30'f32
    underflow[8] = 1e-30'f32
    underflow[9] = 1e30'f32
    let underflowPrefixes = underflow.metal.cumprod().cpu
    for index in 0 ..< 7: check underflowPrefixes[index] == 1'f32
    check underflowPrefixes[7] == 1e-30'f32
    for index in 8 ..< 19: check underflowPrefixes[index] == 0'f32
    for inverse in [false, true]:
      var bounded = newTensorWith[float32]([16], 1'f32)
      let small = if inverse: 1e30'f32 else: 1e-30'f32
      let large = if inverse: 1e-30'f32 else: 1e30'f32
      bounded[0] = small
      bounded[4] = large
      bounded[5] = large
      bounded[6] = small
      let prefixes = bounded.metal.cumprod().cpu
      var carry = 1'f32
      for index in 0 ..< 16:
        carry *= bounded[index]
        check prefixes[index] == carry
    var owner = newTensor[float32](35)
    for index in 0 ..< 35:
      owner[index] = (if index mod 2 == 0: 2'f32 else: -0.5'f32)
    let reversed = owner.metal.metalView([33].toMetadata, [-1].toMetadata, 33)
    let actual = reversed.cumprod().cpu
    var carry = 1'f32
    for index in 0 ..< 33:
      carry *= owner[33 - index]
      check actual[index] == carry

  test "Column and 3D scans preserve independent axes":
    let host = [[2'f32, 3, 4], [5'f32, 2, 3], [3'f32, 4, 2]].toTensor
    let input = host.metal
    check input.cumprod(0).cpu == [[2'f32, 3, 4], [10'f32, 6, 12],
      [30'f32, 24, 24]].toTensor
    check input.cumprod(1).cpu == [[2'f32, 6, 24], [5'f32, 10, 30],
      [3'f32, 12, 24]].toTensor

    let cube = [[[2'f32, 3], [5'f32, 2]], [[3'f32, 4], [2'f32, 3]]].toTensor
    check cube.metal.cumprod(0).cpu == [[[2'f32, 3], [5'f32, 2]],
      [[6'f32, 12], [10'f32, 6]]].toTensor
    check cube.metal.cumprod(1).cpu == [[[2'f32, 3], [10'f32, 6]],
      [[3'f32, 4], [6'f32, 12]]].toTensor
    check cube.metal.cumprod(2).cpu == [[[2'f32, 6], [5'f32, 10]],
      [[3'f32, 12], [2'f32, 6]]].toTensor

  test "Offset, transposed, reversed and broadcast views scan natively":
    let owner = [[0'f32, 1, 2, 3], [4'f32, 5, 6, 7],
                 [8'f32, 9, 10, 11]].toTensor.metal
    let offset = owner.metalView([2, 3].toMetadata, [4, 1].toMetadata, 1)
    let offsetOutput = offset.cumprod(1)
    check offsetOutput.storage != offset.storage
    check offsetOutput.cpu == [[1'f32, 2, 6], [5'f32, 30, 210]].toTensor
    check owner.transpose.cumprod(1).cpu ==
      owner.cpu.transpose.cumprod(1)
    let packed = [1'f32, 2, 3, 4].toTensor.metal
    let reversed = packed.metalView([4].toMetadata, [-1].toMetadata, 3)
    check reversed.cumprod().cpu == [4'f32, 12, 24, 24].toTensor
    let broadcast = [2'f32, 3, 4].toTensor.metal.broadcast(4, 3)
    check broadcast.cumprod(0).cpu == [[2'f32, 3, 4], [4'f32, 9, 16],
      [8'f32, 27, 64], [16'f32, 81, 256]].toTensor

  test "Input, prefix and seed descriptors use independent signed offsets":
    let ownerHost = [[2'f32, 0.5, -1, -1], [-1'f32, 2, 0.5, -1],
                     [0.5'f32, -1, 2, -1]].toTensor
    var seedHost = newTensor[float32](4, 4)
    for index in 0 ..< 16:
      seedHost.unsafe_raw_offset[index] =
        float32(index + 1) * (if index mod 2 == 0: 1'f32 else: -1'f32)
    let input = ownerHost.metal.metalView([3, 3].toMetadata, [-4, -1].toMetadata, 11)
    let seed = seedHost.metal.metalView([3, 3].toMetadata, [-1, -4].toMetadata, 14)
    var logicalInput = newTensor[float32](3, 3)
    var logicalSeed = newTensor[float32](3, 3)
    for row in 0 ..< 3:
      for column in 0 ..< 3:
        logicalInput[row, column] = ownerHost.unsafe_raw_offset[11 - 4 * row - column]
        logicalSeed[row, column] = seedHost.unsafe_raw_offset[14 - row - 4 * column]
    for axis in 0 ..< 2:
      let logicalPrefixes = logicalInput.cumprod(axis)
      check input.cumprod(axis).cpu == logicalPrefixes
      var prefixHost = newTensor[float32](4, 4)
      for row in 0 ..< 3:
        for column in 0 ..< 3:
          prefixHost.unsafe_raw_offset[3 + 4 * row - column] = logicalPrefixes[row, column]
      let prefixes = prefixHost.metal.metalView(
        [3, 3].toMetadata, [4, -1].toMetadata, 3)
      var expected = newTensor[float32](3, 3)
      for row in 0 ..< 3:
        for column in 0 ..< 3:
          let position = [row, column]
          var gradient = 0'f32
          for endpoint in position[axis] ..< 3:
            var derivative = 1'f32
            for factor in 0 .. endpoint:
              if factor == position[axis]: continue
              var coordinate = position
              coordinate[axis] = factor
              derivative *= logicalInput[coordinate[0], coordinate[1]]
            var coordinate = position
            coordinate[axis] = endpoint
            gradient += derivative * logicalSeed[coordinate[0], coordinate[1]]
          expected[row, column] = gradient
      let actual = cumprodGradient(input, prefixes, seed, axis)
      check actual.shape == [3, 3].toMetadata
      check actual.cpu == expected

  test "Float32 carries match MPS overflow and underflow semantics":
    let overflowInput = [1e20'f32, 1e20'f32, 1e-20'f32].toTensor
    let overflow = overflowInput.metal.cumprod().cpu
    check overflowInput.cumprod()[2] == 1e20'f32
    check overflow[0] == 1e20'f32
    check overflow[1].classify == fcInf
    check overflow[2].classify == fcInf
    let underflowHost = [1e-30'f32, 1e-30'f32, 1e30'f32].toTensor
    let underflowInput = underflowHost.metal
    let underflow = underflowInput.cumprod().cpu
    check underflowHost.cumprod()[2] == 1e-30'f32
    check underflow == [1e-30'f32, 0'f32, 0'f32].toTensor
    let context = newContext(MetalTensor[float32])
    let x = context.variable(underflowInput, requires_grad = true)
    let output = x.cumprod()
    output.backprop([1'f32, 2, 3].toTensor.metal)
    check x.grad.cpu == [1'f32, 0, 0].toTensor

  test "Broadcast inputs and seeds produce exact axis VJPs":
    let context = newContext(MetalTensor[float32])
    let source = [2'f32, 3].toTensor.metal.reshape(1, 2).broadcast(3, 2)
    let seed = [1'f32, 2].toTensor.metal.reshape(1, 2).broadcast(3, 2)
    let x = context.variable(source, requires_grad = true)
    let output = x.cumprod(0)
    output.backprop(seed)
    check output.value.cpu == [[2'f32, 3], [4'f32, 9], [8'f32, 27]].toTensor
    check x.grad.cpu == [[7'f32, 26], [6'f32, 24], [4'f32, 18]].toTensor


  test "Zero and nonfinite masks retain cumulative-product VJP distinctions":
    let context = newContext(MetalTensor[float32])
    let x = context.variable([0'f32, Inf.float32, 0, 3].toTensor.metal,
                             requires_grad = true)
    let output = x.cumprod()
    output.backprop([1'f32, 2, 3, 4].toTensor.metal)
    let grad = x.grad.cpu
    check grad[0].classify == fcNan
    check grad[1] == 0'f32
    check grad[2] == 0'f32
    check grad[3] == 0'f32

  test "Scalar, empty and singleton scans preserve identity semantics":
    let scalar = newTensorWith[float32]([], -0'f32).metal
    let scalarOutput = scalar.cumprod(-1)
    check scalarOutput.storage != scalar.storage
    let scalarHost = scalarOutput.cpu
    check cast[ptr uint32](scalarHost.get_offset_ptr)[] == 0x80000000'u32
    let scalarContext = newContext(MetalTensor[float32])
    let scalarVariable = scalarContext.variable(
      newTensorWith[float32]([], NaN.float32).metal, requires_grad = true)
    let scalarResult = scalarVariable.cumprod(-1)
    scalarResult.backprop(newTensorWith[float32]([], 3'f32).metal)
    check scalarVariable.grad.cpu.atContiguousIndex(0) == 3'f32

    let empty = newMetalTensorUninit[float32](2, 0, 3)
    let emptyOutput = empty.cumprod(1)
    check emptyOutput.shape == [2, 0, 3].toMetadata
    check emptyOutput.size == 0
    check emptyOutput.storage != empty.storage

    let singletonContext = newContext(MetalTensor[float32])
    let singleton = singletonContext.variable([[NaN.float32], [Inf.float32]].toTensor.metal,
                                               requires_grad = true)
    let singletonOutput = singleton.cumprod(1)
    let seed = [[2'f32], [-3'f32]].toTensor.metal
    singletonOutput.backprop(seed)
    check singleton.grad.cpu == [[2'f32], [-3'f32]].toTensor

  test "Retained VJPs accumulate and no-grad disables tracing":
    let context = newContext(MetalTensor[float32])
    let x = context.variable([2'f32, -3, 4].toTensor.metal, requires_grad = true)
    let output = x.cumprod()
    output.backprop([1'f32, 2, 3].toTensor.metal, retain_graph = true)
    output.backprop([1'f32, 1, 1].toTensor.metal, retain_graph = true)
    check x.grad.cpu == [-55'f32, 38, -24].toTensor
    var untracked: Variable[MetalTensor[float32]]
    context.no_grad_mode:
      untracked = x.cumprod()
    check not untracked.requires_grad

  test "Invalid axes fail safely":
    let input = [1'f32, 2, 3].toTensor.metal
    expect MetalError: discard input.cumprod(-2)
    expect MetalError: discard input.cumprod(1)
    let scalar = newTensorWith[float32]([], 1'f32).metal
    expect MetalError: discard scalar.cumprod(1)
