# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises runtime shaders, offline libraries, byte offsets and queued access leases.
import std/[unittest, os, tempfiles]
import ../../src/arraymancer

const customSource = staticRead(currentSourcePath().parentDir / "kernels/double.metal")

suite "Metal custom shader execution":
  test "Runtime compilation and independent metallib loading both double values":
    let device = metalDevice()
    for library in [compileMetalShader(device, customSource),
        loadMetalLibrary(device, arraymancerAppleLibDir / "arraymancer_test.metallib")]:
      let kernel = metalKernel(library, "vector_double")
      let input = [1'f32, 2, 3, 4].toTensor.metal(device)
      let output = newMetalTensorUninit[float32]([4].toMetadata, device)
      dispatchMetalKernel(kernel, [metalBinding(input, mbaRead), metalBinding(output, mbaWrite)],
        [4, 1, 1], [4, 1, 1])
      check input.storage != output.storage
      check output.cpu == [2'f32, 4, 6, 8].toTensor
      check input.cpu == [1'f32, 2, 3, 4].toTensor
      var evidence: AMEvidence
      checkMetal metalApi().evidence(output.storage.owner.handle, addr evidence)
      check evidence.kind == 2 and evidence.completed == 1 and evidence.kernelBits == 300
      check evidence.cpuFallbackCount == 0

  test "Binding offsets preserve guards without packing or copying the view":
    let device = metalDevice()
    let kernel = metalKernel(compileMetalShader(device, customSource), "vector_double")
    let input = [99'f32, 1, 2, 3, 4, 99].toTensor.metal(device)
    let output = [77'f32, 0, 0, 0, 0, 77].toTensor.metal(device)
    let inputView = input.metalView([4].toMetadata, [1].toMetadata, 1)
    let outputView = output.metalView([4].toMetadata, [1].toMetadata, 1)
    dispatchMetalKernel(kernel, [metalBinding(inputView, mbaRead), metalBinding(outputView, mbaWrite)],
      [4, 1, 1], [2, 1, 1])
    check output.cpu == [77'f32, 2, 4, 6, 8, 77].toTensor
    check input.cpu == [99'f32, 1, 2, 3, 4, 99].toTensor
    var evidence: AMEvidence
    checkMetal metalApi().evidence(output.storage.owner.handle, addr evidence)
    check evidence.materializationBytes == 0

  test "Pending dispatch owns dropped libraries, pipelines, inputs and both outputs":
    let device = metalDevice()
    let stream = newMetalStream(device)
    let gate = newMetalEvent(device)
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      metalSynchronize(stream)
    wait(stream, gate, 1)
    var doubled, tripled: MetalTensor[float32]
    block:
      let library = compileMetalShader(device, customSource)
      let kernel = metalKernel(library, "vector_split")
      let input = [1'f32, 2, 3, 4].toTensor.metal(device)
      let first = newMetalTensorUninit[float32]([4].toMetadata, device)
      let second = newMetalTensorUninit[float32]([4].toMetadata, device)
      doubled = first
      tripled = second
      dispatchMetalKernel(kernel, [metalBinding(input, mbaRead), metalBinding(first, mbaWrite),
        metalBinding(second, mbaWrite)], [4, 1, 1], [4, 1, 1], stream)
    GC_fullCollect()
    metalEmptyCache(device)
    for output in [doubled, tripled]:
      var pending: AMEvidence
      checkMetal metalApi().evidence(output.storage.owner.handle, addr pending)
      check pending.kind == 2 and pending.completed == 0 and pending.kernelBits == 300
    signalMetalEvent(gate, 1)
    released = true
    check tripled.cpu == [3'f32, 6, 9, 12].toTensor
    check doubled.cpu == [2'f32, 4, 6, 8].toTensor

  test "Same-queue mutable chaining works while cross-queue leases reject conflicts":
    let device = metalDevice()
    let kernel = metalKernel(compileMetalShader(device, customSource), "vector_inplace")
    let values = [1'f32, 2, 3, 4].toTensor.metal(device)
    let stream = newMetalStream(device)
    let other = newMetalStream(device)
    let gate = newMetalEvent(device)
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      metalSynchronize(stream)
      metalSynchronize(other)
    wait(stream, gate, 1)
    for pass in 0 ..< 2:
      dispatchMetalKernel(kernel, [metalBinding(values, mbaReadWrite)], [4, 1, 1], [4, 1, 1], stream)
    expect MetalBufferBusyError:
      dispatchMetalKernel(kernel, [metalBinding(values, mbaReadWrite)], [4, 1, 1], [4, 1, 1], other)
    check not metalStreamIsComplete(stream)
    signalMetalEvent(gate, 1)
    released = true
    metalSynchronize(stream)
    check values.cpu == [4'f32, 8, 12, 16].toTensor
    dispatchMetalKernel(kernel, [metalBinding(values, mbaReadWrite)], [4, 1, 1], [4, 1, 1], other)
    check values.cpu == [8'f32, 16, 24, 32].toTensor

  test "Compiler, library, function and binding errors remain visible":
    let device = metalDevice()
    expect MetalShaderCompileError:
      discard compileMetalShader(device, "kernel void broken( this is not Metal source")
    let (file, path) = createTempFile("arraymancer-corrupt-", ".metallib")
    file.write("not a metallib")
    file.close()
    defer: removeFile(path)
    expect MetalLibraryLoadError: discard loadMetalLibrary(device, path)
    let library = compileMetalShader(device, customSource)
    expect MetalMissingFunctionError: discard metalKernel(library, "absent_function")
    let kernel = metalKernel(library, "vector_double")
    let input = [1'f32, 2, 3, 4].toTensor.metal(device)
    let output = [9'f32, 9, 9, 9].toTensor.metal(device)
    expect MetalError:
      discard metalBinding(input.metalView([2].toMetadata, [2].toMetadata, 0), mbaRead)
    let bindings = [metalBinding(input, mbaRead), metalBinding(output, mbaWrite)]
    expect MetalError: dispatchMetalKernel(kernel, bindings, [0, 1, 1], [4, 1, 1])
    expect MetalError: dispatchMetalKernel(kernel, bindings, [4, 1, 1], [0, 1, 1])
    expect MetalError: dispatchMetalKernel(kernel, bindings, [4, 1, 1], [1048576, 1, 1])
    expect MetalBufferBusyError:
      dispatchMetalKernel(kernel, [metalBinding(input, mbaRead), metalBinding(input, mbaWrite)],
        [4, 1, 1], [4, 1, 1])
    check output.cpu == [9'f32, 9, 9, 9].toTensor
    check input.cpu == [1'f32, 2, 3, 4].toTensor
