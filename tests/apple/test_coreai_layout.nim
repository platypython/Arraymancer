# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises real shared-buffer SDK views, descriptor resolution and pending borrow exclusion.
import std/[options, os, unittest]
import ../../src/arraymancer
import ./coreai_test_support

const customSource = staticRead(currentSourcePath().parentDir / "kernels/double.metal")

suite "Core AI strided shared-buffer inference":
  test "SDK dynamic resolution preserves preferred layout and rejects incompatible shapes":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    let dynamic = function.descriptor.inputs[0]
    check dynamic.dynamicShape and not dynamic.resolved
    check dynamic.minimumBytes == 0
    for rows in [2, 3]:
      let resolved = resolveDescriptor(dynamic, [rows, 3].toMetadata)
      check resolved.shape == [rows, 3].toMetadata
      check resolved.resolved and not resolved.dynamicShape
      check resolved.dtype == "float32"
      check resolved.strides[0] > 0 and resolved.strides[1] > 0
      let extent = (rows - 1) * resolved.strides[0] + 2 * resolved.strides[1] + 1
      check resolved.minimumBytes >= uint64(extent * sizeof(float32))
    expect CoreAIDescriptorError:
      discard resolveDescriptor(dynamic, [2, 4].toMetadata)
    expect CoreAIDescriptorError:
      discard resolveDescriptor(dynamic, [2].toMetadata)
    expect CoreAIDescriptorError:
      discard resolveDescriptor(dynamic, [-1, 3].toMetadata)

  test "Contiguous and offset-strided identities return actual row-major allocated outputs":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    let expected = [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor
    let storage = [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor.metal
    let strided = storage.metalView([2, 3].toMetadata, [1, 2].toMetadata, 1)
    let borrowed = coreAIView(strided)
    let info = borrowed.descriptor
    check info.borrowed and info.byteOffset == 4
    check info.bufferIdentity == originalMetalBufferIdentity(storage)
    check info.strides == [1, 2].toMetadata
    check info.hostCopyBytes == 0
    for input in [toCoreAI(expected), borrowed]:
      let outputs = function.infer([CoreAIBinding(name: "x", array: input)],
        states = [], outputViews = [])
      check outputs[0].array.cpu(float32) == expected
      let output = outputs[0].array.descriptor
      check output.shape == [2, 3].toMetadata
      check output.strides == [3, 1].toMetadata
      check not output.borrowed
    check borrowed.descriptor.hostCopyBytes == 0
    check storage.cpu == [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor
    let another = [[1'f32, 2, 3], [4'f32, 5, 6], [7'f32, 8, 9]].toTensor
    let output = function.infer([CoreAIBinding(name: "x", array: toCoreAI(another))],
      states = [], outputViews = [])
    check output[0].array.cpu(float32) == another
    check output[0].array.descriptor.strides == [3, 1].toMetadata

  test "Explicit host reads of a borrowed view record the requested copy":
    let storage = [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor.metal
    let borrowed = coreAIView(storage.metalView([2, 3].toMetadata, [1, 2].toMetadata, 1))
    check borrowed.descriptor.hostCopyBytes == 0
    check borrowed.cpu(float32) == [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor
    check borrowed.descriptor.hostCopyBytes == 24

  test "Invalid bounds, overlapping strides and unsupported dtype fail before unsafe views":
    let storage = [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor.metal
    let valid = storage.metalView([2, 3].toMetadata, [1, 2].toMetadata, 1)
    var invalid = valid
    invalid.offset = 7
    expect CoreAIDescriptorError: discard coreAIView(invalid)
    invalid = valid
    invalid.strides = [0, 1].toMetadata
    expect CoreAIDescriptorError: discard coreAIView(invalid)
    invalid = valid
    invalid.strides = [1, 1].toMetadata
    expect CoreAIDescriptorError: discard coreAIView(invalid)
    invalid = valid
    invalid.offset = 2
    invalid.strides = [-1, 2].toMetadata
    expect CoreAIDescriptorError: discard coreAIView(invalid)
    expect CoreAIDescriptorError:
      discard coreAIView([1'i64, 2].toTensor.metal)
    check storage.cpu == [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor

  test "Pending read borrows reject scalar, math, host-upload and custom writes without blocking":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    let kernel = metalKernel(compileMetalShader(metalDevice(), customSource), "vector_inplace")
    let source = [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor.metal
    let onesCPU = newTensorWith[float32]([8], 1'f32)
    let ones = onesCPU.metal
    let producer = newMetalStream()
    let unrelated = newMetalStream()
    let gate = newMetalEvent()
    let otherGate = newMetalEvent()
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      signalMetalEvent(otherGate, 1)
      metalSynchronize(producer)
      metalSynchronize(unrelated)
    wait(producer, gate, 1)
    wait(unrelated, otherGate, 1)
    var pending: MetalTensor[float32]
    withMetalStream(producer): pending = source + 0'f32
    let view = coreAIView(pending.metalView([2, 3].toMetadata, [1, 2].toMetadata, 1))
    let job = function.submitInference([CoreAIBinding(name: "x", array: view)],
      states = [], outputViews = [])
    expect MetalBufferBusyError: pending[1] = 100'f32
    expect MetalBufferBusyError: pending += ones
    expect MetalBufferBusyError:
      checkMetal metalApi().transfer(pending.storage.owner.handle, pending.descriptor,
        cast[pointer](onesCPU.get_data_ptr), 1)
    expect MetalBufferBusyError:
      dispatchMetalKernel(kernel, [metalBinding(pending, mbaReadWrite)], [8, 1, 1], [8, 1, 1])
    expect CoreAITimeoutError: discard job.wait(timeoutMs = 0)
    signalMetalEvent(gate, 1)
    released = true
    let outputs = job.wait()
    check outputs[0].array.cpu(float32) == [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor
    check pending.cpu == [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor
    check not metalStreamIsComplete(unrelated)
    let evidence = job.coreAIJobEvidence
    check evidence.borrowedInputCount == 1 and evidence.producerCount == 1
    check evidence.completed
    pending[1] = 10'f32
    check pending.cpu == [99'f32, 10, 4, 2, 5, 3, 6, 99].toTensor
    check outputs[0].array.cpu(float32) == [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor

  test "Borrowed native owner survives dropped tensors, collection and cache eviction":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity_matrix").get
    var borrowed: CoreAIArray
    block:
      let storage = [99'f32, 1, 4, 2, 5, 3, 6, 99].toTensor.metal
      borrowed = coreAIView(storage.metalView([2, 3].toMetadata, [1, 2].toMetadata, 1))
    GC_fullCollect()
    metalEmptyCache()
    let pressure = newTensorWith[float32]([8], -9'f32).metal
    let outputs = function.infer([CoreAIBinding(name: "x", array: borrowed)],
      states = [], outputViews = [])
    check outputs[0].array.cpu(float32) == [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor
    check pressure.cpu == newTensorWith[float32]([8], -9'f32)
    check borrowed.descriptor.hostCopyBytes == 0
