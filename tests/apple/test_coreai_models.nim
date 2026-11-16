# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises authentic source/AOT models, strict descriptors and native job ownership.
import std/[algorithm, json, options, unittest]
import ../../src/arraymancer
import ./coreai_test_support

proc detachedIdentityJob(): CoreAIJob =
  let model = openCoreAIModel(coreAISourceAsset)
  let function = model.loadFunction("identity").get
  let array = toCoreAI([1.25'f32, -2].toTensor)
  function.submitInference([CoreAIBinding(name: "x", array: array)])

suite "Core AI authentic model loading":
  test "Manifest binds actual source files and pinned converter versions":
    let actual = coreAIManifestObservation()
    check actual["matches_manifest"].getBool

  test "Source and SDK-matching AOT assets expose named identity inference":
    for path in coreAIModelPaths():
      let model = openCoreAIModel(path)
      var names = model.functionNames
      names.sort()
      check names == @["identity", "identity_matrix"]
      check model.loadFunction("absent_function").isNone
      let function = model.loadFunction("identity").get
      let desc = function.descriptor
      check desc.name == "identity"
      check desc.inputs.len == 1 and desc.outputs.len == 1 and desc.states.len == 0
      check desc.inputs[0].name == "x" and desc.outputs[0].name == "y"
      check desc.inputs[0].dtype == "float32" and desc.outputs[0].dtype == "float32"
      check desc.inputs[0].shape == [2].toMetadata
      check not desc.inputs[0].dynamicShape
      let outputs = function.infer([CoreAIBinding(name: "x", array: toCoreAI([1.25'f32, -2].toTensor))])
      check outputs.len == 1 and outputs[0].name == "y"
      check outputs[0].array.cpu(float32) == [1.25'f32, -2].toTensor

  test "Dynamic descriptors remain unresolved until a concrete input exists":
    let model = openCoreAIModel(coreAISourceAsset)
    let function = model.loadFunction("identity_matrix").get
    let desc = function.descriptor
    check desc.inputs[0].shape == [-1, 3].toMetadata
    check desc.outputs[0].shape == [-1, 3].toMetadata
    check desc.inputs[0].dynamicShape and desc.outputs[0].dynamicShape
    check desc.inputs[0].minimumBytes == 0
    let input = [[1'f32, 2, 3], [4'f32, 5, 6]].toTensor
    let outputs = function.infer([CoreAIBinding(name: "x", array: toCoreAI(input))])
    check outputs[0].array.cpu(float32) == input

  test "Wrong names, dtype, static shape and rank fail without conversion":
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity").get
    let valid = CoreAIBinding(name: "x", array: toCoreAI([1'f32, 2].toTensor))
    expect CoreAIDescriptorError:
      discard function.infer([CoreAIBinding(name: "x", array: toCoreAI([1'i32, 2].toTensor))])
    expect CoreAIDescriptorError:
      discard function.infer([CoreAIBinding(name: "x", array: toCoreAI([1'f32, 2, 3].toTensor))])
    expect CoreAIDescriptorError:
      discard function.infer([CoreAIBinding(name: "x", array: toCoreAI([[1'f32, 2]].toTensor))])
    expect CoreAIDescriptorError:
      discard function.infer([CoreAIBinding(name: "wrong", array: valid.array)])
    expect CoreAIDescriptorError:
      discard function.infer(newSeq[CoreAIBinding]())
    expect CoreAIDescriptorError:
      discard function.infer([valid, valid])
    expect CoreAIDescriptorError:
      discard valid.array.cpu(int32)
    expect CoreAIError:
      discard openCoreAIModel("tests/apple/models/absent.aimodel")

  test "Imports snapshot logical CPU views once and native jobs retain dependencies":
    var storage = [99'f32, 1.25, 99, -2, 99].toTensor
    var view = storage
    view.offset = 1
    view.shape = [2].toMetadata
    view.strides = [2].toMetadata
    let array = toCoreAI(view)
    check array.descriptor.hostCopyBytes == 8'u64
    storage[1] = 100'f32
    storage[3] = 200'f32
    let function = openCoreAIModel(coreAISourceAsset).loadFunction("identity").get
    let outputs = function.infer([CoreAIBinding(name: "x", array: array)])
    check outputs[0].array.cpu(float32) == [1.25'f32, -2].toTensor
    check array.descriptor.hostCopyBytes == 8'u64
    let job = detachedIdentityJob()
    try: discard job.wait(timeoutMs = 0)
    except CoreAITimeoutError: discard
    GC_fullCollect()
    let retained = job.wait()
    check retained[0].array.cpu(float32) == [1.25'f32, -2].toTensor
    check retained[0].array.descriptor.hostCopyBytes == 8'u64

  test "Int32 bits and negative CPU strides survive exact typed host transfers":
    let integers = [low(int32), -7'i32, 0'i32, high(int32)].toTensor
    let integerArray = toCoreAI(integers)
    check integerArray.descriptor.dtype == "int32"
    check integerArray.cpu(int32) == integers
    let storage = [99'f32, 1.25, 99, -2, 99].toTensor
    var reverse = storage
    reverse.offset = 3
    reverse.shape = [2].toMetadata
    reverse.strides = [-2].toMetadata
    check toCoreAI(reverse).cpu(float32) == [-2'f32, 1.25].toTensor
