# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes direct shader compilation, independent metallib dispatch and native diagnostics.
import std/[json, os, tempfiles]
import ../../src/arraymancer
import ./[parity_support, metal_math_probes]

const customSource = staticRead(currentSourcePath().parentDir.parentDir / "apple/kernels/double.metal")

proc metalKernelObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.metal.custom_kernels", profile: "apple-metal", execution: aeError,
    actual: newJObject())
  try:
    let device = metalDevice()
    let libraries = [compileMetalShader(device, customSource),
      loadMetalLibrary(device, arraymancerAppleLibDir / "arraymancer_test.metallib")]
    var invocations = newJArray()
    var lastOutput: MetalTensor[float32]
    var unchanged, independent = true
    for index, library in libraries:
      let kernel = metalKernel(library, "vector_double")
      let input = [1'f32, 2, 3, 4].toTensor.metal(device)
      let output = newMetalTensorUninit[float32]([4].toMetadata, device)
      dispatchMetalKernel(kernel, [metalBinding(input, mbaRead), metalBinding(output, mbaWrite)],
        [4, 1, 1], [4, 1, 1])
      let observed = observeMetal(output)
      result.actual[if index == 0: "compiled" else: "loaded"] = observed
      result.differences.add compareObservations(%*{"shape": [4], "numel": 4,
        "dtype": "float32", "device": "metal", "values": [2.0, 4.0, 6.0, 8.0]}, observed, 0, 0)
      unchanged = unchanged and input.cpu == [1'f32, 2, 3, 4].toTensor
      independent = independent and input.storage != output.storage
      invocations.add metalEvidence(output)
      lastOutput = output
    var compileError, libraryError, functionError: bool
    var diagnostics = newJArray()
    try: discard compileMetalShader(device, "kernel void broken( this is not Metal source")
    except MetalShaderCompileError as error:
      compileError = true
      diagnostics.add %*{"category": "shader_compile", "message": error.msg}
    let (file, path) = createTempFile("arraymancer-corrupt-", ".metallib")
    file.write("not a metallib")
    file.close()
    defer: removeFile(path)
    try: discard loadMetalLibrary(device, path)
    except MetalLibraryLoadError as error:
      libraryError = true
      diagnostics.add %*{"category": "library_load", "message": error.msg}
    try: discard metalKernel(libraries[0], "absent_function")
    except MetalMissingFunctionError as error:
      functionError = true
      diagnostics.add %*{"category": "missing_function", "message": error.msg}
    let facts = %*{"inputs_unchanged": unchanged, "outputs_independent": independent,
      "malformed_source_errors": compileError,
      "corrupt_library_errors": libraryError, "missing_function_errors": functionError}
    result.actual["invariants"] = facts
    result.differences.add compareObservations(%*{"inputs_unchanged": true,
      "outputs_independent": true, "malformed_source_errors": true,
      "corrupt_library_errors": true, "missing_function_errors": true}, facts, 0, 0)
    result.evidence = metalEvidence(lastOutput)
    result.evidence["invocations"] = invocations
    result.evidence["native_diagnostics"] = diagnostics
    result.evidence["loaded_library_path"] = %(arraymancerAppleLibDir / "arraymancer_test.metallib")
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
