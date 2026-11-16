# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides exact representation probes shared by the native gate and focused test.
import std/[json, strutils, complex]
import ../../src/arraymancer
import ./parity_support

proc bits[T](t: Tensor[T]): JsonNode =
  result = newJArray()
  for i in 0 ..< t.size:
    let bytes = cast[ptr UncheckedArray[uint8]](t.get_offset_ptr)
    var hex = ""
    for j in countdown(sizeof(T) - 1, 0): hex.add(toHex(bytes[i * sizeof(T) + j], 2).toLowerAscii)
    result.add(%hex)

proc dtypeProbe*[T](values: openArray[T]; name: string): tuple[actual, expected, evidence: JsonNode] =
  var input = newTensorUninit[T](values.len)
  if values.len > 0: copyMem(input.get_offset_ptr, unsafeAddr values[0], values.len * sizeof(T))
  let source = input.metal
  let output = source.clone
  let readback = output.cpu
  let scalar = newTensorUninit[T]([])
  var initializedScalar = scalar
  copyMem(initializedScalar.get_offset_ptr, unsafeAddr values[0], sizeof(T))
  let scalarCopy = initializedScalar.metal.clone.cpu
  let empty = newMetalTensorUninit[T](2, 0, 3).clone.cpu
  result.actual = %*{"dtype": name, "device": "metal", "width": sizeof(T),
    "distinct_buffers": source.storage.owner.handle != output.storage.owner.handle,
    "values": bits(readback), "scalar_shape": scalarCopy.shape.len,
    "scalar_values": bits(scalarCopy), "empty_size": empty.size}
  result.expected = %*{"dtype": name, "device": "metal", "width": sizeof(T),
    "distinct_buffers": true, "values": bits(input), "scalar_shape": 0,
    "scalar_values": bits(initializedScalar), "empty_size": 0}
  var evidence: AMEvidence
  checkMetal metalApi().evidence(output.storage.owner.handle, addr evidence)
  result.evidence = %*{"invocation_id": evidence.invocationId,
    "completed": evidence.completed == 1, "kind": (if evidence.kind == 2: "gpu_command" else: "device_storage_control"),
    "device_registry_id": $evidence.registryId, "kernel_labels": ["copy" & $evidence.kernelBits],
    "transfer_bytes": evidence.transferBytes, "materialization_bytes": evidence.materializationBytes,
    "cpu_fallback_count": evidence.cpuFallbackCount}

proc metalDtypeObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.metal.dtypes", profile: "apple-metal", execution: aeError,
    actual: newJArray())
  var expected = newJArray()
  var invocations = newJArray()
  try:
    template probe(values, name: untyped) =
      block:
        let observation = dtypeProbe(values, name)
        result.actual.add(observation.actual)
        expected.add(observation.expected)
        result.evidence = observation.evidence
        invocations.add(observation.evidence)
    probe([1'f32, -2.5'f32], "float32")
    probe([MetalFloat16.fromBits(0x8000), MetalFloat16.fromBits(0x7e01)], "float16")
    probe([MetalBFloat16.fromBits(0x3f80), MetalBFloat16.fromBits(0xff80)], "bfloat16")
    probe([low(int8), -1'i8], "int8")
    probe([low(int16), -1'i16], "int16")
    probe([low(int32), -1'i32], "int32")
    probe([low(int64), -1'i64], "int64")
    probe([high(uint8), 0'u8], "uint8")
    probe([high(uint16), 0'u16], "uint16")
    probe([high(uint32), 0'u32], "uint32")
    probe([high(uint64), 0'u64], "uint64")
    probe([false, true], "bool")
    probe([MetalComplexFloat16(re: MetalFloat16.fromBits(0x3c00), im: MetalFloat16.fromBits(0xc000))], "complex32")
    probe([complex(1'f32, -2'f32), complex(-3'f32, 4'f32)], "complex64")
    var rejectedFloat, rejectedComplex = false
    try: discard newMetalTensorUninit[float64](1)
    except MetalDtypeError: rejectedFloat = true
    try: discard newMetalTensorUninit[Complex64](1)
    except MetalDtypeError: rejectedComplex = true
    result.actual.add(%*{"float64_rejected": rejectedFloat, "complex128_rejected": rejectedComplex})
    expected.add(%*{"float64_rejected": true, "complex128_rejected": true})
    result.differences = compareObservations(expected, result.actual, 0, 0)
    result.evidence = parseJson($result.evidence)
    result.evidence["invocations"] = invocations
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
