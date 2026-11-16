# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Executes selected native contracts; expectations never become execution evidence.
import std/json
import ../../src/arraymancer
import ./parity_support
import ./metal_dtype_probes
import ./metal_view_probes
import ./metal_math_probes
import ./metal_autograd_probes

proc deviceDispatch(): AppleObservation =
  result.id = "apple.metal.device_dispatch"
  result.profile = "apple-metal"
  result.execution = aeError
  try:
    let input = [1'f32, 2, 3, 4, 5, 6].toTensor
    let device = metalDevice()
    let native = input.metal(device)
    let readback = native.cpu
    var unavailable = false
    try: discard metalDevice(metalDeviceCount())
    except MetalUnavailableError: unavailable = true
    var evidence: AMEvidence
    checkMetal metalApi().evidence(native.storage.owner.handle, addr evidence)
    result.evidence = %*{"invocation_id": evidence.invocationId,
      "kind": "device_storage_control", "completed": evidence.completed == 1,
      "device_registry_id": $evidence.registryId, "kernel_labels": [],
      "transfer_bytes": evidence.transferBytes, "materialization_bytes": evidence.materializationBytes,
      "cpu_fallback_count": evidence.cpuFallbackCount}
    result.actual = %*{"available": metalIsAvailable(), "device_count_positive": metalDeviceCount() > 0,
      "explicit_device_matches": native.storage.device.registryId == device.registryId,
      "readback_matches": readback == input,
      "input_unchanged": input == [1'f32, 2, 3, 4, 5, 6].toTensor,
      "unavailable_request_errors": unavailable}
    result.differences = compareObservations(%*{"available": true, "device_count_positive": true,
      "explicit_device_matches": true, "readback_matches": true, "input_unchanged": true,
      "unavailable_request_errors": true}, result.actual, 0, 0)
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": (if error of MetalUnavailableError: "unavailable" else: "native_error"),
      "type": $error.name, "message": error.msg}

proc runMetalRequirements*(requirementId: string): seq[AppleObservation] =
  if requirementId.len == 0 or requirementId == "apple.metal.device_dispatch":
    result.add deviceDispatch()
  if requirementId.len == 0 or requirementId == "apple.metal.dtypes":
    result.add metalDtypeObservation()
  if requirementId.len == 0 or requirementId == "apple.metal.tensor_views":
    result.add metalViewObservation()
  if requirementId.len == 0 or requirementId == "apple.metal.tensor_math":
    result.add metalMathObservation()
  if requirementId.len == 0 or requirementId == "apple.metal.autograd":
    result.add metalAutogradObservation()
