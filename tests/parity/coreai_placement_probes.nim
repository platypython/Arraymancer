# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes actual SDK policy facts, never inferring hardware placement from preferences.
import std/[json, options]
import ../../src/arraymancer
import ../apple/coreai_test_support
import ./[parity_support, coreai_model_probes]

proc coreAIPlacementObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.coreai.compute_placement", profile: "apple-coreai", execution: aeError, actual: newJObject())
  try:
    var availableMask: uint32
    for unit in availableCoreAIComputeUnits(): availableMask = availableMask or (1'u32 shl ord(unit))
    var cases = newJArray()
    var invocations = newJArray()
    for policy in CoreAIComputePolicy:
      let function = openCoreAIModel(coreAISourceAsset, policy).loadFunction("identity").get
      let job = function.submitInference([CoreAIBinding(name: "x", array: toCoreAI([1.25'f32, -2].toTensor))],
        states = [], outputViews = [])
      let outputs = job.wait()
      var observed = newJArray()
      for binding in outputs: observed.add %*{"name": binding.name, "array": observeCoreAI(binding.array)}
      result.differences.add compareObservations(%*[{"name": "y", "array": {"shape": [2],
        "dtype": "float32", "backend": "coreai", "values": [1.25, -2.0]}}], observed, 0, 0)
      let native = job.coreAIJobEvidence
      let valid = native.requestedPolicy == policy and native.availableMask == availableMask and
        native.allowedMask != 0 and native.actualPlacement == -1 and
        (policy != cpCpuOnly or native.allowedMask == 1) and
        (policy != cpGpuPreferred or native.preferredUnit == 1) and
        (policy != cpNeuralEnginePreferred or native.preferredUnit == 2)
      result.differences.add compareObservations(%true, %valid, 0, 0)
      let evidence = coreAIEvidence(job)
      cases.add %*{"policy": $policy, "outputs": observed, "evidence": evidence}
      invocations.add evidence
      if policy == cpCpuOnly: result.evidence = evidence.copy()
    result.actual["policies"] = cases
    result.actual["hardware_specific_placement"] = newJNull()
    result.evidence["invocations"] = invocations
    result.evidence["placement_source"] = %"SDK exposes no per-invocation actual unit; this invocation was not independently traced"
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
