# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises real policy specialization while keeping unobserved placement unknown.
import std/[json, options, os, unittest]
import ../../src/arraymancer
import ./coreai_test_support
import ../parity/coreai_model_probes

suite "Core AI compute policy evidence":
  test "All policies execute exactly and retain available, allowed and preferred facts":
    let units = availableCoreAIComputeUnits()
    check cacuCPU in units
    var availableMask: uint32
    for unit in units: availableMask = availableMask or (1'u32 shl ord(unit))
    var invocations = newJArray()
    for policy in CoreAIComputePolicy:
      let function = openCoreAIModel(coreAISourceAsset, policy).loadFunction("identity").get
      let job = function.submitInference([CoreAIBinding(name: "x", array: toCoreAI([1.25'f32, -2].toTensor))],
        states = [], outputViews = [])
      let outputs = job.wait()
      check outputs.len == 1 and outputs[0].name == "y"
      check outputs[0].array.cpu(float32) == [1.25'f32, -2].toTensor
      let evidence = job.coreAIJobEvidence
      check evidence.completed and evidence.kind == 3 and evidence.invocationId > 0
      check evidence.requestedPolicy == policy
      check evidence.availableMask == availableMask
      check evidence.allowedMask != 0
      check evidence.actualPlacement == -1
      case policy
      of cpCpuOnly: check evidence.allowedMask == 1
      of cpGpuPreferred: check evidence.preferredUnit == 1
      of cpNeuralEnginePreferred: check evidence.preferredUnit == 2
      of cpDefault: discard
      invocations.add coreAIEvidence(job)
    let path = getEnv("ARRAYMANCER_COREAI_PLACEMENT_EVIDENCE")
    if path.len > 0:
      writeFile(path, pretty(%*{"pid": getCurrentProcessId(), "invocations": invocations,
        "placement_source": "unknown_without_correlated_compute_unit_events"}) & "\n")
