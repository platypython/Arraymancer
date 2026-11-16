# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes a complete native training step without conflating update with gradient reset.
import std/json
import ../../src/arraymancer
import ./[parity_support, metal_math_probes]

proc metalOptimizerObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.metal.optimizer_sgd", profile: "apple-metal", execution: aeError,
    actual: newJObject())
  try:
    let fixtures = loadMetalFixtures()
    for item in fixtures["cases"]:
      if item["id"].getStr != "optim.sgd_step": continue
      let inputs = item["inputs"]
      let ctx = newContext(MetalTensor[float32])
      let weights = ctx.variable(metalInput(inputs["weights"]), requires_grad = true)
      let x = ctx.variable(metalInput(inputs["input"]))
      let frozen = ctx.variable([9'f32].toTensor.metal)
      let prediction = (weights *. x).sum
      let loss = prediction.mse_loss(metalInput(inputs["target"]))
      loss.backprop
      let valueOwner = weights.value.storage
      let gradOwner = weights.grad.storage
      var actual = %*{"prediction": observeMetal(prediction.value), "loss": observeMetal(loss.value),
        "grad": observeMetal(weights.grad)}
      var invocations = %*[metalEvidence(prediction.value), metalEvidence(loss.value), metalEvidence(weights.grad)]
      let optim = optimizer((weights: weights, frozen: frozen), SGD,
        float32(inputs["learning_rate"].getFloat), MetalTensor[float32])
      optim.step
      actual["weights_after_step"] = observeMetal(weights.value)
      invocations.add(metalEvidence(weights.value))
      actual["grad_after_step"] = observeMetal(weights.grad)
      optim.zeroGrads
      actual["grad_after_zero"] = observeMetal(weights.grad)
      result.evidence = metalEvidence(weights.grad)
      invocations.add(result.evidence)
      result.evidence = parseJson($result.evidence)
      result.evidence["invocations"] = invocations
      result.actual["optim.sgd_step"] = actual
      result.differences = compareObservations(metalExpected(item["expected"]), actual,
        item["atol"].getFloat, item["rtol"].getFloat)
      let invariants = %*{"value_owner_preserved": weights.value.storage == valueOwner,
        "gradient_owner_preserved": weights.grad.storage == gradOwner,
        "frozen_unchanged": frozen.value.cpu == [9'f32].toTensor}
      result.actual["invariants"] = invariants
      result.differences.add compareObservations(%*{"value_owner_preserved": true,
        "gradient_owner_preserved": true, "frozen_unchanged": true}, invariants, 0, 0)
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
