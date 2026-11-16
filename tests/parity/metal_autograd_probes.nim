# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes native tape outputs and gradients against independent MPS reference data.
import std/json
import ../../src/arraymancer
import ./[parity_support, metal_math_probes]

proc metalAutogradObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.metal.autograd", profile: "apple-metal", execution: aeError,
    actual: newJObject())
  try:
    let fixtures = loadMetalFixtures()
    var invocations = newJArray()
    for item in fixtures["cases"]:
      let id = item["id"].getStr
      if id notin ["autograd.mean_gradient", "loss.mse_mean"]: continue
      let ctx = newContext(MetalTensor[float32])
      let inputs = item["inputs"]
      let input = if id == "autograd.mean_gradient": inputs["x"] else: inputs["prediction"]
      let x = ctx.variable(metalInput(input), requires_grad = true)
      let loss = if id == "autograd.mean_gradient": x.mean else: x.mse_loss(metalInput(inputs["target"]))
      loss.backprop
      let actual = %*{"loss": observeMetal(loss.value), "grad": observeMetal(x.grad)}
      result.actual[id] = actual
      let differences = compareObservations(metalExpected(item["expected"]), actual,
        item["atol"].getFloat, item["rtol"].getFloat)
      for difference in differences:
        result.differences.add Difference(path: "/" & id & difference.path,
          actual: difference.actual, expected: difference.expected)
      invocations.add(metalEvidence(loss.value))
      result.evidence = metalEvidence(x.grad)
      invocations.add(result.evidence)
    result.evidence = parseJson($result.evidence)
    result.evidence["invocations"] = invocations
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
