# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Dispatches only implemented Core AI contracts to real native inference probes.
import ./[parity_support, coreai_model_probes, coreai_layout_probes, coreai_stream_probes, coreai_state_probes, coreai_concurrency_probes, coreai_placement_probes]

proc runCoreAIRequirements*(requirementId = ""): seq[AppleObservation] =
  if requirementId.len == 0 or requirementId == "apple.coreai.model_functions":
    result.add coreAIModelObservation()
  if requirementId.len == 0 or requirementId == "apple.coreai.ndarray_descriptors":
    result.add coreAIDescriptorObservation()
  if requirementId.len == 0 or requirementId == "apple.coreai.strided_layout":
    result.add coreAILayoutObservation()
  if requirementId.len == 0 or requirementId == "apple.coreai.metal_buffer_interop":
    result.add coreAIInteropObservation()
  if requirementId.len == 0 or requirementId == "apple.coreai.stream_lifetime":
    result.add coreAIStreamObservation()
  if requirementId.len == 0 or requirementId == "apple.coreai.stateful_execution":
    result.add coreAIStateObservation()
  if requirementId.len == 0 or requirementId == "apple.coreai.concurrency":
    result.add coreAIConcurrencyObservation()
  if requirementId.len == 0 or requirementId == "apple.coreai.compute_placement":
    result.add coreAIPlacementObservation()
