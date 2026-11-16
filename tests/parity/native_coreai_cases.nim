# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Dispatches only implemented Core AI contracts to real native inference probes.
import ./[parity_support, coreai_model_probes]

proc runCoreAIRequirements*(requirementId = ""): seq[AppleObservation] =
  if requirementId.len == 0 or requirementId == "apple.coreai.model_functions":
    result.add coreAIModelObservation()
  if requirementId.len == 0 or requirementId == "apple.coreai.ndarray_descriptors":
    result.add coreAIDescriptorObservation()
