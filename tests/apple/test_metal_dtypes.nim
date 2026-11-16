# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises all pinned representations through distinct native GPU buffers.
import std/[unittest, json]
import ../parity/[parity_support, metal_dtype_probes]
suite "Native Metal storage dtypes":
  test "Every pinned dtype preserves exact bits on GPU copy":
    let observation = metalDtypeObservation()
    if observation.error != nil: echo observation.error
    for difference in observation.differences: echo difference.path
    check observation.execution == aePass
    check observation.actual.len == 15
    check observation.evidence["invocations"].len == 14
    for invocation in observation.evidence["invocations"]:
      check invocation["kind"].getStr == "gpu_command"
      check invocation["completed"].getBool
