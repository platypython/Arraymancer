# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises native signed views and mutation without host replacement.
import std/[unittest, json]
import ../parity/[parity_support, metal_view_probes]
suite "Native Metal views":
  test "Aliases mutate storage while clones and materializations remain independent":
    let observation = metalViewObservation()
    if observation.error != nil: echo observation.error
    for difference in observation.differences:
      echo difference.path, ": ", difference.expected, " != ", difference.actual
    check observation.execution == aePass
