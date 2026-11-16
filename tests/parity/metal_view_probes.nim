# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Observes alias mutation, signed views, independent clones and GPU materialization.
import std/json
import ../../src/arraymancer
import ./parity_support

proc logical(t: MetalTensor[float32]): JsonNode =
  let host = t.cpu
  result = newJArray()
  let data = cast[ptr UncheckedArray[float32]](host.get_offset_ptr)
  for i in 0 ..< host.size: result.add(%data[i])

proc metalViewObservation*(): AppleObservation =
  result = AppleObservation(id: "apple.metal.tensor_views", profile: "apple-metal", execution: aeError)
  try:
    var base = [1'f32, 2, 3, 4, 5, 6].toTensor.reshape(2, 3).metal
    var view = base.transpose
    view[0, 1] = 9'f32
    let offset = base.metalView([3].toMetadata, [1].toMetadata, 2)
    let reverse = base.metalView([6].toMetadata, [-1].toMetadata, 5)
    var independent = view.clone
    independent[0, 0] = 77'f32
    let packed = view.asContiguous
    let column = base.asContiguous(colMajor)
    let empty = base.metalView([2, 0, 3].toMetadata, [0, 3, 1].toMetadata, high(int))
    var badBounds, badBroadcast, badOverlap = false
    try: discard base.metalView([4].toMetadata, [2].toMetadata, 0)
    except MetalError: badBounds = true
    var broadcast = base.metalView([2, 3].toMetadata, [0, 1].toMetadata, 0)
    try: broadcast[0, 0] = 1'f32
    except MetalError: badBroadcast = true
    var overlap = base.metalView([2, 2].toMetadata, [1, 1].toMetadata, 0)
    try: overlap[0, 0] = 1'f32
    except MetalError: badOverlap = true
    result.actual = %*{"base": logical(base), "view": logical(view),
      "offset": logical(offset), "reverse": logical(reverse), "clone": logical(independent),
      "packed": logical(packed), "column": logical(column),
      "aliases": base.storage == view.storage and base.storage == offset.storage and base.storage == reverse.storage,
      "clone_independent": independent.storage != base.storage,
      "packed_materialized": packed.storage != view.storage,
      "column_materialized": column.storage != base.storage,
      "contiguous_preserved": base.asContiguous.storage == base.storage,
      "column_contiguous": column.is_F_contiguous, "empty_size": empty.cpu.size,
      "bounds_rejected": badBounds, "broadcast_write_rejected": badBroadcast, "overlap_write_rejected": badOverlap}
    let expected = %*{"base": [1.0, 2.0, 3.0, 9.0, 5.0, 6.0],
      "view": [1.0, 9.0, 2.0, 5.0, 3.0, 6.0], "offset": [3.0, 9.0, 5.0],
      "reverse": [6.0, 5.0, 9.0, 3.0, 2.0, 1.0], "clone": [77.0, 9.0, 2.0, 5.0, 3.0, 6.0],
      "packed": [1.0, 9.0, 2.0, 5.0, 3.0, 6.0], "column": [1.0, 2.0, 3.0, 9.0, 5.0, 6.0],
      "aliases": true, "clone_independent": true, "packed_materialized": true,
      "column_materialized": true, "contiguous_preserved": true, "column_contiguous": true,
      "empty_size": 0, "bounds_rejected": true, "broadcast_write_rejected": true, "overlap_write_rejected": true}
    result.differences = compareObservations(expected, result.actual, 0, 0)
    var native: AMEvidence
    checkMetal metalApi().evidence(base.storage.owner.handle, addr native)
    result.evidence = %*{"invocation_id": native.invocationId, "kind": "gpu_command",
      "completed": native.completed == 1 and native.kind == 2, "device_registry_id": $native.registryId,
      "kernel_labels": ["set" & $(-native.kernelBits)], "transfer_bytes": native.transferBytes,
      "cpu_fallback_count": native.cpuFallbackCount, "materializations": []}
    for tensor in [packed, column]:
      var copied: AMEvidence
      checkMetal metalApi().evidence(tensor.storage.owner.handle, addr copied)
      result.evidence["materializations"].add(%*{"invocation_id": copied.invocationId,
        "completed": copied.completed == 1, "materialization_bytes": copied.materializationBytes,
        "transfer_bytes": copied.transferBytes, "kernel_labels": ["copy" & $copied.kernelBits]})
    result.execution = if result.differences.len == 0: aePass else: aeMismatch
  except CatchableError as error:
    result.error = %*{"category": "native_error", "message": error.msg}
