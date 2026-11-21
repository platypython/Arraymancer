# Beat measured PyTorch benchmark losses before further parity expansion

## Current sequencing decision — 2026-11-21

The user instructed: “Note the performance gap and continue.” Resume parity
implementation; the historical performance criterion remains open rather than
blocking API slices.

V14 passed validation and every median comparison, but missed four of 123 p95
comparisons: round 1 Metal exp (1.24853× Torch) and preallocated ln (1.12417×),
round 3 CPU preallocated sqrt (1.03294×) and Metal vector-matrix (1.81128×).
The V13 and V14 CPU executables are byte-identical; V13 passed all 114 CPU
comparisons. Neither result establishes three unchanged all-row wins.

Native-only tracing found substantial post-driver GPU-start gaps and prompt
completion callbacks. The trace supplies no wait reason or causal attribution;
it does not justify queue/QoS changes or a claim that the performance gap is fixed.

## Context

Pause parity expansion and beat PyTorch in every currently losing benchmark, committing verified production optimizations slice by slice. The user chose **both median and p95 lower in three consecutive paired runs** as the win criterion. Of 41 historical rows, 31 fail that gate: 12 CPU rows, including rectangular matmul's p95-only loss, and all 19 native Metal/MPS computation rows.

Production HEAD at planning entry is `f390bd9d4a`; the historical benchmark measures immutable baseline `f528af4467`. Preserve the original benchmark and measure current-source candidates separately. A passing result is limited to the measured hardware, versions, shapes and settings—not a universal performance claim.

## Findings

- `run_benchmark.py` hardcodes `f528af4467`; its archive/native-hash checks prevent measuring optimizations unchanged. `compile_runner` already accepts source/native roots, and `tools/build_apple.nim` supports a separate output directory.
- The protocol is float32, seed `20261116`, one thread, 20 warmups and 15 samples of 50 calls; transfers use five calls/sample. p95 is nearest-rank over sample-average latency. The runner fences GPU samples and protects output computation with compiler barriers.
- CPU normal/preallocated operations use different map/apply entry points. A Tensor-only unary optimization cannot improve preallocated `apply2_inline` calls. Broadcast loops, scalar sum accumulation and expanded MSE-gradient temporaries are concrete shared optimization targets.
- Metal submission is already asynchronous. Per dispatch it creates a command buffer/encoder and pending records; native Clang compilation has no optimization flag. Broadcast-batch matmul submits 12 matrix commands. Dot uses the 16×16 matrix shader despite having one output; sum/mean use two passes for 16,384 elements.
- Existing stream, custom-kernel, memory and optimizer tests enforce event ordering, pending owners, queue-local fences, separate `step`/reset semantics and stable parameter/gradient storage. Preserve those contracts.
- Public system APIs available for optimization: [vForce exponential](https://developer.apple.com/documentation/accelerate/vvexpf(_:_:_:)), [vector sums](https://developer.apple.com/documentation/accelerate/vector-summation), [MPS matrix multiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication) and [MPS matrix/vector multiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixvectormultiplication). Their speed and numerical suitability must be measured.

## Approach

### 1. Measure current candidates, not the historical snapshot

Extend `benchmarks/pytorch_head_to_head/run_benchmark.py` with `--candidate` (`store_true`), preserving default baseline behavior.

- Require explicit `--output`; reject historical `build/pytorch_head_to_head` and its descendants.
- Compile the existing Nim runner against current repository `src`, with unchanged release/ARC settings. Build candidate native code using `tools/build_apple.nim --backend=metal --output=<output>/candidate_native`, and pass that directory through `-d:arraymancerAppleLibDir=...`. Never overwrite/load old native artifacts for candidate measurements.
- Record HEAD and a SHA256 source manifest for `src/**`, `tools/build_apple.nim`, `nim.cfg`, `arraymancer.nimble` and both benchmark sources. Store `candidate_provenance.json`, runner/native hashes and exact commands. In candidate results use `candidate_provenance`, never mislabel current code as the old baseline. Hash source and inputs before and after timing; changed content invalidates the run. Candidate reuse validates candidate provenance/hashes; baseline reuse stays unchanged.
- Reuse existing inputs, PyTorch callables, correctness comparisons, metadata, timing and table generation. Recheck output and gradients before timing. Preserve dtype, seed, shapes, input ranges, one-thread policy, `atol=rtol=5e-5`, gradient requests, optimizer settings, allocation labels and synchronization boundaries.
- Extend existing untimed `torch_train_check`, `cpuTrainCheck` and Metal training checks with a nested `trajectory` object containing `step_1`, `step_2`, `step_3`, each recording loss, pre-update weight gradient, post-update weight and cleared gradient. Keep existing first-step fields and the 19-case count; leave timed training callables unchanged. Nested objects reuse the current recursive comparator without changing numeric-array handling.
- Add `--timing-order=original|arraymancer-first|pytorch-first`, default `original` to preserve the current CPU/PyTorch-first and Metal/Arraymancer-first ordering. Apply explicit orders to whole backend timing blocks, never concurrent jobs; record `timing_order` in each candidate result and alternate explicit orders for acceptance rounds.

Add isolated `benchmarks/pytorch_head_to_head/check_performance_gate.py` taking exactly three candidate `results.json` paths. Validate matching source/input hashes, workloads, hardware/toolchain, allocations and thread settings. Require exactly the existing 19 `FORWARD_CASES` on each resident backend plus the three named Metal/MPS transfer rows, and 19 passing correctness cases per backend. Reject missing/unsupported/failed rows, nonpositive/nonfinite samples, changed settings or changed candidate code. Recompute median/p95 from raw samples. Exit 0 only when both Arraymancer metrics are strictly lower than PyTorch's for every row in all three runs; otherwise print every failing row with both ratios. The existing harness has no performance acceptance gate, so this checker is necessary.

Initial CPU worklist: `broadcast_add`, `sum`, `mean`, `exp`, `ln`, `sqrt`, `mse_forward`, `mse_backward`, `broadcast_add_preallocated`, `exp_preallocated`, `ln_preallocated`, `matmul_rectangular`. Metal worklist: all 19 existing `FORWARD_CASES`. Recheck the ten existing winners, including transfers; any regression joins the worklist.

All benchmark-directory sources, including extensions, remain isolated and uncommitted under the prior benchmark-agent ownership constraint. Commit only verified production/test changes. Do not change dependencies/configuration pins, add threads, use result caches, substitute CPU for GPU or hard-code benchmark shapes/seeds.

### 2. Beat CPU broadcast add through shared iteration

Keep arbitrary-expression map/apply signatures unchanged in `higher_order_applymap.nim`. Add offset-aware C-contiguous raw-pointer loops to `map_inline`, `map2_inline`, `map3_inline`, `apply2_inline` and `apply3_inline`, evaluating each tensor expression once. Flat loops require matching C-order logical traversal; managed/non-memory-copyable element types keep their storage path.

For broadcasts, collapse compatible trailing axes into inner spans with each input's inner stride zero or one. Calculate outer offsets once per span and run a plain vectorizable inner loop; choose from metadata, not dimensions such as `[64,96]`. Follow `laser/strided_iteration/foreach.nim`'s contiguous/strided split, but do not copy its `restrict` assumptions onto aliased apply buffers. Aliases and unsupported layouts retain existing traversal.

Both normal `+.` in `operators_broadcasted.nim` and existing preallocated `apply3_inline` must benefit. Gate both rows before completing this slice.

### 3. Beat CPU sum/mean and rectangular-matmul tails

Add private system-Accelerate bindings in new `src/arraymancer/tensor/private/p_accelerate.nim`; no equivalent bindings were found in the inspected mapping/reduction paths. For macOS float32 contiguous sum use `vDSP_sve` with offset-aware pointers and stride one. Empty input returns zero without dereference. Mean reuses that sum plus one division, preserving empty floating mean NaN. Keep scalar/generic, integer/complex, non-macOS and strided behavior.
Declare `vDSP_sve` with float32 pointers, `clong` stride and `culong` length; vForce functions take output/input float32 pointers and a pointer to `cint` count. Gate bindings and `-framework Accelerate` linkage with `defined(macosx)`. Keep `UnaryMathOp` in the higher-order module, not the binding module, to avoid an import cycle.

In `operators_blas_l2l3.nim:matmulImpl`, add a rank-two/rank-two path before vector promotion/batch metadata: validate contraction, allocate final output once, zero empty K, call existing `gemm`. In `private/p_operator_blas_l2l3.nim:blasMM_C_eq_aAB_p_bC`, bypass `asContiguous` aliases/layout rediscovery when A/B/C are C-contiguous; call existing nimblas row-major GEMM with no transposes and `lda=K`, `ldb=N`, `ldc=N`. Keep general layouts and non-BLAS paths.

Gate sum/mean/rectangular matmul, and recheck all CPU matrix/vector/batch rows sharing this code. Do not change the BLAS library or compiler configuration.

### 4. Beat CPU unary operations, including preallocated variants

Introduce a shared production bulk-unary path in `higher_order_applymap.nim`, without recognizing arbitrary-expression ASTs:

```nim
type UnaryMathOp* = enum
  unaryExp, unaryLn, unarySqrt
proc applyUnary*(dest: var Tensor[float32], src: Tensor[float32],
                 operation: static[UnaryMathOp])
```

Require equal shapes (`ValueError` otherwise). Empty/scalar shapes survive. Disjoint C-contiguous macOS buffers use `vvexpf`, `vvlogf`, `vvsqrtf` through `p_accelerate.nim`; split counts exceeding `high(cint)` into consecutive chunks without copying. Aliases/unsupported layouts/platforms use existing scalar `apply2_inline`. This bulk intrinsic path does not exist in the current arbitrary-expression templates.

Replace only generated Tensor `exp`, `ln`, `sqrt` declarations in `ufunc.nim` with equivalent generic overloads: float32 allocates once and calls `applyUnary`; other dtypes retain existing scalar mapping/empty handling. Leave custom `makeUniversal` behavior unchanged. `tensor.nim` already exports `higher_order_applymap`.

Route CPU preallocated unary benchmark calls through this same production `applyUnary` enum API. Keep output reuse/timing labels; no benchmark-local kernels. Verify IEEE exceptional values before enabling vForce, and gate normal/preallocated exp/log/sqrt, including the previously winning preallocated sqrt row.

### 5. Beat CPU MSE without expanded backward temporaries

Keep `mse_loss` signatures, reduction enum and gate registration. In `nn/loss/mean_square_error_loss.nim`, apply shared inner-span traversal to forward differences/squares and scalar accumulation; allocate only the final scalar for mean/sum.

Add private:

```nim
proc mseBackwardCpu[T: SomeFloat](upstream, prediction, target: Tensor[T],
    reduction: MseReduction, needPrediction, needTarget: bool):
    tuple[prediction, target: Tensor[T]]
```

Allocate/zero only requested final gradient shapes. Traverse broadcasted errors once, compute the scaled seeded difference and accumulate into original operand coordinates, negating the target contribution. Do not materialize an expanded gradient or mutate a broadcast view. Preserve actual upstream strides, `mseNone` seeds, empty zero gradients, empty mean NaN, target-only gradients and shared-Variable accumulation. Unrequested tuple fields remain default/unread tensors.

Route CPU `mse_backward_ag` through this helper; do not change unrelated autograd retention/reset. Gate both MSE rows and recheck the full CPU training step.

### 6. Beat native Metal matrix products

First add `-O3` to native Clang compilation in `tools/build_apple.nim:buildMetal`; retain shader `-fno-fast-math`. Link system `MetalPerformanceShaders` only in this optional Apple build.

Keep operation 20 and existing ABI layouts. Extend native matmul validation for promoted rank-two/broadcast-batched descriptors, checking batch broadcasting, output dimensions, dtype/device, bounds/capacity and grid overflow before encoding. In `operators_metal.nim:*`, replace the per-batch host loop with one dispatch using batch-aligned zero-stride views; retain vector-axis removal and one owning result allocation.

Choose by general dimensions/layout:

- Dot (`M=N=1`): cooperative multiply/reduce GPU kernel, not matrix tiles.
- Dense positive-stride matrix/vector and vector/matrix: cached `MPSMatrixVectorMultiplication`, using the matrix transpose for vector/matrix.
- Unbatched dense matrices: cached `MPSMatrixMultiplication`, alpha one/beta zero, direct existing buffers/offsets.
- Broadcast batches or layouts not representable directly by MPS: extend existing tiled Metal shader with a batch grid dimension and descriptor-derived offsets. Preserve signed/zero strides, tile edges and empty contractions; submit all batches together.

Cache MPS kernels by device/stream/dimensions/transpose flags. Per-call wrappers and normal outputs remain timed. Use existing leases, owner retention and evidence; MPS-incompatible layouts use the GPU strided shader, never CPU, packing or hidden transfers. Gate all six Metal matrix-product rows and their backward/training consumers.

### 7. Beat native Metal reductions, elementwise math and MSE

Use shader names `math_dot`, `math_matmul_batched`, `math_elementwise_packed`, `math_elementwise_inner_span`, `math_reduce_small`, `math_mse_reduce` and `math_mse_backward_reduce` for specialized paths. Existing general kernels remain fallbacks, not compatibility aliases.

Add packed elementwise direct `offset+index` addressing plus metadata-selected inner-span broadcasts; preserve `math_elementwise` as the general signed/zero-stride fallback. Normal/preallocated calls select identical kernels and retain ordinary allocations/IEEE instructions.

For sum/mean of at most 65,536 elements, use one 256-lane workgroup: each lane accumulates strided chunks, reduce within SIMD groups then subgroup partials, normalize mean once. Change `aggregate_metal.nim:reduce` and native operation-21 validation together to select one scalar output at this threshold; larger inputs retain the existing `ceil(count/256)` partial outputs and final pass. Empty sum/mean remain zero/NaN without dereference. Do not hard-code 16,384 or leave inconsistent group-count contracts.

Add math operation **24**, fused squared-difference mean reduction, through existing `am_math` with `reserved=0`. Validate broadcast-aligned matching inputs and scalar output. Use the same cooperative reduction; larger inputs may allocate GPU partial buffers retained through completion. Route `mse_forward_metal` through it, eliminating the full squared-difference tensor.

Keep `am_mse_backward`'s ABI signature but extend validation/shader selection to outputs reducible to an original operand shape. Equal shapes retain pointwise backward; reduced outputs use one cooperative group per output element summing broadcast contributions. Read upstream on-device; empty reductions produce zero, not zero times infinite normalization.

Add `mse_backward_reduced_metal*(upstream, prediction, target: MetalTensor[float32]; target_shape: Metadata): MetalTensor[float32]` in `nnp_mse_metal.nim`; allocate final gradient and call the extended entry point. Route the existing full-gradient primitive through this helper with full prediction shape. In mean-MSE's Metal gate use the reduced helper for each requested operand, swapping prediction/target for the target gradient; remove expanded-gradient/negation temporaries and subsequent unbroadcast passes on this path. Preserve other reduction modes.

Do not skip gradient initialization, accumulation/reset, SGD update or zeroing. Gate all remaining Metal normal/preallocated rows, MSE and full training.

### 8. Amortize default-stream submission only if Metal rows still fail

If steps 6–7 leave any Metal computation row failing, aggregate built-in dispatches internally on the default stream; explicitly created streams retain eager commit. Do not add a public benchmark batch API.

Add pending command buffer/encoder state to native `AMStream` and `flushMathBatch(AMStream *)`. Commit at 32 encoded dispatches, 4 MiB retained output capacity or 100 microseconds from the first dispatch, whichever occurs first. An independent timer guarantees forward progress without more user calls. End compute encoding before an MPS encode while retaining the same command buffer.

Serialize admission/encoding with the existing stream lock and preserve existing lock order. Validate/admit leases before appending a dispatch, so rejected calls cannot execute later. Retain accepted sequences, per-invocation evidence, buffers and pipeline/MPS objects even before commit; finalize every record after shared-buffer completion. Never recycle in-flight outputs. Timer flush failures enter existing stream failure reporting and release accepted leases, not silent success.

Flush before markers/fences, stream/device sync/completion observation, event record/wait, foreign queue exposure, transfer/upload/host or foreign lease waits, capture boundaries and separately submitted copy/set/RNG/custom commands. Before a host wait, flush all producer streams outside state/lease locks; never wait on uncommitted work. Memory pressure/cache clearing flushes before waiting/reclaiming. Do not wait for unrelated blocked streams.

Public native signatures and `reserved` remain unchanged. Preserve custom-kernel, memory and Core AI fence contracts; add default-stream coverage because current interop tests mostly exercise explicit streams. Skip this architectural change entirely if earlier optimizations already meet the gate.

### 9. Commit verified slices and enforce the parity barrier

Commit each production optimization family only after focused correctness and three paired runs pass both metrics for that family. Use eval `commit.draft({agents: "self"})` and select only this task's production/test lines. Exclude all benchmark paths and unrelated changes; never use `git add`/`git commit`.

Retain raw samples, exact commands, source/native hashes and concise per-row median/p95 tables for every run. Final acceptance requires all 41 rows passing both metrics in three consecutive fresh paired runs. A tie, missing/unsupported row, mismatch, invalid timing or ratio at/above one remains pending. Do not cherry-pick runs, weaken precision/gates, drop cases or resume parity while any row fails. Report remaining losses explicitly rather than claiming success.

## Verification

### Prerequisites and measurement commands

Working directory: `/Users/derrick/git/ai/Arraymancer`. Use existing `build/parity-oracle/bin/python` (CPython 3.13.18, Torch 2.15.0), Nim 2.4.2, installed BLAS/LAPACK and Apple SDK. No dependency or pin changes.

Every shell script must pass ShellCheck before execution. Every edited Python source must pass exact-source Python LSP diagnostics before execution; if generic LSP is unconfigured, connect `pyright-langserver --stdio` through JS eval, send initialize/didOpen and require a matching error-free diagnostics publication.

Before Torch import and in every child:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
```

For each slice use distinct directories, without overwriting the historical report:

```sh
build/parity-oracle/bin/python benchmarks/pytorch_head_to_head/run_benchmark.py --candidate --output build/pytorch_performance/SLICE/round_1 --timing-order=arraymancer-first
build/parity-oracle/bin/python benchmarks/pytorch_head_to_head/run_benchmark.py --candidate --output build/pytorch_performance/SLICE/round_2 --timing-order=pytorch-first
build/parity-oracle/bin/python benchmarks/pytorch_head_to_head/run_benchmark.py --candidate --output build/pytorch_performance/SLICE/round_3 --timing-order=arraymancer-first
```

CPU intermediate slices may add existing `--cpu-only`; final acceptance cannot. The checker can report intermediate family ratios, but its successful full-gate exit always requires 41 rows. Run timing sequentially on a quiet machine; no competing builds/tests/benchmarks during samples. A changed source hash or competing activity invalidates a round; retain it as invalid and start a fresh three-run sequence.

After three full final runs:

```sh
build/parity-oracle/bin/python benchmarks/pytorch_head_to_head/check_performance_gate.py build/pytorch_performance/final/round_1/results.json build/pytorch_performance/final/round_2/results.json build/pytorch_performance/final/round_3/results.json
```

Checker regression inputs: p95-only loss, missing transfer row, changed input/source hash, failed gradient check and NaN samples each fail; three complete records with lower medians/p95s pass. Synthetic checker records are never performance evidence.

### Correctness proof

- Extend existing `tests/tensor/test_broadcasting.nim` and `test_ufunc.nim` for offset/transposed/negative-stride/zero-stride reads, aliases, scalar/empty shapes and float32/float64. Reversed `[1,4,9,16]` sqrt produces logical `[4,3,2,1]` without altering its owner. Allocating and `applyUnary` outputs agree, including NaN, infinities, ±0 and invalid log/sqrt domains.
- Keep aggregate and pinned 30-case CPU parity tests passing. Empty float32 sum is zero and mean NaN; preserve empty-axis shapes. Verify dense/transposed/offset/empty-contraction GEMM and existing vector/batch contracts.
- Reuse CPU/Metal MSE regressions: prediction `[[2],[4]]`, target `[1,2,3]`, mean loss `16/6`, gradients `[[0],[2]]` and `[-8/6,-4/6,0]`. Include non-unit seeds, target-only/shared operands and empty zero gradients.
- Metal math tests cover dot, both vector orientations, rectangular tile edges, batches, signed/zero strides, empty K, reduction sizes crossing group thresholds and nonfinite unary values. Benchmark checks compare authentic Torch outputs/gradients for every measured shape before timing.
- Preserve optimizer storage identity, `step` retaining gradients and `update` clearing them. Add a deterministic three-step matmul→MSE→backward→SGD comparison with authentic Torch states to catch stale/reused gradient buffers.
- If aggregation runs, add default-stream pending-batch readback, dropped-owner, timer-forward-progress, event handoff and foreign-queue fence checks. Existing explicit-stream blocked-queue tests must still complete without globally draining unrelated work.

Run affected focused modules once per applied slice; no repository-wide suite. Concrete commands:

```sh
nim c -r -d:ssl -o:build/test_broadcasting_performance tests/tensor/test_broadcasting.nim
nim c -r -d:ssl -o:build/test_ufunc_performance tests/tensor/test_ufunc.nim
nim c -r -d:ssl -o:build/test_aggregate_performance tests/tensor/test_aggregate.nim
nim c -r -d:ssl -o:build/test_gate_basic_performance tests/autograd/test_gate_basic.nim
nim c -r -d:ssl -o:build/pytorch_parity_performance tests/parity/run_parity.nim --profile=cpu --output=build/parity-performance.json
nim c -r -d:release -o:build/build_apple tools/build_apple.nim --backend=metal
nim c -r -d:ssl -d:metal -o:build/test_metal_math_performance tests/apple/test_metal_math.nim
nim c -r -d:ssl -d:metal -o:build/test_metal_autograd_performance tests/apple/test_metal_autograd.nim
nim c -r -d:ssl -d:metal -o:build/test_metal_optimizer_performance tests/apple/test_metal_optimizer.nim
nim c -r -d:ssl -d:metal -o:build/test_metal_streams_performance tests/apple/test_metal_streams.nim
```

If shared submission changes, additionally run:

```sh
nim c -r -d:ssl -d:metal -o:build/test_metal_kernels_performance tests/apple/test_metal_kernels.nim
nim c -r -d:ssl -d:metal -o:build/test_metal_memory_performance tests/apple/test_metal_memory.nim
nim c -r -d:release -o:build/build_apple tools/build_apple.nim --backend=all
nim c -r -d:ssl -d:coreai -o:build/test_coreai_streams_performance tests/apple/test_coreai_streams.nim
```

The last test requires the authentic `tests/apple/models/parity.aimodel` and compiled Core AI dependency produced by the existing builder. Do not introduce Swift changes just to optimize Metal submission.

## Assumptions and contingencies

- **User-selected gate:** both median and p95 lower in three consecutive paired runs; no optional tail criterion.
- Initial target is the existing shapes/settings and workstation. No claim extends to every tensor size/device/platform.
- System Accelerate/MPS fast paths retain portable/general GPU fallbacks. If an intrinsic fails numerical behavior, keep it disabled and continue that operation's kernel work, not relaxed correctness.
- Unavailable hardware/API, compilation failure, contention or a still-losing row is explicit unfinished work, never a win or permission to resume parity.
- Benchmark sources remain runnable local deliverables and uncommitted; verified production improvements are committed as they land.

