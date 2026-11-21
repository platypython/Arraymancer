## Context

Build and commit the entire remaining Apple parity implementation slice by slice: both native backends, all ten Metal contracts, all eight Core AI contracts, and CPU regression coverage. Reuse the already committed foundation and CPU corrections; every new slice must implement real behavior, build, pass its focused behavioral gate, and then receive a separate selective commit before the next slice. Completion requires genuine native observations for every existing Apple requirement, not annotations, CPU fallback, or framework-only demos.

## Grounded starting point

- Live `git log -10 --oneline` ends at `99b851f fix(autograd): return scalar mean-MSE losses`, preceded by `bf7de87` scalar CPU tensors, `e9b1d58` trailing-axis broadcasts, `6131467` MSE normalization, `2da1989` exception-safe no-grad, and `414b702` the parity foundation. Do not recreate or amend these existing commits.
- Live `git status --short` contains only modified `.gitignore`, untracked `.omp/`, and untracked `AGENTS.md`; preserve these pre-existing changes and exclude them from slice commits.
- `tests/parity/run_parity.nim:collectReport` executes CPU cases and calls `collectAppleCoverage`; its Apple counters explicitly start with zero native executions and all selected requirements unobserved. `reportExitCode` rejects CPU mismatches/errors and missing required Apple observations.
- `arraymancer.nimble:parity` builds and runs `tests/parity/run_parity.nim` with all profiles and writes `build/pytorch-parity.json`; CPU-only checks must explicitly select `--profile=cpu`.
- `/opt/homebrew/bin/nim`, `/opt/homebrew/bin/nimble`, and `/opt/homebrew/bin/shellcheck` resolve. No build, test, installation, or commit has run in this planning session.
- Saved `build/pytorch-parity-cpu.json` reports 9 passes, 0 mismatches, and 0 errors; saved `build/pytorch-parity.json` reports 18 missing/unobserved Apple requirements. These are existing reports, not fresh execution proof.
- Read the immutable PyTorch dtype switch at `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457:aten/src/ATen/native/mps/OperationUtils.mm:52–90`: float32, float16, bfloat16, signed/unsigned 8/16/32/64-bit integers, bool, complex-float16 and complex-float32 are mapped; float64 tensors explicitly error. This mapping does not establish operation support.
- `src/arraymancer/tensor/data_structure.nim` currently builds `AnyTensor` from CPU/CUDA/OpenCL, and `GpuTensor` raw pointer access assumes `storage.Fdata`. A Metal buffer must not enter that unsafe-pointer path.
- `src/arraymancer/tensor.nim` uses explicit conditional backend exports; `src/arraymancer.nim` exports it through the public root. Preserve CPU-only imports without Apple toolchains or libraries.
- Primary API evidence: [Core AI encode/async outputs](https://developer.apple.com/documentation/coreai/inferencefunction/encode(inputs:states:outputviews:to:)), [ComputeStream boundaries](https://developer.apple.com/documentation/coreai/computestream), [converter API at the inspected source revision](https://github.com/apple/coreai-torch/blob/31908ce231d0794dd966f9fae610da5f95041288/docs/api/TorchConverter.md), and [released converter metadata](https://pypi.org/pypi/coreai-torch/0.4.3/json). Released conversion compatibility still requires execution proof.

## Approach

### Execution and commit discipline

Work sequentially: implement one numbered slice, run its focused gate, and commit it before editing the next slice. Preserve pre-existing `.gitignore`, `.omp/`, and `AGENTS.md` changes; never stage unrelated work, rewrite previous commits, or fabricate an empty commit for the already completed CPU slices. Use only the eval `commit` namespace: `d = await commit.draft()`, inspect `await d.diff()`, drop any unrelated units, then `await d.commit(subject, ["- Verified: <actual focused check>"])`. Do not use `git add`, `git commit`, stash/reset/restore, or undocumented amend/multiple-glob options.

Before the first source edit, run the existing CPU parity suite and CPU report as the baseline. If both pass, no CPU implementation commit is needed. Treat a newly observed failure as its own correction slice: repair the failing existing contract, add a regression only if uncovered, verify that path, and commit before continuing.

### Native boundary and build contract

Use an Objective-C++ Metal bridge and a separate Swift Core AI bridge, both exposing a typed C ABI. Arraymancer remains Nim; these platform adapters call Apple frameworks directly. No Python, PyTorch, LibTorch, helper process, JSON serialization, or host-side substitute computation is permitted in library execution.

Add optional `-d:metal` and `-d:coreai`; guard Metal imports/exports with `defined(metal) or defined(coreai)` so Core AI includes Metal without relying on define ordering. Reject either define outside macOS arm64 with a compile-time diagnostic. Keep ordinary CPU imports and tests independent of either bridge, the SDK, Python, and GPU discovery. Add explicit conditional exports to existing umbrellas, not an unconditional Apple import.

Create a stdlib-only `tools/build_apple.nim` developer builder; no equivalent Apple builder exists in the inspected source. Discover the selected SDK/compiler with direct argument arrays, never change `xcode-select`, and write only the selected output directory, default `build/apple/`. Build `libarraymancer_metal.dylib` from Objective-C++ with ARC, Metal and Foundation, and `libarraymancer_coreai.dylib` from Swift 6 mode with Core AI; target arm64/macOS 27.0. Add `backend/native/module.modulemap` for the shared `apple_abi.h` import and Swift `@c @implementation` exports. All handles are retained native objects with explicit retain/release operations; fixed-width status plus owned message crosses C, never an exception or Nim-managed closure from a native completion thread.

The public Nim bridge modules lazily load these dylibs only on an explicit backend request. Resolve `arraymancerAppleLibDir` as a string define, defaulting to the owning package's `build/apple`; missing libraries or framework/device availability raise explicit backend errors. Core AI bridge resolves its Metal dependency with `@loader_path`. The builder exposes `--backend=metal|coreai|all`, `--output=PATH`, `--help`; no arguments print help and build the implemented backends. Slice 1 initially builds Metal only; Slice 12 makes the default `all`, requiring both bridges. Never reference an undelivered source file or add a placeholder bridge.

Runtime tensor/NDArray descriptors pass rank, signed element strides, signed element offset, dtype and bounded storage capacity as typed fields; bytes are checked for overflow before native access. An empty shape has one scalar element; a zero dimension has zero elements; a default nil-owner tensor remains an empty sentinel. Validation and native ownership must remain correct with Nim bounds checks disabled.

### Reference data and observation rules

Leave `fixtures.json`, `generate_fixtures.py` and the nine float64/CPU cases unchanged. Add a separate `tests/parity/metal_fixtures.json` and developer-only `references/generate_metal_fixtures.py`, using the same Torch/Python pins and actual `torch.mps` execution with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Generate real float32 view/broadcast/matmul/mean/MSE/SGD observations; record the actual oracle device/version/build identity. Refuse missing MPS or wrong versions without replacing checked-in data. This reference process is never native Arraymancer execution evidence.

The new fixture structure is `{format: 1, reference, oracle, cases}`; each case has `id`, `inputs`, `expected`, `atol`, `rtol`, following the CPU generator convention. Use requirement subcase IDs `tensor.transpose_alias`, `tensor.broadcast_same_rank`, `tensor.broadcast_leading_dims`, `tensor.matmul_2d`, `autograd.mean_gradient`, `loss.mse_mean`, and `optim.sgd_step`; float32 math tolerance is `atol=1e-6, rtol=1e-6`. Preserve shape and dtype exactly. At comparison, map only the reference backend label `mps` to native label `metal`; validate actual Metal placement separately from native completion evidence. Do not normalize actual shapes, cast actual values, or overwrite actual device labels.

Dynamic device/memory/queue/policy facts are predicates over real native measurements, not fabricated golden values. Retain measured counters/identities/trace facts in `evidence`; compare stable observations through `compareObservations`. Normalize native error categories to `unavailable`, `dtype`, `shape`, `bounds`, `device`, `invalid_argument`, `rng_state`, `out_of_memory`, `shader_compile`, `library_load`, `missing_function`, `model_load`, `descriptor`, `buffer_busy`, `timeout`, or `native_error`, retaining original diagnostics separately. A requirement passes only after every prescribed subcheck runs and passes; a missing subcheck cannot be omitted into success.

As each requirement becomes implemented, change its capability annotation from `missing` to `partial`, cite the actual new public/backend symbols, and remove only gaps the slice closes. Keep annotations separate from execution; no `verified` or `pass` annotations. The API-area inventory remains a source catalogue, not a completion percentage.

### Slice 1 — Native Metal storage and explicit dispatch

Create `src/arraymancer/tensor/backend/native/apple_abi.h`, `metal_bridge.mm`, the Nim `backend/metal_abi.nim` loader, `tensor/metal_types.nim`, `tensor/init_metal.nim`, and `tensor/tensor_metal.nim`. These are new because existing backends provide CUDA/OpenCL bindings only. Add `MetalTensor[T]` with existing `Metadata` shape/strides/offset plus a shared native buffer owner; declare `MetalDevice`, `MetalStream`, `MetalError`, `MetalUnavailableError`, and `MetalDtypeError`. No Metal buffer is exposed as a CPU pointer.

Public operations: `metalIsAvailable(): bool`, `metalDeviceCount(): int`, `metalDevice(index = 0): MetalDevice`, `newMetalTensorUninit[T: KnownSupportsCopyMem](shape: Metadata; device: MetalDevice = nil): MetalTensor[T]` plus the existing-style `varargs[int]` shape overload, `metal[T: KnownSupportsCopyMem](t: Tensor[T]; device: MetalDevice = nil): MetalTensor[T]`, and `cpu[T](t: MetalTensor[T]): Tensor[T]`. Start with float32 enabled; unsupported types raise `MetalDtypeError` before allocation. Transfers preserve dtype and values, never change the original CPU tensor, and complete GPU writes before host reads.

Wire conditional tensor exports and `AnyTensor` membership without admitting Metal to the existing `GpuTensor` `Fdata` raw-pointer accessors. Keep those accessors restricted to legacy CUDA/OpenCL tensors; implement Metal rank/size/contiguity metadata access directly, reusing `shapeProduct` and `shape_to_strides`. Native buffers retain queued work and survive metadata aliases; owners release exactly once under ORC and refc. Nil handles, allocation failures, invalid device index, negative/overflowing dimensions and excessive rank produce visible errors.

Track outstanding producers on the native buffer owner, not on one metadata alias. Host readback waits for that buffer's producing work, never unrelated queues; native commands/jobs retain the native owner independently of any Nim reference. This is essential for the later unrelated-queue and dropped-reference checks.

Add `tests/apple/test_metal_storage.nim`: float32 `[1,2,3,4,5,6]` CPU→Metal→CPU round trip, original input unchanged, explicit device identity, invalid device index, initialized scalar size 1, default sentinel size 0, and `[2,0,3]` size 0. An isolated child executable with `arraymancerAppleLibDir` pointing to a nonexistent build directory must make an explicit Metal request fail as unavailable while leaving the original CPU tensor unchanged. Native availability checks alone are not GPU execution claims. Build the Metal dylib and run these tests before commit `feat(metal): add native device storage and transfers`.

### Slice 2 — Strict real-observation gates

Replace the hardcoded Apple gap-only report with typed in-process observations. Add test-only `AppleObservation` and `AppleExecution = aePass | aeMismatch | aeError` in `parity_support.nim`; observations carry requirement ID/profile, actual JSON, `seq[Difference]`, error and native evidence. Change `collectAppleCoverage` to accept required `observations: openArray[AppleObservation]` and optional `requirementId = ""` after its current arguments. Its exhaustive current callers are `run_parity.nim:105` and `test_parity.nim:78,79,82,95`; migrate all, with no three-argument compatibility overload. Nim reference lookup returned no references despite these callers, so use exact `collectAppleCoverage` text search if the Nim service remains unavailable.

An entry retains `id`, `profile`, `implementation_status`, `execution`, `actual`, `reason` and gains `differences`, `error`, `evidence`. `execution` is `pass`, `mismatch`, `error`, or collector-produced `not_run`. Unknown/duplicate/profile-conflicting observations are invalid input. Missing observations remain `not_run` with null actual; a capability annotation can never provide an observation. Accept no external observation-file/JSON option. Native adapters call public Arraymancer APIs; no result may echo an expected fixture.

Native evidence has a completed invocation kind: device/storage control, GPU command, or Core AI inference. The bridge records actual invocation IDs and completion, device registry ID, dispatched kernel labels, transfer/materialization byte counts and CPU-fallback count. Slice 1's device/storage control is genuine native evidence without claiming a GPU kernel ran; math/training/dtype/kernel requirements require completed GPU commands. Core AI evidence records the completed invocation, actual asset digest, function and stream, requested policy and observed placement or explicit unknown. The adapters cannot supply expected literals as execution evidence; Core AI CPU-only execution is legitimate Core AI evidence.

Update `collectReport` to execute only selected, compiled-in native requirement handlers and then merge observations. New test-only `tests/parity/native_metal_cases.nim` initially implements only `apple.metal.device_dispatch` using Slice 1; add complete handlers in subsequent slices, never placeholder handlers. Disabled/unimplemented native cases contribute no observations; backend runtime errors contribute error observations. Keep CPU selection from loading either native dylib.

Bump report `format` to 2 and scope to `native_apple_conformance`; keep reference/artifact formats at 1 and the existing no-full-parity notice. Add Apple `pass`, `mismatch`, `error` counters, compute `native_executed` from entries with completed native invocation evidence, and compute `unobserved` from `not_run`. Gate `pass` requires every selected requirement to pass; any mismatch/error/not-run yields exit 1. Preserve exit 2 for malformed artifacts/selectors/output I/O and exit 0 for a wholly passing selection.

Add `Selection.requirementId` and CLI `--requirement=ID`, valid only with an explicit single `--profile=apple-metal` or `--profile=apple-coreai`, mutually exclusive with `--case`. Validate ID/profile before execution. This enables a complete single-contract slice gate without dropping requirements from the default `all` run. Existing `--case` remains CPU-only.

Replace absence-pinning checks in `test_parity.nim` with explicit empty-observation policy tests and isolated pass/mismatch/error aggregation tests; retain annotation-forgery, missing profile/requirement/link, complete fixture and selector validation. Verify `--profile=apple-metal --requirement=apple.metal.device_dispatch` with `-d:metal` actually calls the native library and passes, while the default all-profile report remains incomplete for undelivered requirements. Commit `feat(parity): execute strict native Apple observations`.

### Slice 3 — Every pinned Metal storage dtype

Extend the Metal type map, transfers and a GPU-to-distinct-GPU-buffer identity kernel to all 14 pinned dtypes. Exact mapping: Nim `float32`→`float32`; new `MetalFloat16 = distinct uint16`→`float16`; new `MetalBFloat16 = distinct uint16`→`bfloat16`; Nim `int8/int16/int32/int64`, `uint8/uint16/uint32/uint64` and `bool`→their same-named PyTorch dtypes; new `MetalComplexFloat16` with two `MetalFloat16` fields→`complex32`; existing Nim `Complex32` (two float32 components)→PyTorch `complex64`. Nim `float64` and `Complex64` are rejected; never confuse Nim's complex type names with PyTorch's total-bit-width labels.

Keep the new low-precision wrappers in `metal_types.nim`. Add `fromBits(typ: typedesc[MetalFloat16]; bits: uint16): MetalFloat16`, the corresponding `MetalBFloat16` overload, and `toBits(value: MetalFloat16|MetalBFloat16): uint16`; complex components use these explicit bit conversions. No host arithmetic API or dependency is needed.

Add `backend/native/metal_kernels.metal` with a bit-preserving typed copy kernel. Extend the builder here to run `xcrun metal`/`metallib` into co-located `arraymancer.metallib`. Check storage widths, bool 0/1, and distinct input/output buffers. This proves copy/representation, not math/training support. Float32 remains the arithmetic baseline; other arithmetic dispatches visibly reject unsupported dtype/operator combinations.

Add `tests/apple/test_metal_dtypes.nim` with signed minima/negative values, high unsigned bits, both complex components, low-precision bit patterns, scalar and empty representations, and float64/complex128 rejection before allocation. Compare native readback bytes exactly; serialize wide integers and low-precision/complex representations as canonical fixed-width hexadecimal bit strings in dtype observations, never lossy JSON float64 numbers. Add the real `apple.metal.dtypes` adapter only after every mapped representation executes successfully. Commit `feat(metal): support pinned tensor storage dtypes`.

### Slice 4 — Real metadata aliases and strided mutation

Add `tensor/shapeshifting_metal.nim` and `tensor/accessors_metal.nim`. `transpose(t: MetalTensor[T]): MetalTensor[T]` and `metalView(t: MetalTensor[T]; shape, strides: Metadata; offset: int): MetalTensor[T]` share the original storage owner and never copy it. `offset` is absolute in that owner's element storage. Add `setElement(t: var MetalTensor[T]; indices: openArray[int]; value: T)` and 1-D/2-D `[]=` overloads forwarding to it; assignment runs a native GPU kernel, not a host-buffer replacement.

Support signed strides and offset views in transfer/copy kernels. Validate bounds by checked minimum/maximum reachable storage index, not `offset + size`; zero-sized views perform no memory access. Read-only broadcast zero strides are allowed. Mutation accepts layouts proven nonoverlapping by sorted absolute-stride/span checks and rejects broadcast/ambiguous overlapping destinations; this admits the required transpose, offset and reverse views without enumerating tensor elements on the CPU. `clone` makes an independent GPU buffer. Match existing `asContiguous(t; layout = rowMajor; force = false)` naming and behavior: preserve an already C/F-contiguous alias unless ordering must change; otherwise materialize the requested row/column-major layout on the GPU and report it.

Add `tests/apple/test_metal_views.nim`: transpose mutation changes base to `[1,2,3,9,5,6]` and view to `[1,9,2,5,3,6]`; offset view and reversed negative-stride view preserve values and owner identity; `clone` breaks aliasing; empty views stay empty; invalid bounds and broadcast mutation error. Implement `apple.metal.tensor_views` with transfer/materialization evidence. Commit `feat(metal): preserve strided tensor view semantics`.

### Slice 5 — Float32 tensor arithmetic on the GPU

Add `tensor/operators_metal.nim` and `tensor/aggregate_metal.nim`, extending Slice 3's builtin `backend/native/metal_kernels.metal`. Implement float32 `+.`, `-.`, `*.`, `/.`, tensor/scalar arithmetic, tensor `*` as 2-D matmul, in-place add/scaled subtraction/fill, and GPU-resident scalar `sum`/`mean`. These kernels are new because existing compiled kernels target CPU/CUDA/OpenCL, not Metal. Unsupported dtype/rank combinations error rather than invoking CPU routines.

Broadcast by trailing-axis shape alignment and zero-stride input views, producing one packed output; reject incompatible dimensions before dispatch. Kernels consume signed strides/offsets so transpose, negative-stride, offset and broadcast inputs do not need host packing. Use a 16×16 threadgroup tiled float32 matmul with masked edge loads/stores, and deterministic tree reductions with GPU partial buffers only when more than one threadgroup is needed. Cache compiled pipelines per device; never compile a pipeline or allocate an input copy on every arithmetic call.

`sum(t: MetalTensor[float32]): MetalTensor[float32]` and `mean(t: MetalTensor[float32]): MetalTensor[float32]` return shape `[]` on the same device. Add `sum(t; axis: int)` retaining the reduced dimension at size 1, GPU `zeros_like`/float32 `ones_like`, unary negation, and `+=` needed by the existing autograd tape. Empty sum is scalar zero; empty mean is scalar NaN. Skip zero-work dispatches for empty elementwise outputs. In-place operations may reuse identical-index inputs, but must reject overlapping destinations or materialize a conflicting differently-strided read alias on the GPU before writing.

Use Slice 3's mandatory Metal compilation, failing on missing Metal Toolchain or shader errors. Create the separate real MPS reference fixture/generator here. Validate every required subcase, finite nonnegative tolerances, and dtype/shape/value counts without weakening CPU validation.

Add `tests/apple/test_metal_math.nim`: both broadcast examples and matmul have the values in `apple_requirements.json`, plus non-contiguous inputs, non-tile-multiple dimensions, scalar/empty results, incompatible shapes and input buffers unchanged. Implement `apple.metal.tensor_math` and assert actual dtype/device plus completed GPU kernels. Commit `feat(metal): execute float32 tensor math natively`.

### Slice 6 — Metal mean and mean-MSE autograd

Reuse `Context[TT]`, `Variable[TT]`, `Gate[TT]`, `backprop` and exception-safe `no_grad_mode`; do not introduce a second tape. Extend `shapeshifting_metal.nim` with `transpose(t, dim1, dim2)`, last-two-axis `transpose2d`, `reshape` with Metadata/varargs shapes, leading-axis `broadcast`, and `squeeze(axis)`. Reuse `shape_to_strides`, `reshape_no_copy` and `broadcast2Impl` only for metadata, explicitly preserving the Metal owner. Metal reshape checks use checked scalar-aware `shapeProduct`; do not widen CPU-only raw storage helpers or change legacy CUDA/OpenCL products. A noncontiguous reshape materializes once on the GPU; Metal `squeeze` may produce rank zero, unlike the existing legacy minimum-rank-one helper.

In `autograd/gates_reduce.nim`, keep CPU scalar wrapping unchanged and add native Metal result handling so mean/sum graph values remain Metal shape `[]`. Normalize full mean gradients with scalar-aware cached-shape size, not empty `Metadata.product`; use GPU tensor/scalar division and broadcast views. Add a Metal overload of `autograd/private/p_broadcast.nim:reduce_broadcast_dims` using axis sum/Metal squeeze; leave the existing CPU overload contract intact. In `gates_hadamard.nim`, unbroadcast both gradients for Metal through this overload. Existing generic basic/matmul gates then use the native operations; CPU-specific `ScaleGate`, unrelated activations and optimizers are not broadened.

Create `nn_primitives/nnp_mse_metal.nim` with `mse_forward_metal(prediction, target)` and `mse_backward_metal(upstream, prediction, target)`, all taking/returning `MetalTensor[float32]`. Forward returns device scalar `mean((prediction-target)^2)`; backward reads the scalar upstream gradient on the GPU and writes `upstream * 2 * (prediction-target) / numel` to one allocated gradient. Enforce identical input/target shapes and scalar upstream metadata. Empty forward is NaN and backward is empty without reading/dividing nonexistent elements. Import these helpers conditionally in `nn/loss/mean_square_error_loss.nim`; retain the existing `MSELoss` gate/cache/public signature, using private backend overloads to keep CPU raw reads/maps off the Metal path. Do not copy a Metal loss or gradient to the host during graph execution.

Add `tests/apple/test_metal_autograd.nim`: mean of `[[1,2],[3,4]]` is scalar 2.5 with four gradients 0.25; mean-MSE of `[[1,2,3],[4,5,6]]` against zero is scalar `91/6`, gradients `[1/3,2/3,1,4/3,5/3,2]`. Verify all values/gradients/owners remain Metal before explicit assertion readback. Cover scalar mean derivative 1, empty mean metadata, a leading-axis broadcast gradient, and no-grad restoration after an exception. Compare the required observations to the real float32 MPS fixtures. Implement `apple.metal.autograd`, run the affected existing CPU gate tests and CPU parity suite, then commit `feat(autograd): differentiate Metal mean and MSE natively`.

### Slice 7 — Native SGD with separate gradient reset

Extend `nn/optimizers/optimizers.nim` with explicit-backend `optimizer(model; OptimizerKind: typedesc[SGD]; learning_rate: TT.T; tensorType: typedesc[TT]): SGD[TT]`, supporting homogeneous `MetalTensor[float32]` models. Reuse one private typed recursive parameter collector for the existing three-argument CPU SGD factory and this overload; keep CPU call syntax/return types, momentum/Adam and deprecated constructors unchanged. Reject model variables of a different backend/dtype rather than silently omitting them.

Add `step(self: SGD[MetalTensor[float32]])` to update trainable weights in place with one native scaled-subtract kernel per parameter, preserving gradients and storage identity. Add `zeroGrads(self: SGD[MetalTensor[float32]])` to zero existing gradient buffers in place without allocating or rebinding them. Validate device/shape/mutable-layout before each dispatch. Provide the Metal `update` overload as the existing compound operation `step` followed by `zeroGrads`; retain existing CPU `update` behavior. The parity scenario calls `step` and `zeroGrads` separately; never pretend legacy `update` has PyTorch step semantics.

Add `tests/apple/test_metal_optimizer.nim`: native trainable weights `[1,2]`, input `[3,4]`, prediction `(weights *. input).sum`, scalar target 0 and learning rate 0.01 produce prediction 11, scalar mean-MSE 121, gradients `[66,88]`, then weights `[0.34,1.12]`. Gradients remain `[66,88]` after `step`; only explicit `zeroGrads` makes them zero, with unchanged buffer identity. All intermediate graph/gradient/update tensors remain Metal and the arithmetic has completed native GPU evidence. Verify frozen parameters remain unchanged and legacy CPU SGD still updates numeric weights and clears gradients through its existing factory; existing optimizer tests cover momentum, not this SGD contract. Compare the required trace to MPS `optim.sgd_step`, implement `apple.metal.optimizer_sgd`, and commit `feat(optim): add native Metal SGD and explicit zeroing`.

### Slice 8 — Streams, events, and pending-work lifetime

Create `tensor/metal_runtime.nim` and wire its conditional tensor exports. Expose `newMetalStream(device: MetalDevice = nil): MetalStream`, `defaultMetalStream(device = nil)`, `currentMetalStream(device = nil)`, `setMetalStream(stream)`, and exception-safe `withMetalStream(stream): body`. Selection is thread-local and retains its stream; reject cross-device operations. Add `MetalEvent`, `newMetalEvent(device = nil)`, `record(event, stream)`, and `wait(stream, event)` using an `MTLSharedEvent` monotonic timeline. The wait captures the recorded value. Also expose `wait(stream, event; value: uint64)` for a future-value dependency and `signalMetalEvent(event; value: uint64)` for an explicit native host signal; reject nonmonotonic signals. These real events provide deterministic pending-work tests, not sleep-based timing.

Add `metalSynchronize(stream: MetalStream)` for that queue's previous work and `metalSynchronize(device: MetalDevice = nil)` for previous work across all registered queues. Keep the single default queue as the fast path, but do not present it as all-stream coverage. Host readback waits for producing work. Cross-stream data dependencies use explicit record/wait events; the API does not promise implicit ordering of independent queues.

Retain buffers, library pipelines, streams and event values through command completion. Native completion handlers may release native leases/update native counters, but never enter Nim GC. A released tensor's allocation cannot enter the reusable cache while any queued command still uses it.

Add `tests/apple/test_metal_streams.nim`: producer `[1,2]+1` on one queue, event handoff, consumer `*2` on another produces `[4,6]`; drop creating references before completion while an alias survives; test current/set/default stream identities and restoration after an exception. Implement every subcheck of `apple.metal.queue_lifetime`. Commit `feat(metal): add streams events and safe queued lifetimes`.

### Slice 9 — Real allocation accounting and profiling

Extend `metal_runtime.nim` with `metalCurrentAllocatedMemory(device = nil): uint64`, `metalDriverAllocatedMemory(device = nil): uint64`, `metalRecommendedMaxMemory(device = nil): uint64`, `metalEmptyCache(device = nil)`, `metalMemoryFraction(device = nil): float64`, and `setMetalMemoryFraction(device: MetalDevice; fraction: float64)`. Track owned live tensor allocations once per shared storage, cached bytes separately, and in-flight leases separately. Read driver bytes from `MTLDevice.currentAllocatedSize` and the recommended limit from `recommendedMaxWorkingSetSize`; do not equate either with tensor payload bytes.

Use a size-binned native buffer cache. Cache only completed, unowned allocations; `metalEmptyCache` actually releases unoccupied cached buffers and never invalidates a live alias or pending command. Enforce the requested process limit before allocation; finite fractions are in `[0,2]`, `0` means unlimited, and the limit is `fraction * recommendedMaxWorkingSetSize`, matching the pinned MPS contract. Allocation-limit failure is a visible out-of-memory error.

Expose `startMetalCapture(device: MetalDevice; outputPath: string): MetalCapture` and `stopMetalCapture(capture)` through `MTLCaptureManager`; unsupported capture destinations and capture errors are visible. Record real dispatched kernels/completion, host copies/materializations, and any fallback attempts, not an availability-based GPU claim.

Add `tests/apple/test_metal_memory.nim`: allocation increases live bytes; metadata aliasing does not; dropping references then synchronizing/collecting restores live baseline; cache clear releases cached bytes; pending work survives cache clear; invalid fractions and a deliberately tiny limit error; restore the previous fraction in `defer`. Capture one real arithmetic dispatch into `build/apple/` and retain measured driver/recommended/copy facts without equality assumptions. Implement `apple.metal.memory_profiling`. Commit `feat(metal): report allocations limits and GPU profiling`.

### Slice 10 — Native replayable random generation

Create `tensor/random_metal.nim` and its conditional export. Add `MetalGenerator`, `MetalRngState`, `newMetalGenerator(seed: uint64; device: MetalDevice = nil)`, `defaultMetalGenerator(device = nil)`, `manualSeed(generator, seed)`, `seed(generator): uint64`, `getRngState(generator): MetalRngState`, `setRngState(generator, state)`, and `randMetal(shape: Metadata; generator: MetalGenerator): MetalTensor[float32]` plus its varargs shape overload. `seed` reseeds from OS entropy and returns the new seed; `manualSeed` resets the counter to zero for the supplied seed. Default seeding uses OS entropy; explicit seeding/state restore remain deterministic.

Implement Philox4x32-10 in a Metal kernel, not CPU draws followed by upload. State version 1 contains the 64-bit seed and 128-bit next-block counter; reserve `ceil(numel/4)` blocks atomically before enqueue, rejecting counter overflow and incompatible state versions. Use multipliers `0xD2511F53`, `0xCD9E8D57`, key increments `0x9E3779B9`, `0xBB67AE85`; derive float32 uniform `[0,1)` from each word's top 24 bits times `2^-24`. State snapshot/restore ordering is protected independently of queue completion.

Add `tests/apple/test_metal_rng.nim`: save seeded state, draw to one buffer, restore, draw to a distinct buffer and compare exact values/Metal placement; pending draws and state reads have the documented reservation order; invalid state/version errors; scalar and empty shapes preserve size/counter semantics. Do not claim Torch/CPU/GPU bitwise RNG equivalence. Implement `apple.metal.rng`. Commit `feat(metal): add replayable GPU random generation`.

### Slice 11 — Public shader compilation and metallib execution

Create `tensor/custom_kernels_metal.nim` and its conditional export. Add `MetalLibrary`, `MetalKernel`, `MetalBufferAccess = mbaRead | mbaWrite | mbaReadWrite`, and `MetalBufferBinding`. Expose `compileMetalShader(device: MetalDevice; source: string): MetalLibrary`, `loadMetalLibrary(device: MetalDevice; path: string): MetalLibrary`, `metalKernel(library: MetalLibrary; name: string): MetalKernel`, `metalBinding[T](t: MetalTensor[T]; access: MetalBufferAccess): MetalBufferBinding`, and `dispatchMetalKernel(kernel; buffers: openArray[MetalBufferBinding]; grid, group: array[3,int]; stream: MetalStream = nil)`.

These are direct Metal compilation/load/pipeline/dispatch operations. Cache pipelines by library/function/device. Bindings retain storage and honor byte offsets; reject non-contiguous custom bindings, mismatched devices, invalid dimensions and conflicting mutable leases. User-authored shader indexing/bounds remain the caller's responsibility; built-in kernels do not use this unchecked source path. Return compiler/load/function errors with original diagnostic detail.

Add checked-in `tests/apple/kernels/double.metal` and `tests/apple/test_metal_kernels.nim`. Both runtime source compilation and separately built metallib execution map float32 `[1,2,3,4]` to `[2,4,6,8]`, with distinct output storage and completed native GPU evidence. Malformed source, corrupt metallib and missing function yield native errors. Implement `apple.metal.custom_kernels`; now the full Metal profile must have ten real passes. Commit `feat(metal): compile and dispatch custom GPU kernels`.

### Slice 12 — Authentic Core AI models, loading and descriptors

Add developer-only `tests/parity/references/generate_coreai_models.py`. Pin CPython 3.13.18, Torch 2.15.0, `coreai-torch==0.4.3` and `coreai-core==1.0.0b3` in a separate `build/coreai-oracle` environment, leaving the existing CPU oracle untouched. Their version metadata permits this combination but conversion compatibility is unverified: conversion plus real SDK loading is the proof, not dependency resolution.

Generate and check in the authentic asset `tests/apple/models/parity.aimodel` with explicit entrypoints `identity` (float32 `[2]`) and `identity_matrix` (float32 `[rows,3]`, `torch.export.Dim("rows", min=1, max=8)`). Both return `x.clone()` with input `x` and output `y`. Run `torch.export.export(...).run_decompositions(get_decomp_table())`, add each program with `TorchConverter.add_exported_program(..., input_names=["x"], output_names=["y"], entrypoint_name=...)`, then `to_coreai().save_asset(path)`. Check in `tests/apple/models/manifest.json` containing actual versions, graph signatures, source-model definitions and per-file SHA-256 hashes. Ordinary native checks consume this real source asset offline; Python is developer-only regeneration, not a runtime/test dependency. Do not fabricate model bytes or copy expected output into an asset.

Read-only probes verified `xcrun --find coreai-build` and `xcrun --find metal`: both resolve inside the installed Metal Toolchain cryptex. The builder uses the help-verified command `xcrun coreai-build compile tests/apple/models/parity.aimodel --platform macOS --min-deployment-version 27.0 --output build/apple/models/compiled`. It emits architecture-suffixed `.aimodelc`; select the one matching `AIModel.deviceArchitectureName`, never the first glob match. Keep architecture-specific AOT products under ignored `build/`. `aimodelc` is a different installed executable and is not substituted.

Create `src/arraymancer/coreai.nim`, `coreai/runtime.nim`, and `coreai/native/coreai_bridge.swift`; extend the shared C header and builder for the Swift dylib. Export Core AI conditionally from `src/arraymancer.nim`. Introduce `CoreAIModel`, `CoreAIFunction`, `CoreAIArray`, `CoreAIBinding` (name plus array), `CoreAIJob`, `CoreAIError`, and `CoreAIDescriptorError`. `CoreAIArray` owns an SDK NDArray or retains a borrowed Metal-buffer owner plus metadata; it never persists a nonescaping Swift raw view.

Public surface: `openCoreAIModel(path: string; policy: CoreAIComputePolicy = cpDefault): CoreAIModel`, `functionNames(model): seq[string]`, `loadFunction(model; name: string): Option[CoreAIFunction]`, `descriptor(function): CoreAIFunctionDescriptor`, `toCoreAI[T: float32|int32](t: Tensor[T]): CoreAIArray`, `cpu[T: float32|int32](array: CoreAIArray; typ: typedesc[T]): Tensor[T]`, `submitInference(function; inputs: openArray[CoreAIBinding]): CoreAIJob`, `wait(job: CoreAIJob; timeoutMs = 30000): seq[CoreAIBinding]`, and `infer(function; inputs): seq[CoreAIBinding]` as submit-plus-wait. Declare and implement all four compute-policy mappings here; Slice 17 adds their complete acceptance observations. Host import copies logical data once before async return; host output is copied only after completion, without dtype conversion.

`CoreAIFunctionDescriptor` exposes named input/output/state array descriptors; each array descriptor contains exact dtype, shape (SDK `-1` for dynamic dimensions), and dynamic-shape flag. Do not query preferred strides/minimum bytes until concrete dimensions are resolved. Validate names, dtype, rank and static dimensions before entering unsafe SDK view APIs. Missing function returns `none` from SDK nil; invalid model/load failure throws `CoreAIError`, and wrong dtype/shape throws `CoreAIDescriptorError`. Normalize categories in the parity adapter while retaining original diagnostics.

Model loading and inference start a retained native job. A non-MainActor Swift root `Task` owns the model/function/arrays through completion. Publish result/error and signal the native waiter only after the await; no lock/semaphore is held across await, and no native thread calls Nim GC. Wait timeout reports a timeout but retains pending resources until completion. Release job/task cycles at completion. C ABI uses fixed-width statuses and copied messages, never borrowed Swift strings or captured caller host pointers.

Add `tests/apple/test_coreai_models.nim` and real `native_coreai_cases.nim` handlers for `apple.coreai.model_functions` and `apple.coreai.ndarray_descriptors`. Load both `.aimodel` and the matching `.aimodelc`; infer `[1.25,-2]` exactly, inspect actual names/descriptors, reject int32 and wrong shape/rank without conversion, distinguish absent function from invalid model, and verify the actual asset hashes. Commit only after real Nim→Swift→Core AI inference passes: `feat(coreai): load real models and validate inference descriptors`.

### Slice 13 — Strided NDArrays and shared Metal-buffer interop

Add `coreAIView[T](tensor: MetalTensor[T]): CoreAIArray` for supported Core AI dtypes/layouts. Retain the same native Metal owner and `MTLBuffer`, not just its host pointer. Within the job's Swift borrowing scope construct `NDArray.RawView(metalBuffer:byteOffset:scalarType:shape:strides:interleaveLayout:)` and insert that raw view into `InferenceFunction.Inputs`. Validate `.shared` storage, capacity, byte offset, scalar width, strides and shape before the unsafe initializer; unsupported layout/storage is an explicit error, never an implicit copy.

Add `resolveDescriptor(descriptor; shape: Metadata): CoreAIArrayDescriptor`: verify rank/static dimensions, call SDK `resolvingDynamicDimensions`, then expose actual `preferredStrides` and `minimumByteCount`. Preserve SDK preferred layout instead of assuming contiguous. Newly allocated returned NDArrays must report their actual row-major strides.

Acquire native read/write leases for borrowed inputs/state/outputs before returning the job, then asynchronously await those buffers' preceding producers before constructing SDK views. Normal Arraymancer mutation cannot race an active borrow; reject conflicting writes with `CoreAIError` category `buffer_busy`. Raw views do not escape into stored Swift fields or outlive owners. Await only the relevant producer completions, not a device-wide fence.

Add `tests/apple/test_coreai_layout.nim`. Compare contiguous matrix identity with storage `[99,1,4,2,5,3,6,99]`, shape `[2,3]`, strides `[1,2]`, element offset 1 / byte offset 4. Both produce logical `[1,2,3,4,5,6]`; both guards remain 99. Resolve the dynamic row dimension before layout queries and test another valid row count. Record identical borrowed-buffer identity, bridge copy counts and actual output layout; report SDK-internal copies as unknown unless observed, not “end-to-end zero-copy.” Cover invalid bounds/layout and competing writes. Implement `apple.coreai.strided_layout` and `apple.coreai.metal_buffer_interop`. Commit `feat(coreai): preserve strided shared-buffer inference`.

### Slice 14 — One-queue producer, inference and consumer

Add `CoreAIStream`, `newCoreAIStream(stream: MetalStream): CoreAIStream`, `encodeInference(function; inputs: openArray[CoreAIBinding]; outputViews: openArray[CoreAIBinding] = []; stream: CoreAIStream): CoreAIJob`, and `awaitCurrentWorkCompleted(stream: CoreAIStream; timeoutMs = 30000)`. Swift wraps the retained same `MTLCommandQueue` with `ComputeStream(commandQueue:)` and uses SDK `InferenceFunction.encode`, `AsyncValue(unsafeBuffer:...)`, `AsyncMutableValue(unsafeBuffer:...)` and consuming `AsyncMutableViews`. Preallocated output support is required now so the subsequent Metal consumer already owns its output buffer; Slice 15 extends the ordinary run API and adds declared state.

Keep ordering correct even if preferred compute falls back to CPU/ANE: acquire ownership/leases before returning the job, asynchronously await a marker for preceding producer work on this queue, then encode inference and await its real completion in the native task. The Nim caller may wait for that job before queuing the consumer; this is a completion fence, not tensor readback/reupload. Do not claim an entirely host-wait-free pipeline. `awaitCurrentWorkCompleted` snapshots this stream's accepted jobs, awaits them and SDK `currentWorkCompleted`, then fences its same Metal queue so an already queued consumer is complete. Report SDK-stream and Metal-queue completion separately; never use device-wide synchronization to certify this boundary.

Add `tests/apple/test_coreai_streams.nim`: enqueue a Metal producer, encode identity into a preallocated shared output, wait for native inference without reading tensor data, enqueue a Metal consumer doubling that output, then await this stream and read the final result. Assert no intermediate host copy and retained owners. An unrelated queue waits on a fresh real event's future value 1; it must remain incomplete after the Core AI stream wait, then complete only after `signalMetalEvent(event, 1)`. Always release the event in `defer` so failure cannot strand a queue. No sleeps/timing race serves as proof. Implement `apple.coreai.stream_lifetime`. Commit `feat(coreai): order native inference on Metal streams`.

### Slice 15 — Declared mutable state and preallocated outputs

Extend the authentic generator with entrypoint `increment`: register a scalar float32 buffer `counter`, mutate `counter.add_(delta)` in `forward`, and return `counter.clone()` as `y`. Export a scalar float32 input `delta`, explicitly set `state_names=["counter"]`, `input_names=["delta"]`, `output_names=["y"]`, and verify the SDK descriptor actually declares exactly that scalar state. Regenerate/AOT-compile the model and manifest; never implement the increment in Nim or Swift.

Extend the existing APIs, not parallel wrappers: `submitInference(function; inputs: openArray[CoreAIBinding]; states, outputViews: openArray[CoreAIBinding] = []): CoreAIJob`, matching `infer`, and `encodeInference(function; inputs; states = []; outputViews = []; stream)`. Migrate Slice 12–14 callers explicitly. Construct SDK consuming `MutableViews`/`AsyncMutableViews` within their borrowing scopes. Require every declared state, reject duplicate/unknown state/output names and dtype/shape mismatch, and hold exclusive leases on mutable backing storage. Regenerate the checked-in source asset/manifest and its build-only AOT products.

Add `tests/apple/test_coreai_state.nim`: same scalar state initialized to 0 and `delta=1` gives state/output 1 then 2; missing `counter` errors. A caller-provided output view mutates in place, keeps its buffer identity, and `y` is absent from returned allocated outputs. Core AI function weights/intermediates are not used as caller state. Implement `apple.coreai.stateful_execution`. Commit `feat(coreai): execute model state and caller output views`.

### Slice 16 — Concurrent calls with isolated mutable storage

Use the SDK's Sendable `InferenceFunction` through one shared loaded function, with one native job per end-to-end call. Protect only job completion state and lease bookkeeping with a native lock; no global lock serializes inference and no actor/semaphore is held across await. Each job captures independent input/output owners; sharing immutable function resources does not share mutable call state.

Add `tests/apple/test_coreai_concurrency.nim`: submit identity `[1,2]` and `[3,4]` before waiting either; verify independent exact outputs and distinct writable buffers. Concurrent increment calls use separate zero states and both produce 1. For deterministic same-state rejection, place a future-value event wait on the producer queue, call `encodeInference` so its job owns state but awaits that real producer fence, then submit a second call with the same state and require `buffer_busy`. Signal the event in `defer`, wait for the first job, then reuse state successfully. Never infer overlap from a fast call's timing. Drop Nim creating references while jobs are pending. Implement `apple.coreai.concurrency`. Commit `feat(coreai): isolate concurrent inference state and lifetimes`.

### Slice 17 — Truthful compute policy and complete default gate

Expose `CoreAIComputePolicy = cpDefault | cpCpuOnly | cpGpuPreferred | cpNeuralEnginePreferred` (declare in Slice 12) and `availableCoreAIComputeUnits(): seq[CoreAIComputeUnit]`, with unit enum CPU/GPU/Neural Engine. Translate to SDK `.default`, `.cpuOnly`, and `SpecializationOptions(preferredComputeUnitKind: .gpu/.neuralEngine)` respectively. Record actual `ComputeUnitKind.availableKinds`, allowed units and preferred unit for every loaded policy.

The inspected SDK exposes no per-invocation selected/actual-unit property. Keep `actualPlacement` optional/unknown unless independent evidence proves it; no preference/availability/compiled architecture can set a GPU/ANE pass. CPU-only Core AI correctness counts as real Core AI inference, not Metal or ANE execution. The policy requirement passes for correct inference and truthful policy/placement reporting; hardware-specific subcoverage remains unverified without evidence.

Add `tests/apple/test_coreai_placement.nim`: load/run identity under all four policies; check exact output, CPU-only allowed set, and retained allowed/preferred/available facts. Emit native job signpost intervals with invocation ID, asset digest and function for trace correlation. Read-only `xcrun xctrace list templates` verified the installed `Core AI` template. Capture/export this one test process with the commands below, not all processes; correlate actual SDK compute-unit events to that exact invocation. Retain a separate trace evidence artifact. Never apply a traced probe's placement to a later untraced invocation; absent conclusive evidence remains unknown. Implement `apple.coreai.compute_placement`.

Wire Nimble `apple_build`, `test_metal`, and `test_coreai` tasks to the builder and new `tests/apple/test_metal.nim`/`test_coreai.nim` aggregates importing the respective focused modules. `test_metal` builds Metal only; `test_coreai` builds both bridges and AOT-compiles the checked-in authentic model, never running Python. On macOS arm64, strict `parity` builds both bridges/model and compiles the runner with `-d:coreai`; elsewhere it retains a strict all-profile report with Apple unavailable, not a green CPU-only substitution. Do not add Apple tests to `tests_cpu.nim`.

Run the final verification below before commit `feat(parity): gate complete native Metal and Core AI coverage`. Preserve every previous successful slice commit; no squash/amend is needed.



## Verification

Run from `/Users/derrick/git/ai/Arraymancer`. No commands below have run during planning. ShellCheck shell command text before executing it; direct argument-array subprocesses are preferred. Before executing either new Python generator, configure `pyright-langserver` with its pinned environment, open the exact current source version and receive successful diagnostics; no Python execution bypasses this prerequisite.

### Existing CPU baseline

```sh
mkdir -p build
nim c -r -d:ssl -o:build/test_parity tests/parity/test_parity.nim
nim c -d:ssl -o:build/pytorch_parity tests/parity/run_parity.nim
build/pytorch_parity --profile=cpu --output=build/pytorch-parity-cpu.json
```

Require nine CPU passes, zero mismatches/errors and Apple gate `not_selected`, plus the existing focused assertions. Keep the no-full-feature-parity notice. Install only missing declared Nim dependencies if needed. Use the host's Accelerate/default BLAS; do not copy historical Linux/CUDA/OpenBLAS paths from `nim.cfg`. No reimplementation/empty commit is needed for a passing baseline.

### Developer-only genuine reference generation

Use the existing CPU oracle only if its versions match; otherwise create `build/parity-oracle` with CPython 3.13.18 and Torch 2.15.0 before running the Metal generator. Do not change either pin. After successful Python LSP diagnostics:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 build/parity-oracle/bin/python tests/parity/references/generate_metal_fixtures.py --output tests/parity/metal_fixtures.json
uv venv --python 3.13.18 build/coreai-oracle
uv pip install --python build/coreai-oracle/bin/python 'torch==2.15.0' 'coreai-torch==0.4.3' 'coreai-core==1.0.0b3'
build/coreai-oracle/bin/python tests/parity/references/generate_coreai_models.py --output tests/apple/models
```

The Core AI generator validates installed versions and the declared graph signatures, then saves authentic assets and their manifest. It must succeed with the pinned versions and those assets must execute through the real SDK before Slice 12 can commit. No Torch/Python process participates in ordinary native tests or runtime execution.

### Each slice: build, scenario, strict requirement, commit

Compile the stdlib builder in Slice 1. Rebuild the changed native bridge/builtin metallib before that slice's focused test; never verify against a stale dylib. Starting in Slice 12 use `--backend=all`, which also performs source-asset AOT compilation.

```sh
nim c -o:build/build_apple tools/build_apple.nim
build/build_apple --backend=metal
nim c -r -d:ssl -d:metal --gc:orc -o:build/test_metal_storage tests/apple/test_metal_storage.nim
```

For each Metal slice substitute its exact test module from that slice; compile the focused requirement runner once after its adapter changes and require the selected contract to pass:

```sh
nim c -d:ssl -d:metal --gc:orc -o:build/pytorch_parity_metal tests/parity/run_parity.nim
build/pytorch_parity_metal --profile=apple-metal --requirement=apple.metal.device_dispatch --output=build/parity-metal-device.json
```

Use the slice's actual requirement ID, never weaken it to one passing subcase. Slice 1 precedes the adapter, so its native test is its gate. Slice 2 runs policy/unit tests plus the first real requirement. For Core AI slices:

```sh
build/build_apple --backend=all
nim c -r -d:ssl -d:coreai --gc:orc -o:build/test_coreai_models tests/apple/test_coreai_models.nim
nim c -d:ssl -d:coreai --gc:orc -o:build/pytorch_parity_coreai tests/parity/run_parity.nim
build/pytorch_parity_coreai --profile=apple-coreai --requirement=apple.coreai.model_functions --output=build/parity-coreai-models.json
```

Again substitute the named focused module/requirement; Slices 12 and 13 each close two contracts and must gate both before committing. Run `test_parity.nim` after Slice 2's report changes. After Slice 6's generic integration run existing `tests/autograd/test_gate_basic.nim`, `test_gate_blas.nim`, `test_gate_hadamard.nim`, plus `tests/parity/test_parity.nim`, each with `nim c -r -d:ssl -o:build/<module> <path>` and no Apple define. Slice 7's focused module includes numeric CPU SGD preservation as well as native SGD; do not treat the existing momentum-only optimizer tests as SGD coverage.

For the Slice 9 capture test set `MTL_CAPTURE_ENABLED=1` only for that test process, retain a real `.gputrace` under `build/apple/`, and restore memory-fraction/capture state on every exit. For Slice 17 independently collect placement evidence from the one owned test process:

```sh
xcrun xctrace record --template 'Core AI' --output build/apple/coreai-placement.trace --launch -- build/test_coreai_placement
xcrun xctrace export build/apple/coreai-placement.trace --toc --output build/apple/coreai-placement-toc.xml
```

Export the actual compute-unit/signpost tables identified by that TOC using `--xpath`, and correlate PID, invocation interval, model/function and explicit unit identity. A preference or mere device activity is insufficient. Do not suppress privacy prompts, grant permissions, or trace other processes; if authorization is unavailable, retain the concrete capture error and leave hardware-specific placement unknown, as the existing requirement permits.

After the focused scenario and every contract it closes pass, create the slice's selective commit with the specified subject and actual verification command in its body. Generated authentic model files are owned outputs of the audited generator: include only those exact asset/manifest paths through the selective commit draft's reviewed `take` mechanism if not automatically tracked. Exclude unrelated changes and all `build/` output. Do not start the next implementation slice before that commit succeeds.

### Final acceptance before Slice 17 commit

```sh
nimble test_metal
nimble test_coreai
nimble parity
```

Require ten Metal and eight Core AI requirement passes, `native_executed = 18`, `unobserved = 0`, no Apple mismatch/error, nine CPU passes, gate `pass` and exit 0. Every GPU claim must have real completed GPU commands and zero CPU fallback; Core AI policy/unknown placement follows its separate truthful contract.

Also compile/run the ownership-sensitive focused modules once under `--gc:refc`: Metal storage/streams and Core AI layout/state/concurrency. ORC is already exercised by each slice. Recompile the unchanged CPU-only runner without Apple defines and require its explicit CPU selection to pass without loading bridges; its all-profile selection must instead fail closed with 18 unobserved Apple requirements. Policy tests cover malformed/duplicate observations and selector failures; do not add broad dataset/example/backend suites beyond these requested gates.

## Critical integration files

- `src/arraymancer/tensor/data_structure.nim`: include Metal in metadata dispatch, never in legacy GPU pointer access.
- `src/arraymancer/tensor.nim`: conditional explicit Metal imports/exports without CPU SDK dependency.
- `src/arraymancer/nn/optimizers/optimizers.nim`: native SGD with preserved existing CPU factory/update contracts.
- `tests/parity/parity_support.nim`: genuine execution state, counters, comparisons and fail-closed selection.
- `tests/parity/run_parity.nim`: selected in-process native execution and strict default integration.

## Assumptions & contingencies

- “Entire thing” means both Apple backends and every current `apple_requirements.json` contract, plus CPU regressions. It does not mean implementing every schema in the broader PyTorch inventory. No weakening or removal of the 18 required contracts.
- Execution begins only after plan approval. Existing commits/reports are context, not fresh proof. Preserve unrelated user changes throughout.
- Xcode 27.1/27A9275, macOS SDK 27.0, Swift 6.4, Metal Toolchain, `coreai-build` and the Core AI Instruments template were discovered live. Runtime device/capture/converter compatibility remains to be proved; tool presence is not execution proof.
- If the pinned converter fails, fix the developer generator against those actual APIs or report the external incompatibility; never drift Torch pins, hand-build a fake model, skip cases or substitute another framework. Continue reachable independent work, but no incomplete slice receives a success commit and final completion requires the real full gate.
- CPU/CUDA/OpenCL interfaces retain their current meaning. Unsupported Apple operations, unavailable libraries/devices and invalid layouts fail explicitly; no hidden backend transfer, host computation, mock native evidence or “expected failure” success.
