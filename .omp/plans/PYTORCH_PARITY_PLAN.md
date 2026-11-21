## Context

The overall goal is native-Nim feature parity with the latest stable PyTorch, without a Python/PyTorch/LibTorch dependency in Arraymancer itself. The user confirmed that the first deliverable remains a version-pinned capability inventory and executable conformance foundation—not native backend implementation or a full-parity claim. Core AI inference and Metal tensor/training execution are separate, required first-class targets: inventory their contracts and expose missing execution through strict Apple gates, not an unreviewed device row. Target PyTorch 2.15.0 from `ref/pytorch/`; retain Nim >= 1.6.0 compatibility and current public APIs. Python/PyTorch remain developer-only oracle tooling; ordinary Nim checks consume checked-in data offline.

## Grounded findings

- The upstream [latest-release API](https://api.github.com/repos/pytorch/pytorch/releases/latest) identifies stable `v2.15.0`, published 2026-10-28. The local tag resolves directly to commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`; its source `version.txt` is `2.15.0a0`, not the installed wheel version. The checked-out `ref/pytorch` main is a different revision, `2a2d844941bf776d1a49a8ff8730e90f5242f836`, marked `2.16.0a0`.
- At the stable commit, `aten/src/ATen/native/native_functions.yaml` contains 2,590 `- func:` records and `tools/autograd/derivatives.yaml` contains 688 `- name:` records. A read-only scan verified every declaration is a unique, plain, single-line signature containing `(`, `)` and `->`; no quoted or multiline declaration parser is needed for this pin. These include private operators and overloads; do not call either count a public-API count. `docs/source/pytorch-api.md` additionally lists compiler/export, distributed, device, serialization, data, sparse/quantization, profiling and other API areas.
- `tests/manual_checks/autograd_mean_{arraymancer.nim,pytorch.py}` and `tests/nn_primitives/references/test_nnp_gru_cell.py` establish the existing native/reference-pair convention. Searches of `tools`, `tests`, and `arraymancer.nimble` found no shared parity/conformance infrastructure.
- Direct source inspection establishes useful probes: rank-zero size reports zero (`laser/dynamic_stack_arrays.nim:product`, `laser/tensor/datatypes.nim:size`); `autograd/autograd_common.nim:no_grad_mode` is not exception-safe; `nn/loss/mean_square_error_loss.nim:mse_backward_ag` divides by the one-element loss-gradient size rather than the input size. These are source findings, not observed test results. Expose them; do not fix them in this foundation.
- The available compiler reports Nim 2.4.2. Git, ShellCheck, uv, Python 3.13, and `pyright-langserver` resolve on PATH. The LSP tool has no Python server registered; installed server availability does not mean diagnostic checks have passed.
- Pin the developer oracle to CPython 3.13.18 and `torch==2.15.0`. [Python.org](https://www.python.org/downloads/release/python-31318/) verifies that interpreter release; [PyPI metadata](https://pypi.org/pypi/torch/2.15.0/json) lists Python 3.13 support, Python >= 3.11, and a non-yanked Torch 2.15.0 release. No dependency installation, compilation, or runtime comparison occurred during planning.
- New Apple requirement: Core AI and Metal are first-class targets. Apple's [Core AI overview](https://developer.apple.com/core-ai/) describes native Swift model inference, `.aimodel` conversion through `coreai-torch`, CPU/GPU/Neural Engine execution, zero-copy paths and custom Metal kernels. This is Core AI, not a request to substitute Core ML.
- Apple's [Core AI DocC metadata](https://developer.apple.com/tutorials/data/documentation/coreai.json) marks macOS/iOS/etc. availability at 27.0 and identifies `AIModel`, `InferenceFunction`, `NDArray`, `ComputeStream`, `ComputeUnitKind`, and `SpecializationOptions`. Core AI is the inference/artifact path; Metal is the tensor-compute path, not an interchangeable name for Core AI or PyTorch MPS.
- A scoped case-insensitive identifier search for Metal/MPS/CoreAI/CoreML under native `src`, `tests`, `arraymancer.nimble`, and `nim.cfg` returned no matches. Existing public tensor exports previously inspected expose CPU plus conditional CUDA/OpenCL, not a native Apple tensor backend. Do not invent native Apple capability or label CPU execution as Apple execution.
- At the pinned PyTorch commit, `torch/mps/__init__.py:244-266` exports 21 public names, including shader/metallib loading, streams/events, RNG, memory controls and profiling. `synchronize` at lines 30-32 waits for all kernels in all MPS streams. These are now verified stable-tag observations, not assumptions from the development checkout.
- Stable `aten/src/ATen/native/mps/OperationUtils.mm:52-90` maps float32, float16, bfloat16, integer, bool and selected complex tensor types; it explicitly rejects float64 tensors with a type error. A dtype mapping is not proof that every operation supports that dtype. CPU float64 observations therefore cannot certify Metal.
- Stable `aten/src/ATen/mps/MPSFallback.mm:101-115` disables the general CPU fallback when `PYTORCH_ENABLE_MPS_FALLBACK` is absent or `"0"`, but still registers an explicit CPU fallback for `embedding_renorm_`. Neither an MPS output tensor nor that environment setting alone proves GPU execution.
- Apple's documented `NDArray.RawView(metalBuffer:...)` requires `.shared` storage, in-bounds shape/strides/byte offset, and exclusion of competing writes while the borrowed view lives. `InferenceFunction` owns weights/intermediates; caller-provided state/output views are updated in place, and preallocated outputs are omitted from returned outputs. `ComputeStream.currentWorkCompleted()` waits for prior work on that stream.
- Core AI `SpecializationOptions.preferredComputeUnitKind` permits fallback to other allowed units. Core AI inference on its explicitly declared CPU policy is legitimate Core AI execution, but is not Metal or Neural Engine coverage. Do not infer hardware placement from a preference.
- ShellCheck-approved read-only version commands found Xcode 27.1 build 27A9275, macOS SDK 27.0, and Apple Swift 6.4. The SDK's actual macOS `CoreAIRuntime` interface marks `AIModel`, `InferenceFunction` and `ComputeStream` available from macOS 27.0; `CoreAIDelegates` reexports that runtime and supplies specialization/loading APIs. Pin this installed stable Apple baseline; [Apple's support table](https://developer.apple.com/xcode/system-requirements/) corroborates Xcode 27.1/macOS SDK 27/Swift 6.4. No Swift compilation or Apple execution occurred.

## Approach

Execute steps 1–2 first. Capability annotations and mandatory Apple contracts in step 3 can proceed independently of CPU oracle generation in step 4. Step 5 validates these checked-in artifacts and compares real CPU observations; step 6 exposes both execution results and strict Apple coverage gaps. New infrastructure is necessary because existing paired references have no shared provenance, catalogue, comparator, or reporting contract.

### 1. Establish one reference identity

Create `tests/parity/reference.json` with exactly these pin fields. Apple versions identify the reviewed SDK/API baseline, not a new dependency or deployment requirement for ordinary CPU users:

```json
{
  "format": 1,
  "torch": "2.15.0",
  "source_commit": "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457",
  "python": "3.13.18",
  "apple": {
    "xcode": "27.1",
    "xcode_build": "27A9275",
    "macos_sdk": "27.0",
    "swift": "6.4",
    "minimum_macos": "27.0",
    "architecture": "arm64",
    "metal_api": "4"
  }
}
```

Load this file in the source-inventory generator, Python oracle, and native fixture validator. Reject missing fields, unsupported `format`, or inconsistent reference identities; never select a newer installed Torch or fall back to `ref/pytorch` main. Do not change `arraymancer.nimble` library dependencies or export the test infrastructure through `src/arraymancer.nim`.

### 2. Inventory immutable source, not checkout contents

Create the standalone Nim developer tool `tools/pytorch_inventory.nim`, using Nim standard-library JSON, string, option-parsing and process modules only. Read Git objects with direct Git argument arrays, not interpolated shell commands or Python imports. Default source repository: `ref/pytorch`; always address the full pinned commit.

Extract these four inventories from that commit:

| Kind | Input | Record boundary / identity |
| --- | --- | --- |
| `api_area` | `docs/source/pytorch-api.md` | Every non-option entry inside its MyST `toctree` blocks, including the external C++ entry. Preserve caption and target; ID is `api:` plus the exact target. |
| `aten_schema` | `aten/src/ATen/native/native_functions.yaml` | Every top-level `- func:` declaration, preserving its full scalar text, including private names and overloads. ID is `aten:` plus that declaration. |
| `autograd_rule` | `tools/autograd/derivatives.yaml` | Every top-level `- name:` declaration, preserving its full scalar text. ID is `autograd:` plus that declaration. |
| `mps_export` | `torch/mps/__init__.py` | Every string in its literal `__all__` list: 21 exports at this pin. ID is `mps:torch.mps.` plus the exported name, including exported classes/modules without pretending to inventory all their members. |

Recognize MyST block boundaries rather than treating every bare line as an API entry. For each YAML record, strip only the exact declaration prefix and surrounding whitespace, preserving the remaining plain signature verbatim. The pinned sources were verified to use unique, plain, single-line declarations; reject a quoted/block scalar or a signature lacking `(`, `)` or `->` with exit 2 instead of introducing a general YAML parser or silently ignoring it.

Parse only the literal `__all__ = [` block and its closing `]`, requiring one unique quoted identifier per entry with an optional trailing comma. Reject expressions, malformed entries, duplicate names or an empty block with exit 2. Include its pinned source line/blob just like the other records.

Write `tests/parity/inventory.json` only when explicitly requested. Its structure is `{format: 1, reference: <reference.json>, granularity: "api_areas_aten_schemas_autograd_rules_and_mps_exports", records: [...]}`. Each record has `kind`, `id`, `declaration`, and `source: {path, line, blob}`; for an MPS export, `declaration` is its fully qualified `torch.mps.<name>`. `line` is one-based and `blob` is that pinned source object's Git hash. Sort by `(kind, id)`, use UTF-8/LF with a final newline, and include no timestamps, absolute paths, or host-dependent fields.

CLI: `--source=PATH`, `--output=PATH`, `--check`, `--help`. With no arguments, show concise usage on stderr and emit inventory JSON on stdout without writing files. `--check` compares the canonical generated inventory against `tests/parity/inventory.json`, exits 0 when equal and 1 when stale, and never rewrites it; reject combining `--check` with `--output`. Missing repository/commit/blob, duplicate IDs, malformed declarations, or an empty source inventory exit 2 with the source path/reason. An explicitly requested output may be replaced only after successful extraction; missing parent directories or write failures exit 2. Never fetch, checkout, build, import, or modify `ref/pytorch`.

This catalogue is exhaustive at its stated granularity, not an exhaustive list of Python object members or a percentage of feature parity. Preserve that distinction in reporting.

### 3. Attach evidence-backed native capability annotations

Create `tests/parity/capabilities.json` as `{format: 1, reference: <reference.json>, areas: [...], apple_requirements: [...]}` with exactly one area row per `api_area` inventory ID and exactly one Apple row per requirement defined below. Rows contain `id`, `status`, `evidence`, and `gaps`; evidence entries contain repository-relative `path` and `symbol`, while gaps are short behavioral descriptions. Allowed initial statuses: `partial`, `missing`, `unreviewed`. No area or requirement receives a complete/verified label from compiling, finding a similarly named symbol, or running another framework.

Use these initial classifications; match the exact toctree targets, not displayed aliases:

- `partial`: `torch`, `tensors`, `tensor_attributes`, `tensor_view`, `storage` — cite `laser/tensor/datatypes.nim` (`Tensor`, `size`), `laser/dynamic_stack_arrays.nim` (`LASER_MAXRANK`, `product`), and tensor export/view APIs. Record static dtype, six-rank default capacity, scalar/empty differences, and incomplete operator/indexing semantics.
- `partial`: `nn`, `nn.functional`, `nn.init`, `nn.attention` — cite `src/arraymancer/nn.nim` exports and `nn/layers/attention.nim` (`MultiHeadAttention`, `scaled_dot_product_attention`). Record existing GQA/RoPE/KV caching rather than declaring attention absent; record missing full module lifecycle/Transformer/loss options.
- `partial`: `autograd` — cite `autograd_common.nim` (`backprop`, `no_grad_mode`): reverse-mode tape exists; seeded VJPs, retained/higher-order graphs, forward AD, mutation checks, and exception-safe mode restoration are gaps.
- `partial`: `optim` — cite `nn/optimizers/optimizers.nim` (`SGD`, `SGDMomentum`, `Adam`, `update`): limited algorithms/state options and no general optimizer checkpoint API.
- `partial`: `cpu`, `cuda`, `random` — cite CPU tensor storage/random constructors and the conditional CUDA tensor/global-stream implementation. Mark CUDA as source-present but not executed by this CPU suite, not hardware-verified. Record global RNG/stream limitations.
- `missing`: targets equal to or beginning `distributed`, plus `fsdp`, `rpc`, `symmetric_memory`, and `ddp_comm_hooks` — no native process-group/distributed-training counterpart found in the scoped native source/tests search.
- `missing`: `user_guide/torch_compiler/export`, `torch.compiler_api`, `data`, `sparse`, `quantization`, `profiler` — record the missing general native graph compiler/export, DataLoader, sparse/quantized tensor and profiler contracts. Numeric tensor I/O is not graph export or checkpoint/state-dict support; full-dataset MNIST/IMDb loaders are not a DataLoader.
- `missing`: `mps` — cite the inspected public tensor umbrella and CPU/CUDA/OpenCL storage/type boundaries, and the scoped Apple-identifier search. All 21 MPS exports remain visible in the source inventory; do not count CUDA/OpenCL availability, a PyTorch MPS oracle, or the host SDK as native Arraymancer Apple support.
- Every remaining API area starts `unreviewed`, with no fabricated implementation evidence. In particular, do not infer full linalg/distribution/device support from umbrella names or silently omit those areas.

Expand abbreviated native paths above under `src/arraymancer/`. Before authoring each evidence entry, re-read that symbol and retain its exact spelling. For missing areas, include the search scope `src/arraymancer;tests` and terms used in the gap description; do not invent source anchors for absent code. Validate exact area/Apple-requirement ID coverage and matching reference identities. These annotations are implementation evidence, not passing conformance observations.

#### Required Apple profiles and contracts

Create `tests/parity/apple_requirements.json` as `{format: 1, reference, profiles, requirements}`. Define exactly two profiles: `apple-metal` with `native_backend: "metal"`, `reference_backend: "torch.mps"` and `role: "tensor_training"`; `apple-coreai` with `native_backend: "coreai"`, `reference_backend: "apple_sdk"` and `role: "model_inference"`. Both have `priority: "required"` and use `reference.apple` as the reviewed macOS/Apple-silicon baseline. Other Apple platforms are not runtime-verified by this foundation.

Each requirement contains `id`, `profile`, `reference_ids`, `contract`, `scenario`, and `sources`. `scenario` is a concrete acceptance description, not a fabricated fixture or execution result. PyTorch sources carry the pinned commit/path/line/blob; Apple sources carry the public documentation URL, symbol, SDK-relative interface path, and SHA-256 calculated from the installed pinned SDK. Do not vendor SDK interfaces or fetch documentation during ordinary checks. Core AI is an Apple extension requirement, not an invented PyTorch API area or ATen operator.

Record these ten Metal requirements, with all initial native implementation statuses `missing`:

| Requirement ID | Required contract / deterministic acceptance scenario |
| --- | --- |
| `apple.metal.device_dispatch` | Explicit native Metal device selection and availability; CPU tensors remain CPU unless transferred. An unavailable Metal request is visible, never silent CPU execution. Link `device_count` and `is_available`. |
| `apple.metal.dtypes` | Inventory the pinned dtype mappings separately from per-operation support. Float32 is the baseline math/training dtype; float16/bfloat16, signed/unsigned integers, bool and mapped complex types need individual coverage. Float64 tensor requests must be explicitly rejected, not silently cast or executed on CPU. |
| `apple.metal.tensor_views` | CPU↔Metal round trip for `[1,2,3,4,5,6]`; transpose and offset-slice observations preserve logical values and aliasing. Reuse the transpose-mutation scenario from step 4 in float32. Track materialization explicitly; a contiguous copy cannot masquerade as an aliasing view. |
| `apple.metal.tensor_math` | Float32 versions of the two broadcast cases and 2-D matmul from step 4, plus scalar/empty metadata. Preserve shape, numel, dtype and native device. CPU float64 fixtures do not satisfy this requirement. |
| `apple.metal.autograd` | Native Metal mean and mean-MSE forward/backward, using step 4's values/gradients in float32; graph values and gradients stay on Metal. Report unsupported operations rather than calculating replacement gradients on CPU. |
| `apple.metal.optimizer_sgd` | Native forward→mean-MSE→backward→SGD: weights `[1,2]`, input `[3,4]`, scalar target 0, learning rate 0.01 produce prediction 11, loss 121, gradients `[66,88]`, updated weights `[0.34,1.12]`. Inventory gradient-reset semantics separately; PyTorch `step` does not itself imply `zero_grad`. |
| `apple.metal.queue_lifetime` | Queued dependent kernels complete in order; synchronization precedes host reads. Buffers survive queued work and metadata aliases. Inventory `synchronize`, `Event`, `Stream`, `stream`, `current_stream`, `set_stream`, and `default_stream` separately from a single global queue. |
| `apple.metal.memory_profiling` | Distinguish active tensor bytes, driver/cache allocation and recommended limits; cover allocation/release/cache behavior and invalid memory fractions. Driver/cache bytes need not equal live tensor bytes. GPU/copy/fallback observations—not an availability flag—are needed for placement claims; include the `profiler` export. |
| `apple.metal.rng` | Seed a native generator, save state, draw, restore state and redraw identically on that backend. Inventory `manual_seed`, `seed`, `get_rng_state`, `set_rng_state`; do not assume CPU/GPU bitwise RNG equivalence. |
| `apple.metal.custom_kernels` | Inventory `compile_shader` and `load_metallib`; a native vector-doubling kernel maps float32 `[1,2,3,4]` to `[2,4,6,8]`, with actual device execution/completion evidence and visible compilation/load errors. A PyTorch shader demo is not Arraymancer kernel support. |

Collectively link every one of the 21 `mps_export` IDs to at least one Metal requirement. Link dtype/math/view/autograd/optimizer requirements to their existing API-area IDs as appropriate. Source the dtype/fallback rules from the pinned files in the findings; the public MPS exports come from the immutable `__all__` block, not main.

Record these eight distinct Core AI requirements, also initially `missing`:

| Requirement ID | Required contract / deterministic acceptance scenario |
| --- | --- |
| `apple.coreai.model_functions` | Native `.aimodel`/`.aimodelc` loading, function-name/descriptor inspection, missing-function versus load error, and inference. A named identity function on float32 `[1.25,-2]` returns exactly those values. Use `AIModel`/`loadFunction(named:)`; do not treat a file's existence as execution. |
| `apple.coreai.ndarray_descriptors` | Honor strict descriptor dtype/shape/rank. A float32 `[2]` identity accepts `[1.25,-2]`; wrong dtype or shape is rejected explicitly, not silently converted. Enumerating `NDArray.ScalarType` does not establish model support for every type. |
| `apple.coreai.strided_layout` | Identity inference on logical `[[1,2,3],[4,5,6]]` preserves values for contiguous input and storage `[99,1,4,2,5,3,6,99]` with shape `[2,3]`, element strides `[1,2]` and a one-element offset. Resolve dynamic dimensions before preferred-stride/minimum-byte queries; do not assume preferred layout is contiguous. Returned allocated NDArrays are row-major contiguous as documented. |
| `apple.coreai.metal_buffer_interop` | Borrow the strided-layout scenario from a float32 `.shared` `MTLBuffer`, using byte offset 4, and preserve both guard values 99. Keep the owner and borrowed view alive, and exclude competing writes through inference. Unsupported storage/layout is visible; no zero-copy claim without actual buffer/copy evidence. |
| `apple.coreai.stream_lifetime` | Encode producer→inference→consumer work using `ComputeStream(commandQueue:)`; await `currentWorkCompleted()` before host access/reuse. The owning buffer and loaded function outlive pending work. Waiting on one stream does not certify unrelated queues. |
| `apple.coreai.stateful_execution` | A declared scalar state starts at 0 and increments to 1 then 2 across two calls using the same mutable state view; missing required state produces an error. Preallocated output views update in place and do not also appear in returned outputs. |
| `apple.coreai.concurrency` | Independent identity calls on `[1,2]` and `[3,4]` through one loaded function produce their own outputs without shared-state corruption. Function-owned weights/intermediates and caller-owned mutable state remain distinct; no unsynchronized concurrent reuse of one state buffer. |
| `apple.coreai.compute_placement` | Record available units and explicit `.default`, `.cpuOnly`, GPU-preferred and Neural-Engine-preferred policies. CPU-only Core AI correctness is Core AI coverage, not Metal/ANE coverage. Preferences permit fallback; unobserved actual placement stays unverified rather than becoming a GPU/ANE pass. |

Use the verified Apple pages for [AIModel](https://developer.apple.com/documentation/coreai/aimodel), [InferenceFunction](https://developer.apple.com/documentation/coreai/inferencefunction), [NDArrayDescriptor](https://developer.apple.com/documentation/coreai/ndarraydescriptor), [Metal raw views](https://developer.apple.com/documentation/coreai/ndarray/rawview/init(metalbuffer:byteoffset:scalartype:shape:strides:interleavelayout:)), [ComputeStream](https://developer.apple.com/documentation/coreai/computestream), and [compute-unit preference](https://developer.apple.com/documentation/coreai/specializationoptions/preferredcomputeunitkind). SDK paths are relative to `MacOSX27.0.sdk`: `System/Library/SubFrameworks/CoreAIRuntime.framework/Versions/A/Modules/CoreAIRuntime.swiftmodule/arm64e-apple-macos.swiftinterface` and the analogous `CoreAIDelegates.framework/Versions/A/Modules/CoreAIDelegates.swiftmodule/arm64e-apple-macos.swiftinterface`. These interfaces establish macOS availability where individual web pages omit it. Pin their observed SHA-256 values in Apple source records:

| SDK interface | SHA-256 |
| --- | --- |
| `CoreAIRuntime` | `b5ca382d23c0177490b5b8cf1b14e7fb45b9a483b3feee072060a57a12f5a5dd` |
| `CoreAIDelegates` | `cf66ad72fab3fd71bf6c7cf2e5d96cfb8b1ebd9d4345ab8a2cfbb4f611db032c` |

The foundation delivers these 18 requirement records and their strict coverage gates, not their native implementations or model assets. Add no placeholder Metal/Core AI runtime, stub adapter, Swift bridge, synthetic backend observation, Python runtime dependency, or fake `.aimodel` fixture. Availability probes, framework-only Swift demos and PyTorch MPS execution cannot satisfy native conformance. Metal training and Core AI inference never satisfy each other's gate.

### 4. Generate real CPU reference observations

Create `tests/parity/references/generate_fixtures.py` with a module docstring and explicit per-case functions; reuse the existing reference-script convention, but emit structured JSON rather than human-transcribed tensor output. It runs only installed Torch 2.15.0 on CPython 3.13.18, checks both versions before computation/output, and never imports or builds `ref/pytorch`.

Use CPU float64 tensors, fixed inputs, `torch.set_num_threads(1)`, and explicit gradient-enabled scopes for AD cases. No RNG, datasets, networking, GPU discovery, or benchmark timing is needed. Compare Torch's version after removing only a local `+...` build suffix; reject prerelease or different versions. Record the actual `torch.version.git_version` separately from the target source commit, without pretending that a wheel's build hash was verified to equal the source tag.

Generate `tests/parity/fixtures.json` with `{format: 1, reference: <reference.json>, oracle: {torch, python, git_commit, device: "cpu"}, cases: [...]}`. Store the actual full installed version in `oracle.torch`, normalize only its local build suffix for comparison with the pin, and store the exact interpreter version in `oracle.python`. Cases contain `id`, `area`, `inputs`, `atol: 1e-10`, `rtol: 1e-10`, and `expected`. Tensor input descriptors contain `shape` and flat C-order `values`; the case contract fixes float64/CPU. Tensor observations contain `shape`, `numel`, `dtype: "float64"`, `device: "cpu"`, and flat logical C-order `values`. Metadata-only observations omit `values` deliberately. Preserve scalar shape `[]` versus `[1]`, and zero-size dimensions.

Implement all nine cases below in both oracle and native runner. Inputs live in the fixture and are read by the native adapter, not repeated as a second independent set of literals.

| Case ID / area | Input and action | Expected observation |
| --- | --- | --- |
| `tensor.transpose_alias` / `api:tensor_view` | `a=[[1,2,3],[4,5,6]]`; transpose; set view index `[0,1]` to 9 | Base shape `[2,3]`, values `[1,2,3,9,5,6]`; view shape `[3,2]`, values `[1,9,2,5,3,6]`. Both are actual tensor observations after mutation. |
| `tensor.empty_shape` / `api:tensors` | Construct an empty tensor of shape `[2,0,3]` | Shape `[2,0,3]`, numel 0, values `[]`. |
| `tensor.scalar_metadata` / `api:tensors` | Construct an uninitialized tensor with empty shape `[]` | Metadata only: shape `[]`, numel 1, float64/CPU. Never inspect uninitialized storage. |
| `tensor.broadcast_same_rank` / `api:torch` | `a=[[0],[10]]`, `b=[[0,1,2]]`; add | Shape `[2,3]`, values `[0,1,2,10,11,12]`. |
| `tensor.broadcast_leading_dims` / `api:torch` | `a=[[0,1,2],[3,4,5]]`, `b=[10,20,30]`; add | Shape `[2,3]`, values `[10,21,32,13,24,35]`. |
| `tensor.matmul_2d` / `api:torch` | `[[1,2],[3,4]] @ [[5,6],[7,8]]` | Shape `[2,2]`, values `[19,22,43,50]`. |
| `autograd.mean_gradient` / `api:autograd` | Require grad on `[[1,2],[3,4]]`; backpropagate its mean | Observe input gradient only: shape `[2,2]`, values `[0.25,0.25,0.25,0.25]`. |
| `loss.mse_mean` / `api:nn.functional` | Prediction `[[1,2,3],[4,5,6]]`, zero target; mean MSE; backward | Observe loss as a scalar with value `91/6`, plus input gradient shape `[2,3]`, values `[1/3,2/3,1,4/3,5/3,2]`. |
| `autograd.no_grad_exception` / `api:autograd` | Require grad on `[1]`; enter no-grad, raise/catch `ValueError("parity probe")` outside the scope | Boolean `tracking_after_exception=true`; observe actual tracking state after unwinding. |

Use these exact object keys for the shared case contract:

- Transpose: `inputs.a`, `inputs.mutation: {index: [0,1], value: 9}`; `expected: {base: <tensor>, view: <tensor>}`.
- Empty/scalar construction: `inputs.shape`; `expected: {result: <tensor-or-metadata>}`.
- Both broadcasts and matmul: `inputs.a`, `inputs.b`; `expected: {result: <tensor>}`.
- Mean gradient: `inputs.x`; `expected: {grad: <tensor>}`.
- MSE: `inputs.prediction`, `inputs.target`; `expected: {loss: <tensor>, grad: <tensor>}`.
- No-grad exception: `inputs.x`; `expected: {tracking_after_exception: true}`.


The expected numbers describe acceptance, not permission to fabricate generated data: obtain and check in actual oracle output. Emit NaN and infinities as the JSON strings `"nan"`, `"+inf"`, `"-inf"` rather than non-standard numeric literals; serialize with `allow_nan=False`. CLI: `--output=PATH`, `--help`; no arguments print concise help to stderr and JSON to stdout. Missing/wrong interpreter/Torch or any case failure exits 2 before replacing output. No timestamps or absolute paths in fixtures.

### 5. Execute native behavior and compare it without hiding failures

Create `tests/parity/native_cases.nim` and `tests/parity/parity_support.nim`; these are test-only modules, not a second tensor runtime.

Required cross-module signatures:

```nim
type Difference* = object
  path*: string
  expected*, actual*: JsonNode

proc loadAppleRequirements*(path: string): JsonNode
proc collectAppleCoverage*(requirements, capabilities: JsonNode;
                           profiles: openArray[string]): JsonNode

proc loadFixtures*(path: string): JsonNode
proc compareObservations*(expected, actual: JsonNode;
                          atol, rtol: float64): seq[Difference]
proc runCase*(caseData: JsonNode): JsonNode
```

`loadFixtures` validates format/reference/oracle versions, exactly the nine required unique case IDs from step 4, each case's prescribed area, tensor-input shape/value counts, expected observation structure, and finite nonnegative tolerances. Missing any required case is invalid even with `--case`; selection happens only after the complete dataset is validated. Bad/missing data raises `ValueError`, never becomes an empty or silently reduced passing run. The scalar-metadata case has no input values and must not enter the ordinary packed-data constructor. Use the mathematical empty shape product of 1 when validating reference metadata, not Arraymancer's currently divergent `Metadata.product`.

For ordinary inputs, allocate one `newTensorUninit[float64](shape)` from `tensor/init_cpu.nim` and fill every element exactly once from validated flat fixture values. Use safe, owned contiguous raw access only during this fill. Refuse an input rank above `LASER_MAXRANK` as a case error before metadata writes; do not bypass the capacity or claim such an input is supported. For scalar metadata, call `newTensorUninit[float64]([])` and inspect metadata only. Serialize actual shape/size directly into JSON, and append logical values with the strided `enumerate(t)` pattern used by `tensor/exporting.nim:toFlatSeq`; avoid an intermediate flat sequence, deprecated `toRawSeq`, or a reshape that could hide view semantics.

Implement the cases with existing `.transpose`/index assignment, `+.`, `*`, `newContext`, `variable(..., requires_grad=true)`, `.mean`, `.backprop`, `.mse_loss`, `no_grad_mode`, and `is_grad_needed`. Allocate a fresh context for each AD case. Use the real native loss and graph paths; no corrected formulas, Torch calls, expected-value echoes, synthetic replacement kernels, or normalizing scalar `[1]` to `[]`.

Validate the two required Apple profiles, the exact ten/eight requirement IDs, complete capability rows, all 21 MPS-export links, nonempty concrete scenarios/sources and matching Apple baseline. Keep this validation in test-only support; do not build a generic backend/plugin runtime. `collectAppleCoverage` reports implementation annotations and absent native observations separately. Every selected Apple entry is `{id, profile, implementation_status, execution: "not_run", actual: null, reason: "No native conformance observation in this foundation."}`. This is missing-coverage reporting, not an executor that pretends to run or throws a placeholder exception. Do not accept a JSON annotation or CPU observation as executed Apple evidence.

Compare observation object keys, string/bool values, shape dimensions and integer metadata exactly. For finite float values require `abs(actual-expected) <= atol + rtol*abs(expected)`; compare each flat element, not means or aggregate error. Nonfinite tokens match only the same token; NaN must not accidentally pass against a finite value. Compare shape before values so equal flattened data cannot hide different shapes. Differences use JSON Pointer paths, e.g. `/loss/shape` or `/grad/values/0`, with expected/actual values. A differing array length produces a difference for that array rather than indexing past it; continue comparing other observation branches. Do not loosen tolerances after observing a failure.

### 6. Expose an honest CLI and focused permanent checks

Create `tests/parity/run_parity.nim` with CLI execution under `when isMainModule`. Load `reference.json`, `inventory.json`, `capabilities.json`, `apple_requirements.json`, and `fixtures.json` under `tests/parity/`. Options: `--profile=all|cpu|apple|apple-metal|apple-coreai`, `--case=ID`, `--fixtures=PATH`, `--output=PATH`, `--help`. Default `all` selects CPU plus both required Apple profiles; `apple` selects both Apple profiles. `--case` is valid only with explicit `--profile=cpu`; reject other combinations rather than silently dropping mandatory Apple coverage. With no arguments show usage on stderr, execute all nine CPU cases, include both Apple gap reports, and emit JSON on stdout. For all three developer CLIs, `--help` alone prints help and exits 0 without extraction, validation or execution.

Validate all artifacts/reference identities, exact area/requirement coverage and selectors before running anything; even an Apple-only invocation validates the checked-in data. CPU cases use the real native runner: catch `CatchableError` and `Defect` per case and continue without calling them skipped/passing. Each CPU result is `{id, area, profile: "cpu", status, actual, differences, error}`, with `pass`, `mismatch` or `error`; ordinary results retain actual observations and `error: null`, while exceptions have `actual: null`, `differences: []`, `error: {type, message}`. Apple profiles enumerate required coverage entries without calling CPU cases or inventing Apple execution. Unknown/empty selections are input errors.

Report `{format: 1, reference, scope: "conformance_foundation", profiles, selection, inventory_counts, capability_counts, results, counts, apple_coverage, apple_counts, notice}`. `profiles` is the exact selected profile-ID array; `selection` is `"all"` or the selected CPU case ID. `inventory_counts` separately reports `api_area`, `aten_schema`, `autograd_rule`, `mps_export`; `capability_counts` counts API-area `partial`, `missing`, `unreviewed` rows only. `results`/`counts` report only actually executed CPU `pass`, `mismatch`, `error` observations. `apple_coverage` contains the selected requirement entries; `apple_counts` has `required`, `partial`, `missing`, `unreviewed`, `native_executed: 0`, `unobserved`, `gate: "incomplete"` (or `"not_selected"` with zero counts for a CPU-only selection). No placeholder Apple passes enter execution totals. Keep `notice: "Full feature parity is not established by this run."`; never turn these counts into a global parity percentage.

Exit 0 only when every selected executable case passes and no selected required coverage is unobserved. Exit 1 for any CPU mismatch/error or required Apple observation absent; exit 2 for malformed/missing artifacts, pin/ID mismatch, invalid selector or output I/O failure. Thus `all`, `apple`, `apple-metal` and `apple-coreai` deliberately remain nonzero in this foundation even if CPU observations all pass. Write the collected report before returning 1. The Apple gate stays incomplete regardless of a hand-edited `partial` annotation; `verified`/`pass` implementation annotations are invalid. Do not add xfails, expected-failure suppression, automatic baseline acceptance, or `continue-on-error` CI gates.

Add a `parity` Nimble task beside existing test tasks. Create `build/` using their convention, then run exactly `nim c -r -d:ssl -o:build/pytorch_parity tests/parity/run_parity.nim --profile=all --output=build/pytorch-parity.json`. This strict default includes Apple, not an optional opt-in badge. The already-built CLI provides a dedicated Apple gate with `--profile=apple`. Do not reuse the existing `test` helper, which also runs legacy-RNG tests and compiles examples.

Create `tests/parity/test_parity.nim` with `unittest` and import it from `tests/tests_cpu.nim`. The normal suite checks the five intended matching CPU cases (transpose aliasing, empty shape, same-rank broadcast, 2-D matmul, mean gradient), comparator regressions for same-values/wrong-shape, finite-vs-NaN, infinity sign and tolerance boundaries, plus invalid provenance/IDs/selectors. Add deterministic coverage-policy checks: missing required Apple observation yields exit 1; changing an annotation to `partial` cannot make it pass; deleting an Apple profile/requirement/MPS-export link or injecting `verified` is invalid input. Use isolated malformed copies for these policy checks, not assertions that native support must always stay absent. Do not assert existing CPU bugs must remain broken. Strict commands expose actual mismatches and unobserved Apple requirements separately. Ordinary CI consumes these portable validation/comparator tests via the CPU aggregate; add no Apple SDK/Python/hardware dependency or fake green Apple runtime lane to it.

## Verification

Work from `/Users/derrick/git/ai/Arraymancer`. Run no builds, dependency installation, fixture generation or working-tree changes until execution approval. During execution, ShellCheck every shell script/command before running it, and obtain successful Python-file LSP diagnostics before any Python command.

### Prerequisites and oracle proof

Use Nim 2.4.2, Git, ShellCheck and uv; keep library syntax/dependencies compatible with Nim >= 1.6.0. Install declared Nim dependencies if absent, with Accelerate on macOS or configured BLAS/LAPACK elsewhere. CPU execution and Apple coverage-gap checks need no accelerator or external dataset. The reviewed Apple baseline is Xcode 27.1 build 27A9275/macOS SDK 27.0/Swift 6.4 on Apple silicon; fingerprint its two cited SDK interfaces when authoring Apple requirements, but do not require SDK access in ordinary checks or change the system's selected Xcode.

Create the isolated developer oracle with ShellCheck-approved commands:

```sh
mkdir -p build
uv venv --python 3.13.18 build/parity-oracle
uv pip install --python build/parity-oracle/bin/python 'torch==2.15.0'
```

Before executing the new Python file, request LSP diagnostics on it using that environment's interpreter. The planning session's `lsp` helper could not route Python despite installed `pyright-langserver`; if still unconfigured, use the installed server directly over stdio, without changing OMP settings: initialize with the repository root and diagnostic capability, send `initialized` and Python configuration with `pythonPath` set to the absolute oracle interpreter, open the generator with its current text/version, answer workspace-configuration requests from those settings, and wait for `textDocument/publishDiagnostics` for that document/version. Require no error-severity diagnostics, then send shutdown/exit. Do not substitute executing Python for this prerequisite or count timeout/no diagnostics as success.

Generate the real oracle once:

```sh
build/parity-oracle/bin/python tests/parity/references/generate_fixtures.py --output=tests/parity/fixtures.json
```

Inspect its provenance and the nine observations against the concrete table in step 4, especially scalar shape/numel, transposed alias values, loss `91/6`, gradients `2*x/6`, and restored tracking. This is the version-specific reference proof; do not transcribe numbers into a file and label them runtime-generated.

### Source inventory proof

Generate the pinned inventory and complete its capability rows, then exercise the non-mutating stale-check path:

```sh
nim c -o:build/pytorch_inventory tools/pytorch_inventory.nim
build/pytorch_inventory --output=tests/parity/inventory.json
build/pytorch_inventory --check
```

Expect 2,590 ATen records, 688 derivative records, 21 MPS exports, all API-area entries from the pinned root index, immutable commit/blob provenance, and no 2.16-main identities. `--check` exits 0 without modifying files. An invalid `--source` exits 2 without replacing the inventory. Validate both required Apple profiles, all 18 requirements, complete native-status annotations and links covering all 21 MPS exports; Core AI records remain a separate Apple extension inventory, never extra PyTorch API/ATen counts. Unreviewed areas must remain counted.

### Native end-to-end proof

After the artifacts and code are complete, run the focused new suite once and the strict report once:

```sh
nim c -r -d:ssl -o:build/test_parity tests/parity/test_parity.nim
nimble parity
```

The focused suite must pass the five matching CPU cases and comparator/input/Apple-policy regressions. `nimble parity` must produce `build/pytorch-parity.json` with nine real CPU observations plus all 18 selected Apple requirements, `native_executed: 0`, `unobserved: 18`, and an incomplete Apple gate. Current native source predicts scalar/MSE/no-grad mismatches and possibly a mixed-rank broadcast mismatch/error; record actual results rather than enforce a defect baseline. Any such mismatch yields exit 1; absent Apple observations independently require exit 1 even if CPU behavior changes to match.

Run the already-built `build/pytorch_parity --profile=cpu --case=tensor.matmul_2d`: actual shape `[2,2]`, values `[19,22,43,50]`, exit 0, `profiles: ["cpu"]`, Apple gate `"not_selected"` and the no-full-parity notice. An unknown case, unknown profile, or `--case` without explicit CPU selection exits 2. Ordinary checks must not invoke Python, require `ref/`, download data, inspect an SDK, or discover GPU hardware.

Exercise the dedicated Apple gate once with `build/pytorch_parity --profile=apple --output=build/pytorch-parity-apple.json`: zero executed CPU/Apple cases, ten Metal plus eight Core AI requirements, all native observations absent, and exit 1 after writing the report. No expected values or CPU results may appear as Apple `actual` observations. This proves the gate's honesty and completeness, not accelerator execution. Individual profile selection must retain only its own ten/eight requirements; validate that selection policy in the focused suite rather than rerunning hardware-free commands repeatedly.

First-delivery completion means reproducible pinned PyTorch/CPU-oracle artifacts, the pinned Apple API baseline and its 18 evidence-backed contracts, complete capability annotations, an offline comparator/CLI whose default includes Apple, and passing focused policy/comparator checks. Strict reports must expose CPU divergences and incomplete Apple gates. Native Metal/Core AI execution, platform hardware coverage and full parity remain unestablished, not silently marked done.

## Assumptions & contingencies

- Native functional equivalents are the eventual goal; this delivery leaves existing APIs and implementation behavior unchanged. Python/PyTorch are allowed only for developer-side oracle generation, not library execution or ordinary tests.
- PyTorch 2.15.0 and CPython 3.13.18 remain fixed even if newer versions appear. If the local source commit/blobs are unavailable, stop source regeneration and report the exact missing objects; do not fetch or mutate the reference checkout automatically. Already-checked-in data remain sufficient for offline native checks.
- If the pinned interpreter/wheel cannot be obtained, network access is unavailable, or successful Python LSP diagnostics cannot be obtained, continue source inventory and reachable native-tool work but report oracle generation/verification as blocked. Do not build PyTorch from `ref/`, switch versions, fabricate fixture provenance, or claim the first delivery complete without real oracle output.
- No GPU/Neural Engine/distributed/performance equivalence is measured in this foundation. Required Apple profiles stay explicitly incomplete and nonzero until genuine native observations exist; missing hardware or native executors cannot become skipped or passing coverage. Core AI's declared CPU execution is not Metal fallback, and a GPU preference is not verified placement. The baseline inventories macOS/Apple-silicon contracts; it does not claim runtime verification on other Apple platforms.
