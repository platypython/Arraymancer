# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Runs native Metal contracts separately from the CPU-only aggregate.
import ./[test_metal_storage, test_metal_dtypes, test_metal_views,
  test_metal_math, test_metal_autograd, test_metal_product_oracle,
  test_metal_cumsum, test_metal_cumsum_oracle, test_metal_cumprod, test_metal_optimizer,
  test_metal_streams, test_metal_memory, test_metal_rng, test_metal_kernels]
