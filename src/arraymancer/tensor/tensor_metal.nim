# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides the explicit native Metal tensor API.
import ./[metal_types, init_metal, shapeshifting_metal, accessors_metal]
export metal_types, init_metal, shapeshifting_metal, accessors_metal
import ./metal_runtime
export metal_runtime
import ./[operators_metal, aggregate_metal]
export operators_metal, aggregate_metal
import ./random_metal
export random_metal
