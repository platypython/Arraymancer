// Copyright 2026 Arraymancer contributors. Apache-2.0.
// Provides small custom kernels; dispatch grids must fit the caller's buffers.
#include <metal_stdlib>
using namespace metal;

kernel void vector_double(device const float *input [[buffer(0)]],
                          device float *output [[buffer(1)]],
                          uint index [[thread_position_in_grid]]) {
  output[index] = 2.0f * input[index];
}

kernel void vector_split(device const float *input [[buffer(0)]],
                         device float *doubled [[buffer(1)]],
                         device float *tripled [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
  doubled[index] = 2.0f * input[index];
  tripled[index] = 3.0f * input[index];
}

kernel void vector_inplace(device float *values [[buffer(0)]],
                           uint index [[thread_position_in_grid]]) {
  values[index] *= 2.0f;
}
