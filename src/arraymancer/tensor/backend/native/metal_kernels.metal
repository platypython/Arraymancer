// Copyright 2026 Arraymancer contributors. Apache-2.0.
// Built-in Metal copy and float32 math kernels.
#include <metal_stdlib>
using namespace metal;

struct Descriptor {
  int rank, dtype;
  long shape[8], strides[8];
  long offset;
  ulong capacity;
};
struct MathParameters {
  int operation, scalar_input, axis, reserved;
  float scalar;
};

ulong elementCount(constant Descriptor &d) {
  ulong count = 1;
  for (int axis = 0; axis < d.rank; ++axis) count *= ulong(d.shape[axis]);
  return count;
}

long elementAddress(ulong index, constant Descriptor &d) {
  long result = d.offset;
  for (int axis = d.rank - 1; axis >= 0; --axis) {
    result += long(index % ulong(d.shape[axis])) * d.strides[axis];
    index /= ulong(d.shape[axis]);
  }
  return result;
}

#define COPY_KERNEL(NAME, TYPE) \
kernel void NAME(device const TYPE *source [[buffer(0)]], \
                 device TYPE *destination [[buffer(1)]], \
                 constant Descriptor &input [[buffer(2)]], \
                 constant Descriptor &output [[buffer(3)]], \
                 uint tid [[thread_position_in_grid]]) { \
  ulong count = elementCount(input); \
  if (tid < count) destination[elementAddress(tid, output)] = source[elementAddress(tid, input)]; \
}
COPY_KERNEL(copy8, uchar)
COPY_KERNEL(copy16, ushort)
COPY_KERNEL(copy32, uint)
COPY_KERNEL(copy64, ulong)

#define SET_KERNEL(NAME, TYPE) \
kernel void NAME(device TYPE *destination [[buffer(0)]], \
                 constant TYPE &value [[buffer(1)]], \
                 constant Descriptor &output [[buffer(2)]]) { \
  destination[output.offset] = value; \
}
SET_KERNEL(set8, uchar)
SET_KERNEL(set16, ushort)
SET_KERNEL(set32, uint)
SET_KERNEL(set64, ulong)

kernel void math_elementwise(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Descriptor &ad [[buffer(3)]],
    constant Descriptor &bd [[buffer(4)]],
    constant Descriptor &od [[buffer(5)]],
    constant MathParameters &parameters [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= elementCount(od)) return;

  float av = 0.0f;
  if (parameters.operation != 6)
    av = a[elementAddress(tid, ad)];

  float bv = parameters.scalar;
  if (!parameters.scalar_input && parameters.operation != 5 &&
      parameters.operation != 6 && parameters.operation != 8 &&
      parameters.operation != 9)
    bv = b[elementAddress(tid, bd)];

  float value;
  switch (parameters.operation) {
    case 1: value = av + bv; break;
    case 2: value = av - bv; break;
    case 3: value = av * bv; break;
    case 4: value = av / bv; break;
    case 5: value = -av; break;
    case 6: value = parameters.scalar; break;
    case 7: value = av - parameters.scalar * bv; break;
    case 8: value = parameters.scalar - av; break;
    case 9: value = parameters.scalar / av; break;
    case 23: value = (av - bv) * (av - bv); break;
    default: return;
  }
  output[elementAddress(tid, od)] = value;
}

kernel void math_mse_backward(
    device const float *upstream [[buffer(0)]],
    device const float *prediction [[buffer(1)]],
    device const float *target [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Descriptor &ud [[buffer(4)]],
    constant Descriptor &pd [[buffer(5)]],
    constant Descriptor &td [[buffer(6)]],
    constant Descriptor &od [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  const ulong count = elementCount(pd);
  if (ulong(tid) >= count) return;
  const float upstreamValue = upstream[elementAddress(0, ud)];
  const float difference =
      prediction[elementAddress(tid, pd)] - target[elementAddress(tid, td)];
  output[elementAddress(tid, od)] = upstreamValue * 2.0f * difference / float(count);
}

kernel void math_matmul(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Descriptor &ad [[buffer(3)]],
    constant Descriptor &bd [[buffer(4)]],
    constant Descriptor &od [[buffer(5)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  constexpr uint tile = 16;
  threadgroup float tileA[16][16];
  threadgroup float tileB[16][16];

  const ulong rows = ulong(ad.shape[0]);
  const ulong inner = ulong(ad.shape[1]);
  const ulong columns = ulong(bd.shape[1]);
  const ulong row = ulong(group.y) * tile + local.y;
  const ulong column = ulong(group.x) * tile + local.x;
  float accumulator = 0.0f;

  for (ulong base = 0; base < inner; base += tile) {
    const ulong aRow = ulong(group.y) * tile + local.y;
    const ulong aColumn = base + local.x;
    const ulong bRow = base + local.y;
    const ulong bColumn = ulong(group.x) * tile + local.x;
    tileA[local.y][local.x] =
        (aRow < rows && aColumn < inner)
        ? a[elementAddress(aRow * inner + aColumn, ad)] : 0.0f;
    tileB[local.y][local.x] =
        (bRow < inner && bColumn < columns)
        ? b[elementAddress(bRow * columns + bColumn, bd)] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < tile; ++k)
      accumulator += tileA[local.y][k] * tileB[k][local.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < rows && column < columns)
    output[elementAddress(row * columns + column, od)] = accumulator;
}

kernel void math_reduce(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant Descriptor &od [[buffer(3)]],
    constant MathParameters &parameters [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup float partial[256];
  const ulong count = elementCount(id);
  const ulong index = ulong(group) * 256ul + ulong(lane);
  partial[lane] = index < count ? input[elementAddress(index, id)] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride) partial[lane] += partial[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    float result = partial[0];
    if (parameters.scalar_input) result /= parameters.scalar;
    output[elementAddress(group, od)] = result;
  }
}

kernel void math_axis_sum(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant Descriptor &od [[buffer(3)]],
    constant MathParameters &parameters [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup float partial[256];
  const ulong outputIndex = ulong(group);
  ulong remainder = outputIndex;
  ulong coordinates[8];
  for (int axis = od.rank - 1; axis >= 0; --axis) {
    const ulong dimension = ulong(od.shape[axis]);
    coordinates[axis] = remainder % dimension;
    remainder /= dimension;
  }

  const int reducedAxis = parameters.axis;
  ulong baseIndex = 0;
  ulong inputStride = 1;
  ulong axisStride = 1;
  for (int axis = id.rank - 1; axis >= 0; --axis) {
    if (axis == reducedAxis)
      axisStride = inputStride;
    else
      baseIndex += coordinates[axis] * inputStride;
    inputStride *= ulong(id.shape[axis]);
  }

  const ulong axisLength = ulong(id.shape[reducedAxis]);
  float value = 0.0f;
  for (ulong index = ulong(lane); index < axisLength; index += 256ul)
    value += input[elementAddress(baseIndex + index * axisStride, id)];
  partial[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride) partial[lane] += partial[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    output[elementAddress(outputIndex, od)] = partial[0];
}
