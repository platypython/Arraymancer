// Copyright 2026 Arraymancer contributors. Apache-2.0.
// Provides representation-preserving GPU copies with signed element strides.
#include <metal_stdlib>
using namespace metal;
struct Descriptor {
  int rank, dtype;
  long shape[8], strides[8];
  long offset;
  ulong capacity;
};
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
  ulong count = 1; \
  for (int axis = 0; axis < input.rank; ++axis) count *= input.shape[axis]; \
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
