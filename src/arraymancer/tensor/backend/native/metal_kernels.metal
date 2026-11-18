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

struct RngParameters {
  uint counter[4];
  ulong seed;
};

inline uint2 multiplyWide(uint left, uint right) {
  const ulong product = ulong(left) * ulong(right);
  return uint2(uint(product >> 32), uint(product));
}

inline uint4 philox4x32_10(uint4 counter, uint2 key) {
  for (uint round = 0; round < 10; ++round) {
    const uint2 product0 = multiplyWide(0xD2511F53u, counter.x);
    const uint2 product1 = multiplyWide(0xCD9E8D57u, counter.z);
    counter = uint4(product1.x ^ counter.y ^ key.x, product1.y,
                    product0.x ^ counter.w ^ key.y, product0.y);
    key += uint2(0x9E3779B9u, 0xBB67AE85u);
  }
  return counter;
}

kernel void rng_philox4x32_10(
    device float *output [[buffer(0)]],
    constant RngParameters &parameters [[buffer(1)]],
    constant ulong &count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const ulong index = ulong(tid);
  if (index >= count) return;

  const ulong block = index >> 2;
  uint4 counter = uint4(parameters.counter[0], parameters.counter[1],
                        parameters.counter[2], parameters.counter[3]);
  const uint lowBlock = uint(block);
  const uint highBlock = uint(block >> 32);
  ulong sum = ulong(counter.x) + ulong(lowBlock);
  counter.x = uint(sum);
  ulong carry = sum >> 32;
  sum = ulong(counter.y) + ulong(highBlock) + carry;
  counter.y = uint(sum);
  carry = sum >> 32;
  sum = ulong(counter.z) + carry;
  counter.z = uint(sum);
  carry = sum >> 32;
  counter.w += uint(carry);

  const uint2 key = uint2(uint(parameters.seed), uint(parameters.seed >> 32));
  const uint4 words = philox4x32_10(counter, key);
  const uint word = words[uint(index & 3ul)];
  output[tid] = float(word >> 8) * (1.0f / 16777216.0f);
}

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

long innerSpanAddress(uint index, constant Descriptor &d) {
  const uint width = uint(d.shape[d.rank - 1]);
  long address = d.offset + long(index % width) * d.strides[d.rank - 1];
  index /= width;
  for (int axis = d.rank - 2; axis >= 0; --axis) {
    address += long(index % uint(d.shape[axis])) * d.strides[axis];
    index /= uint(d.shape[axis]);
  }
  return address;
}

template<int layout>
long mathAddress(uint index, constant Descriptor &d) {
  if (layout == 1) return d.offset + long(index);
  if (layout == 2) return innerSpanAddress(index, d);
  return elementAddress(index, d);
}

template<int layout>
void elementwise(device const float *a, device const float *b, device float *output,
    constant Descriptor &ad, constant Descriptor &bd, constant Descriptor &od,
    constant MathParameters &parameters, uint tid) {
  if (ulong(tid) >= elementCount(od)) return;

  float av = 0.0f;
  if (parameters.operation != 6)
    av = a[mathAddress<layout>(tid, ad)];

  float bv = parameters.scalar;
  if (!parameters.scalar_input &&
      (parameters.operation <= 4 || parameters.operation == 7 ||
       parameters.operation == 13 || parameters.operation == 15 ||
       parameters.operation == 17 || parameters.operation == 19 || parameters.operation == 23))
    bv = b[mathAddress<layout>(tid, bd)];

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
    case 10: value = exp(av); break;
    case 11: value = log(av); break;
    case 12: value = sqrt(av); break;
    case 13: value = av / (2.0f * bv); break;
    case 14: value = tanh(av); break;
    case 15: value = av * (1.0f - bv * bv); break;
    case 16: value = 1.0f / (1.0f + exp(-av)); break;
    case 17: value = bv * (1.0f - bv) * av; break;
    case 18: value = av < 0.0f ? 0.0f : av; break;
    case 19: value = bv <= 0.0f ? 0.0f : av; break;
    case 23: value = (av - bv) * (av - bv); break;
    default: return;
  }
  output[layout == 0 ? elementAddress(tid, od) : od.offset + long(tid)] = value;
}

#define ELEMENTWISE_KERNEL(NAME, LAYOUT) \
kernel void NAME(device const float *a [[buffer(0)]], \
    device const float *b [[buffer(1)]], device float *output [[buffer(2)]], \
    constant Descriptor &ad [[buffer(3)]], constant Descriptor &bd [[buffer(4)]], \
    constant Descriptor &od [[buffer(5)]], \
    constant MathParameters &parameters [[buffer(6)]], \
    uint tid [[thread_position_in_grid]]) { \
  elementwise<LAYOUT>(a, b, output, ad, bd, od, parameters, tid); \
}
ELEMENTWISE_KERNEL(math_elementwise, 0)
ELEMENTWISE_KERNEL(math_elementwise_packed, 1)
ELEMENTWISE_KERNEL(math_elementwise_inner_span, 2)

float4 innerSpanVector(device const float *input, uint index, constant Descriptor &d) {
  const long address = innerSpanAddress(index, d);
  if (d.strides[d.rank - 1] == 0) return float4(input[address]);
  return *reinterpret_cast<device const float4 *>(input + address);
}

kernel void math_binary_inner_span4(
    device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Descriptor &ad [[buffer(3)]], constant Descriptor &bd [[buffer(4)]],
    constant Descriptor &od [[buffer(5)]],
    constant MathParameters &parameters [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  const ulong index = ulong(tid) * 4ul;
  if (index >= elementCount(od)) return;
  const float4 av = innerSpanVector(a, uint(index), ad);
  const float4 bv = parameters.scalar_input ? float4(parameters.scalar) :
      innerSpanVector(b, uint(index), bd);
  float4 value;
  switch (parameters.operation) {
    case 1: value = av + bv; break;
    case 2: value = av - bv; break;
    case 3: value = av * bv; break;
    case 4: value = av / bv; break;
    case 7: value = av - parameters.scalar * bv; break;
    default: return;
  }
  reinterpret_cast<device float4 *>(output + od.offset)[tid] = value;
}

struct PackedUnaryParameters {
  uint count;
  long inputOffset, outputOffset;
};
static_assert(sizeof(PackedUnaryParameters) == 24, "Metal direct unary parameter layout");

struct PackedUnaryTask {
  device const float *input;
  ulong packetCount;
  device float *output;
  uint count, padding;
};
static_assert(sizeof(PackedUnaryTask) == 32, "Metal unary task layout");

template<int operation, typename T>
T unaryValue(T value) {
  if (operation == 10) return exp(value);
  if (operation == 11) return log(value);
  return sqrt(value);
}

template<int operation>
void packedUnary(device const float *input, device float *output,
    uint count, uint tid) {
  const uint base = tid * 4;
  if (ulong(base) >= count) return;
  if (ulong(base) + 3 < count) {
    const float4 value = reinterpret_cast<device const float4 *>(input)[tid];
    reinterpret_cast<device float4 *>(output)[tid] = unaryValue<operation>(value);
  } else {
    for (uint lane = 0; lane < 4; ++lane)
      if (base + lane < count)
        output[base + lane] = unaryValue<operation>(input[base + lane]);
  }
}

#define PACKED_UNARY_TASK_KERNEL(NAME, OPERATION) \
kernel void NAME(constant PackedUnaryTask *tasks [[buffer(0)]], \
    uint3 tid [[thread_position_in_grid]]) { \
  if (!tasks[0].packetCount) { \
    constant PackedUnaryTask &task = tasks[tid.z]; \
    packedUnary<OPERATION>(task.input, task.output, task.count, tid.x); \
    return; \
  } \
  for (uint index = 0; index < uint(tasks[0].packetCount); ++index) { \
    constant PackedUnaryTask &task = tasks[index]; \
    packedUnary<OPERATION>(task.input, task.output, task.count, tid.x); \
  } \
}
PACKED_UNARY_TASK_KERNEL(math_exp_tasks, 10)
PACKED_UNARY_TASK_KERNEL(math_ln_tasks, 11)
PACKED_UNARY_TASK_KERNEL(math_sqrt_tasks, 12)

#define PACKED_UNARY_KERNEL(NAME, OPERATION) \
kernel void NAME(device const float *input [[buffer(0)]], \
    device float *output [[buffer(1)]], \
    constant PackedUnaryParameters &parameters [[buffer(2)]], \
    uint tid [[thread_position_in_grid]]) { \
  packedUnary<OPERATION>(input + parameters.inputOffset, \
    output + parameters.outputOffset, parameters.count, tid); \
}
PACKED_UNARY_KERNEL(math_exp_packed, 10)
PACKED_UNARY_KERNEL(math_ln_packed, 11)
PACKED_UNARY_KERNEL(math_sqrt_packed, 12)

// Every lane participates, including empty reductions and partial SIMD groups.
float cooperativeSum(float value, threadgroup float *partial, uint lane,
                     uint simdLane, uint simdGroup, uint simdWidth) {
  value = simd_sum(value);
  if (simdLane == 0) partial[simdGroup] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdGroup == 0) {
    value = 0.0f;
    for (uint index = simdLane; index < (256u + simdWidth - 1u) / simdWidth;
         index += simdWidth)
      value += partial[index];
    value = simd_sum(value);
  }
  return value;
}

kernel void math_reduce_small(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]], constant Descriptor &od [[buffer(3)]],
    constant MathParameters &parameters [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
  threadgroup float partial[256];
  const uint count = uint(elementCount(id));
  float value = 0.0f;
  if (parameters.axis && (id.offset & 3l) == 0) {
    device const float4 *vectors = reinterpret_cast<device const float4 *>(input + id.offset);
    float4 values = 0.0f;
    const uint chunks = count / 4u;
    for (uint index = lane; index < chunks; index += 256u)
      values += vectors[index];
    value = ((values.x + values.y) + values.z) + values.w;
    if (lane < count % 4u)
      value += input[id.offset + long(chunks * 4u + lane)];
  } else {
    for (uint index = lane; index < count; index += 256u)
      value += input[parameters.axis ? id.offset + long(index) : elementAddress(index, id)];
  }
  value = cooperativeSum(value, partial, lane, simdLane, simdGroup, simdWidth);
  if (lane == 0) {
    if (parameters.scalar_input) value /= parameters.scalar;
    output[od.offset] = value;
  }
}

kernel void math_mse_reduce(
    device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Descriptor &ad [[buffer(3)]], constant Descriptor &bd [[buffer(4)]],
    constant Descriptor &od [[buffer(5)]],
    constant MathParameters &parameters [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
  threadgroup float partial[256];
  const ulong count = elementCount(ad);
  float value = 0.0f;
  if (parameters.axis == 3 && ((ad.offset | bd.offset) & 3l) == 0 && count <= 0xfffffffful) {
    device const float4 *av = reinterpret_cast<device const float4 *>(a + ad.offset);
    device const float4 *bv = reinterpret_cast<device const float4 *>(b + bd.offset);
    float4 values = 0.0f;
    const uint chunks = uint(count / 4ul);
    for (uint index = lane; index < chunks; index += 256u) {
      const float4 difference = av[index] - bv[index];
      values += difference * difference;
    }
    value = ((values.x + values.y) + values.z) + values.w;
    if (ulong(lane) < count % 4ul) {
      const uint index = chunks * 4u + lane;
      const float difference = a[ad.offset + long(index)] - b[bd.offset + long(index)];
      value += difference * difference;
    }
  } else {
    for (ulong index = lane; index < count; index += 256ul) {
      const float difference =
        a[(parameters.axis & 1) ? ad.offset + long(index) : elementAddress(index, ad)] -
        b[(parameters.axis & 2) ? bd.offset + long(index) : elementAddress(index, bd)];
      value += difference * difference;
    }
  }
  value = cooperativeSum(value, partial, lane, simdLane, simdGroup, simdWidth);
  if (lane == 0) output[od.offset] = value / float(count);
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

kernel void math_mse_backward_reduce(
    device const float *upstream [[buffer(0)]],
    device const float *prediction [[buffer(1)]],
    device const float *target [[buffer(2)]], device float *output [[buffer(3)]],
    constant Descriptor &ud [[buffer(4)]], constant Descriptor &pd [[buffer(5)]],
    constant Descriptor &td [[buffer(6)]], constant Descriptor &od [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
  threadgroup float partial[256];
  const int leading = pd.rank - od.rank;
  ulong remainder = group, base = 0, inputStride = 1, reducedCount = 1;
  for (int axis = pd.rank - 1; axis >= 0; --axis) {
    if (axis < leading || od.shape[axis - leading] != pd.shape[axis]) {
      reducedCount *= ulong(pd.shape[axis]);
    } else {
      base += (remainder % ulong(od.shape[axis - leading])) * inputStride;
      remainder /= ulong(od.shape[axis - leading]);
    }
    inputStride *= ulong(pd.shape[axis]);
  }
  float value = 0.0f;
  for (ulong index = lane; index < reducedCount; index += 256ul) {
    ulong reduced = index, logical = base, stride = 1;
    for (int axis = pd.rank - 1; axis >= 0; --axis) {
      if (axis < leading || od.shape[axis - leading] != pd.shape[axis]) {
        logical += (reduced % ulong(pd.shape[axis])) * stride;
        reduced /= ulong(pd.shape[axis]);
      }
      stride *= ulong(pd.shape[axis]);
    }
    value += prediction[elementAddress(logical, pd)] - target[elementAddress(logical, td)];
  }
  value = cooperativeSum(value, partial, lane, simdLane, simdGroup, simdWidth);
  if (lane == 0) {
    const ulong count = elementCount(pd);
    output[elementAddress(group, od)] = count
      ? upstream[ud.offset] * 2.0f * value / float(count) : 0.0f;
  }
}

long matrixBatchAddress(ulong batch, constant Descriptor &descriptor) {
  long address = descriptor.offset;
  for (int axis = descriptor.rank - 3; axis >= 0; --axis) {
    address += long(batch % ulong(descriptor.shape[axis])) * descriptor.strides[axis];
    batch /= ulong(descriptor.shape[axis]);
  }
  return address;
}

float dotProduct(device const float *a, device const float *b, ulong inner,
    long aBase, long bBase, long aStride, long bStride, threadgroup float *partial,
    uint lane, uint simdLane, uint simdGroup, uint simdWidth) {
  float value = 0.0f;
  if (aStride == 1 && bStride == 1 &&
      aBase % 4 == 0 && bBase % 4 == 0 && inner <= UINT_MAX) {
    const uint chunks = uint(inner / 4);
    device const float4 *av = reinterpret_cast<device const float4 *>(a + aBase);
    device const float4 *bv = reinterpret_cast<device const float4 *>(b + bBase);
    float4 sums = 0.0f;
    for (uint index = lane; index < chunks; index += 256u)
      sums = fma(av[index], bv[index], sums);
    value = (sums.x + sums.y) + (sums.z + sums.w);
    const ulong tail = ulong(chunks) * 4ul + lane;
    if (tail < inner) value += a[aBase + long(tail)] * b[bBase + long(tail)];
  } else {
    for (ulong index = lane; index < inner; index += 256ul)
      value += a[aBase + long(index) * aStride] * b[bBase + long(index) * bStride];
  }
  return cooperativeSum(value, partial, lane, simdLane, simdGroup, simdWidth);
}

kernel void math_dot(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Descriptor &ad [[buffer(3)]],
    constant Descriptor &bd [[buffer(4)]],
    constant Descriptor &od [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
  threadgroup float partial[256];
  const float value = dotProduct(a, b, ulong(ad.shape[ad.rank - 1]),
      matrixBatchAddress(group, ad), matrixBatchAddress(group, bd),
      ad.strides[ad.rank - 1], bd.strides[bd.rank - 2],
      partial, lane, simdLane, simdGroup, simdWidth);
  if (lane == 0) output[matrixBatchAddress(group, od)] = value;
}

struct DotParameters {
  ulong inner;
  long leftOffset, rightOffset, outputOffset, leftStride, rightStride;
};

kernel void math_dot_unbatched(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant DotParameters &p [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
  threadgroup float partial[256];
  const float value = dotProduct(a, b, p.inner, p.leftOffset, p.rightOffset,
      p.leftStride, p.rightStride, partial, lane, simdLane, simdGroup, simdWidth);
  if (lane == 0) output[p.outputOffset] = value;
}

kernel void math_dot_packed(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &inner [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
  threadgroup float partial[256];
  const float value = dotProduct(a, b, ulong(inner), 0, 0, 1, 1,
      partial, lane, simdLane, simdGroup, simdWidth);
  if (lane == 0) output[0] = value;
}

struct PackedVectorMatrixTask {
  device const float *a;
  device const float4 *b;
  device float4 *output;
  uint inner, columns;
};
static_assert(sizeof(PackedVectorMatrixTask) == 32, "Metal row task layout");

kernel void math_vector_matrix_packed(
    constant PackedVectorMatrixTask *tasks [[buffer(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
  constant PackedVectorMatrixTask &task = tasks[group.z];
  const uint columns = task.columns / 4u;
  const uint columnLanes = min(4u, simdWidth);
  const uint column = group.x * columnLanes + simdLane % columnLanes;
  float4 value = 0.0f;
  // Adjacent lanes load adjacent column blocks; the remaining lanes split K.
  if (column < columns) {
    for (uint index = simdLane / columnLanes; index < task.inner; index += simdWidth / columnLanes)
      value = fma(float4(task.a[index]), task.b[index * columns + column], value);
  }
  for (uint delta = columnLanes; delta < simdWidth; delta *= 2u)
    value += simd_shuffle_xor(value, delta);
  if (simdLane < columnLanes && column < columns) task.output[column] = value;
}

struct VectorMatrixParameters {
  ulong inner, columns;
  long inputOffset, matrixOffset, outputOffset;
  long inputStride, matrixRowStride, matrixColumnStride, outputStride;
};

// Share the reduction; unbatched views need only one compact uniform binding.
void vectorMatrix(device const float *a, device const float *b, device float *output,
    ulong inner, ulong columns, long aBase, long bBase, long outputBase,
    long inputStride, long matrixRowStride, long matrixColumnStride, long outputStride,
    uint simdLane, uint simdGroup, uint simdWidth, uint group) {
  // Four adjacent columns reuse each vector load and reduce within one SIMD group.
  const ulong column = (ulong(group) * (256 / simdWidth) + simdGroup) * 4ul;
  const bool packed = matrixColumnStride == 1 &&
      bBase % 4 == 0 && matrixRowStride % 4 == 0;
  float4 value = 0.0f;
  if (column < columns) {
    for (ulong index = simdLane; index < inner; index += simdWidth) {
      const float av = a[aBase + long(index) * inputStride];
      const long row = bBase + long(index) * matrixRowStride;
      float4 bv = 0.0f;
      if (packed && column + 3 < columns) {
        bv = *reinterpret_cast<device const float4 *>(b + row + long(column));
      } else {
        for (uint lane = 0; lane < 4; ++lane)
          if (column + lane < columns)
            bv[lane] = b[row + long(column + lane) * matrixColumnStride];
      }
      value = fma(float4(av), bv, value);
    }
  }
  value = simd_sum(value);
  if (simdLane == 0) {
    for (uint lane = 0; lane < 4; ++lane)
      if (column + lane < columns)
        output[outputBase + long(column + lane) * outputStride] = value[lane];
  }
}

kernel void math_vector_matrix(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Descriptor &ad [[buffer(3)]],
    constant Descriptor &bd [[buffer(4)]],
    constant Descriptor &od [[buffer(5)]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  vectorMatrix(a, b, output, ulong(ad.shape[ad.rank - 1]), ulong(bd.shape[bd.rank - 1]),
      matrixBatchAddress(group.z, ad), matrixBatchAddress(group.z, bd),
      matrixBatchAddress(group.z, od), ad.strides[ad.rank - 1],
      bd.strides[bd.rank - 2], bd.strides[bd.rank - 1], od.strides[od.rank - 1],
      simdLane, simdGroup, simdWidth, group.x);
}

kernel void math_vector_matrix_unbatched(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant VectorMatrixParameters &parameters [[buffer(3)]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  vectorMatrix(a, b, output, parameters.inner, parameters.columns,
      parameters.inputOffset, parameters.matrixOffset, parameters.outputOffset,
      parameters.inputStride, parameters.matrixRowStride, parameters.matrixColumnStride,
      parameters.outputStride, simdLane, simdGroup, simdWidth, group.x);
}

kernel void math_matmul_batched(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Descriptor &ad [[buffer(3)]],
    constant Descriptor &bd [[buffer(4)]],
    constant Descriptor &od [[buffer(5)]],
    uint3 local [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  constexpr uint tile = 16;
  threadgroup float tileA[16][16];
  threadgroup float tileB[16][16];

  const ulong rows = ulong(ad.shape[ad.rank - 2]);
  const ulong inner = ulong(ad.shape[ad.rank - 1]);
  const ulong columns = ulong(bd.shape[bd.rank - 1]);
  const long aBase = matrixBatchAddress(group.z, ad);
  const long bBase = matrixBatchAddress(group.z, bd);
  const long outputBase = matrixBatchAddress(group.z, od);
  const ulong row = ulong(group.y) * tile + local.y * 2;
  const ulong column = ulong(group.x) * tile + local.x * 2;
  float4 accumulator = 0.0f;

  for (ulong base = 0; base < inner; base += tile) {
    // Four output elements per thread reduce barrier participants without changing K order.
    for (uint dy = 0; dy < 2; ++dy) {
      for (uint dx = 0; dx < 2; ++dx) {
        const uint tileRow = local.y * 2 + dy;
        const uint tileColumn = local.x * 2 + dx;
        const ulong aRow = row + dy, aColumn = base + tileColumn;
        const ulong bRow = base + tileRow, bColumn = column + dx;
        tileA[tileRow][tileColumn] =
            (aRow < rows && aColumn < inner)
            ? a[aBase + long(aRow) * ad.strides[ad.rank - 2] +
                long(aColumn) * ad.strides[ad.rank - 1]] : 0.0f;
        tileB[tileRow][tileColumn] =
            (bRow < inner && bColumn < columns)
            ? b[bBase + long(bRow) * bd.strides[bd.rank - 2] +
                long(bColumn) * bd.strides[bd.rank - 1]] : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < tile; ++k) {
      const float2 left = float2(tileA[local.y * 2][k], tileA[local.y * 2 + 1][k]);
      const float2 right = float2(tileB[k][local.x * 2], tileB[k][local.x * 2 + 1]);
      accumulator += float4(left.x * right.x, left.x * right.y,
                            left.y * right.x, left.y * right.y);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint dy = 0; dy < 2; ++dy)
    for (uint dx = 0; dx < 2; ++dx)
      if (row + dy < rows && column + dx < columns)
        output[outputBase + long(row + dy) * od.strides[od.rank - 2] +
          long(column + dx) * od.strides[od.rank - 1]] = accumulator[dy * 2 + dx];
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

kernel void math_product(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]], constant Descriptor &od [[buffer(3)]],
    constant MathParameters &parameters [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup float partial[256];
  const ulong count = elementCount(id);
  const bool small = count <= 65536ul;
  float value = 1.0f;
  const ulong end = small ? count : min(count, (ulong(group) + 1ul) * 256ul);
  for (ulong index = ulong(group) * 256ul + lane; index < end; index += 256ul)
    value *= input[parameters.axis ? id.offset + long(index) : elementAddress(index, id)];
  partial[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride) partial[lane] *= partial[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) output[elementAddress(group, od)] = partial[0];
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
  long base = id.offset;
  for (int axis = od.rank - 1; axis >= 0; --axis) {
    const ulong coordinate = remainder % ulong(od.shape[axis]);
    remainder /= ulong(od.shape[axis]);
    base += long(coordinate) * id.strides[axis];
  }
  const int reducedAxis = parameters.axis;
  const ulong axisLength = ulong(id.shape[reducedAxis]);
  const bool multiply = parameters.operation == 26;
  float value = multiply ? 1.0f : 0.0f;
  for (ulong index = ulong(lane); index < axisLength; index += 256ul) {
    const float item = input[base + long(index) * id.strides[reducedAxis]];
    value = multiply ? value * item : value + item;
  }
  partial[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride)
      partial[lane] = multiply ? partial[lane] * partial[lane + stride] :
          partial[lane] + partial[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    value = partial[0];
    if (parameters.operation == 27) value /= float(axisLength);
    output[elementAddress(outputIndex, od)] = value;
  }
}

long reductionBase(ulong outputIndex, constant Descriptor &id,
                   constant Descriptor &od) {
  long base = id.offset;
  for (int axis = od.rank - 1; axis >= 0; --axis) {
    base += long(outputIndex % ulong(od.shape[axis])) * id.strides[axis];
    outputIndex /= ulong(od.shape[axis]);
  }
  return base;
}

kernel void math_axis_simd(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]], constant Descriptor &od [[buffer(3)]],
    constant MathParameters &parameters [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]],
    uint tid [[thread_position_in_grid]]) {
  const ulong outputIndex = tid / width;
  if (outputIndex >= elementCount(od)) return;
  const long base = reductionBase(outputIndex, id, od);
  const ulong count = ulong(id.shape[parameters.axis]);
  const long stride = id.strides[parameters.axis];
  const bool multiply = parameters.operation == 26;
  float value = multiply ? 1.0f : 0.0f;
  for (ulong index = lane; index < count; index += width) {
    const float item = input[base + long(index) * stride];
    value = multiply ? value * item : value + item;
  }
  if (multiply) {
    for (uint mask = width / 2; mask > 0; mask >>= 1)
      value *= simd_shuffle_xor(value, ushort(mask));
  } else {
    value = simd_sum(value);
  }
  if (lane == 0) {
    if (parameters.operation == 27) value /= float(count);
    output[elementAddress(outputIndex, od)] = value;
  }
}

kernel void math_axis_serial(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]], constant Descriptor &od [[buffer(3)]],
    constant MathParameters &parameters [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  const ulong outputIndex = tid;
  if (outputIndex >= elementCount(od)) return;
  const long base = reductionBase(outputIndex, id, od);
  const ulong count = ulong(id.shape[parameters.axis]);
  const long stride = id.strides[parameters.axis];
  const bool multiply = parameters.operation == 26;
  float value = multiply ? 1.0f : 0.0f;
  for (ulong index = 0; index < count; ++index) {
    const float item = input[base + long(index) * stride];
    value = multiply ? value * item : value + item;
  }
  if (parameters.operation == 27) value /= float(count);
  output[elementAddress(outputIndex, od)] = value;
}

struct PackedReductionTask {
  device const float *input;
  ulong reserved;
  device float *output;
  uint inner, outputs;
};
static_assert(sizeof(PackedReductionTask) == 32, "Metal reduction task layout");

template<int operation, bool columns>
void packedReduction(constant PackedReductionTask &task, uint tid,
                     uint lane, uint width) {
  const uint outputIndex = columns ? tid : tid / width;
  if (outputIndex >= task.outputs) return;
  float value = operation == 26 ? 1.0f : 0.0f;
  const ulong base = columns ? outputIndex : ulong(outputIndex) * task.inner;
  const uint stride = columns ? task.outputs : 1u;
  for (uint index = columns ? 0u : lane; index < task.inner;
       index += columns ? 1u : width) {
    const float item = task.input[base + ulong(index) * stride];
    value = operation == 26 ? value * item : value + item;
  }
  if (!columns) {
    if (operation == 26) {
      for (uint mask = width / 2; mask > 0; mask >>= 1)
        value *= simd_shuffle_xor(value, ushort(mask));
    } else {
      value = simd_sum(value);
    }
  }
  if (columns || lane == 0) {
    if (operation == 27) value /= float(task.inner);
    task.output[outputIndex] = value;
  }
}

#define PACKED_REDUCTION_KERNEL(NAME, OPERATION, COLUMNS) \
kernel void NAME(constant PackedReductionTask *tasks [[buffer(0)]], \
    uint3 tid [[thread_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint width [[threads_per_simdgroup]]) { \
  packedReduction<OPERATION, COLUMNS>(tasks[tid.z], tid.x, lane, width); \
}
PACKED_REDUCTION_KERNEL(math_sum_rows_tasks, 22, false)
PACKED_REDUCTION_KERNEL(math_mean_rows_tasks, 27, false)
PACKED_REDUCTION_KERNEL(math_prod_rows_tasks, 26, false)
PACKED_REDUCTION_KERNEL(math_sum_columns_tasks, 22, true)
PACKED_REDUCTION_KERNEL(math_mean_columns_tasks, 27, true)
PACKED_REDUCTION_KERNEL(math_prod_columns_tasks, 26, true)

template<int operation>
void packedRowReduction4(constant PackedReductionTask &task, uint tid,
                         uint lane, uint width) {
  // Keep each row inside a SIMD subgroup while loading adjacent values together.
  const uint rowLanes = min(8u, width);
  const uint outputIndex = tid / rowLanes;
  if (outputIndex >= task.outputs) return;
  const uint localLane = lane % rowLanes;
  const uint vectors = task.inner / 4u;
  device const packed_float4 *input =
      reinterpret_cast<device const packed_float4 *>(task.input);
  const ulong base = ulong(outputIndex) * vectors;
  float value = operation == 26 ? 1.0f : 0.0f;
  for (uint index = localLane; index < vectors; index += rowLanes) {
    const float4 item = input[base + index];
    const float reduced = operation == 26 ?
        ((item.x * item.y) * item.z) * item.w :
        ((item.x + item.y) + item.z) + item.w;
    value = operation == 26 ? value * reduced : value + reduced;
  }
  for (uint mask = rowLanes / 2u; mask > 0; mask >>= 1u)
    value = operation == 26 ? value * simd_shuffle_xor(value, mask) :
        value + simd_shuffle_xor(value, mask);
  if (localLane == 0) {
    if (operation == 27) value /= float(task.inner);
    task.output[outputIndex] = value;
  }
}

#define PACKED_ROW_REDUCTION4_KERNEL(NAME, OPERATION) \
kernel void NAME(constant PackedReductionTask *tasks [[buffer(0)]], \
    uint3 tid [[thread_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint width [[threads_per_simdgroup]]) { \
  packedRowReduction4<OPERATION>(tasks[tid.z], tid.x, lane, width); \
}
PACKED_ROW_REDUCTION4_KERNEL(math_sum_rows4_tasks, 22)
PACKED_ROW_REDUCTION4_KERNEL(math_prod_rows4_tasks, 26)
PACKED_ROW_REDUCTION4_KERNEL(math_mean_rows4_tasks, 27)

template<int operation>
void packedColumnReduction(constant PackedReductionTask &task, uint group,
                           uint lane, uint width) {
  const uint columnLanes = min(4u, width);
  const uint block = group * columnLanes + lane % columnLanes;
  const uint blocks = task.outputs / 4u;
  float4 value = operation == 26 ? float4(1.0f) : float4(0.0f);
  if (block < blocks) {
    device const float4 *input = reinterpret_cast<device const float4 *>(task.input);
    for (uint row = lane / columnLanes; row < task.inner; row += width / columnLanes) {
      const float4 item = input[ulong(row) * blocks + block];
      value = operation == 26 ? value * item : value + item;
    }
  }
  for (uint delta = columnLanes; delta < width; delta *= 2u)
    value = operation == 26 ? value * simd_shuffle_xor(value, delta) :
        value + simd_shuffle_xor(value, delta);
  if (lane < columnLanes && block < blocks) {
    if (operation == 27) value /= float(task.inner);
    reinterpret_cast<device float4 *>(task.output)[block] = value;
  }
}

#define PACKED_COLUMN_REDUCTION_KERNEL(NAME, OPERATION) \
kernel void NAME(constant PackedReductionTask *tasks [[buffer(0)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint width [[threads_per_simdgroup]]) { \
  packedColumnReduction<OPERATION>(tasks[group.z], group.x, lane, width); \
}
PACKED_COLUMN_REDUCTION_KERNEL(math_sum_columns4_tasks, 22)
PACKED_COLUMN_REDUCTION_KERNEL(math_prod_columns4_tasks, 26)
PACKED_COLUMN_REDUCTION_KERNEL(math_mean_columns4_tasks, 27)

template<bool vectorized>
void packedFullProduct(constant PackedReductionTask &task,
                       uint tid, uint lane, uint simdGroup, uint width,
                       threadgroup float *partial) {
  float value = 1.0f;
  if (vectorized) {
    device const packed_float4 *input =
        reinterpret_cast<device const packed_float4 *>(task.input);
    for (uint index = tid; index < task.inner / 4u; index += 256u) {
      const float4 item = input[index];
      value *= ((item.x * item.y) * item.z) * item.w;
    }
  } else {
    for (uint index = tid; index < task.inner; index += 256u)
      value *= task.input[index];
  }
  for (uint mask = width / 2; mask > 0; mask >>= 1)
    value *= simd_shuffle_xor(value, ushort(mask));
  if (lane == 0) partial[simdGroup] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdGroup == 0) {
    value = 1.0f;
    for (uint index = lane; index < (256u + width - 1u) / width; index += width)
      value *= partial[index];
    for (uint mask = width / 2; mask > 0; mask >>= 1)
      value *= simd_shuffle_xor(value, ushort(mask));
    if (lane == 0) task.output[0] = value;
  }
}

#define PACKED_FULL_PRODUCT_KERNEL(NAME, VECTORIZED) \
kernel void NAME(constant PackedReductionTask *tasks [[buffer(0)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint simdGroup [[simdgroup_index_in_threadgroup]], \
    uint width [[threads_per_simdgroup]]) { \
  threadgroup float partial[256]; \
  packedFullProduct<VECTORIZED>(tasks[group.z], tid, lane, simdGroup, width, partial); \
}
PACKED_FULL_PRODUCT_KERNEL(math_product_tasks, false)
PACKED_FULL_PRODUCT_KERNEL(math_product4_tasks, true)


long scanOuterOffset(ulong outer, constant Descriptor &descriptor,
                       int axis) {
  long offset = 0;
  for (int dim = axis - 1; dim > 0; --dim) {
    const ulong coordinate = outer % ulong(descriptor.shape[dim]);
    outer /= ulong(descriptor.shape[dim]);
    offset += long(coordinate) * descriptor.strides[dim];
  }
  if (axis > 0) offset += long(outer) * descriptor.strides[0];
  return offset;
}

long scanInnerOffset(ulong inner, constant Descriptor &descriptor,
                       int axis) {
  long offset = 0;
  for (int dim = descriptor.rank - 1; dim > axis + 1; --dim) {
    const ulong coordinate = inner % ulong(descriptor.shape[dim]);
    inner /= ulong(descriptor.shape[dim]);
    offset += long(coordinate) * descriptor.strides[dim];
  }
  if (axis + 1 < descriptor.rank)
    offset += long(inner) * descriptor.strides[axis + 1];
  return offset;
}

kernel void cumsum_scalar(device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant Descriptor &od [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid == 0)
    output[od.offset] = as_type<float>(as_type<uint>(input[id.offset]));
}

kernel void cumsum_rows(device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant int &axis [[buffer(3)]],
    constant ulong &axisLength [[buffer(4)]],
    constant ulong &innerCount [[buffer(5)]],
    constant uint &lineGridWidth [[buffer(6)]],
    constant int &reverse [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
  const ulong line = ulong(group.x) + ulong(group.y) * ulong(lineGridWidth);
  const ulong outer = line / innerCount;
  const ulong inner = line % innerCount;
  const long base = id.offset + scanOuterOffset(outer, id, axis) +
      scanInnerOffset(inner, id, axis);
  float carry = 0.0f;

  // Each lane keeps four adjacent prefixes; exclusive carries preserve Inf.
  for (ulong tile = 0; tile < axisLength; tile += ulong(width) * 4ul) {
    const ulong start = tile + ulong(lane) * 4ul;
    float4 prefixes = 0.0f;
    for (uint part = 0; part < 4; ++part) {
      const ulong scan = start + ulong(part);
      if (scan < axisLength) {
        const ulong axisIndex = reverse ? axisLength - 1ul - scan : scan;
        prefixes[part] = input[base + long(axisIndex) * id.strides[axis]];
      }
    }
    prefixes.y += prefixes.x;
    prefixes.z += prefixes.y;
    prefixes.w += prefixes.z;
    const float preceding = simd_prefix_exclusive_sum(prefixes.w);
    const float tileTotal = simd_shuffle(preceding + prefixes.w, ushort(width - 1));
    const float initial = carry + preceding;
    for (uint part = 0; part < 4; ++part) {
      const ulong scan = start + ulong(part);
      if (scan < axisLength) {
        const ulong axisIndex = reverse ? axisLength - 1ul - scan : scan;
        const ulong outputIndex = (outer * axisLength + axisIndex) *
            innerCount + inner;
        output[outputIndex] = initial + prefixes[part];
      }
    }
    carry += tileTotal;
  }
}

kernel void cumsum_columns(device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant int &axis [[buffer(3)]],
    constant ulong &axisLength [[buffer(4)]],
    constant ulong &innerCount [[buffer(5)]],
    constant ulong &columnTileCount [[buffer(6)]],
    constant uint &tileGridWidth [[buffer(7)]],
    constant int &reverse [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint width [[threads_per_simdgroup]]) {
  threadgroup float4 partials[4];
  threadgroup float4 tileTotal;
  const ulong tileId = ulong(group.x) + ulong(group.y) * ulong(tileGridWidth);
  const ulong outer = tileId / columnTileCount;
  const ulong firstColumn = (tileId % columnTileCount) * 4ul;
  const long outerBase = id.offset + scanOuterOffset(outer, id, axis);
  const ulong groupCount = 128ul / ulong(width);
  float4 carry = float4(0.0f);

  for (ulong tile = 0; tile < axisLength; tile += 128ul) {
    const ulong scan = tile + ulong(simdGroup) * ulong(width) + ulong(lane);
    const bool active = scan < axisLength;
    const ulong axisIndex = reverse ? axisLength - 1ul - scan : scan;
    float4 values = float4(0.0f);
    if (active) {
      const long axisBase = outerBase + long(axisIndex) * id.strides[axis];
      for (uint columnLane = 0; columnLane < 4; ++columnLane) {
        const ulong column = firstColumn + ulong(columnLane);
        if (column < innerCount)
          values[columnLane] = input[axisBase +
              scanInnerOffset(column, id, axis)];
      }
    }
    const float4 prefix = float4(
        simd_prefix_inclusive_sum(values.x),
        simd_prefix_inclusive_sum(values.y),
        simd_prefix_inclusive_sum(values.z),
        simd_prefix_inclusive_sum(values.w));
    const float4 subtotal = float4(
        simd_shuffle_down(prefix.x, ushort(width - 1)),
        simd_shuffle_down(prefix.y, ushort(width - 1)),
        simd_shuffle_down(prefix.z, ushort(width - 1)),
        simd_shuffle_down(prefix.w, ushort(width - 1)));
    if (lane == 0) partials[simdGroup] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 preceding = float4(0.0f);
    for (uint previous = 0; previous < simdGroup; ++previous)
      preceding += partials[previous];
    if (simdGroup == 0 && lane == 0) {
      float4 total = float4(0.0f);
      for (ulong index = 0; index < groupCount; ++index)
        total += partials[index];
      tileTotal = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      const float4 accumulated = (carry + preceding) + prefix;
      for (uint columnLane = 0; columnLane < 4; ++columnLane) {
        const ulong column = firstColumn + ulong(columnLane);
        if (column < innerCount) {
          const ulong outputIndex = (outer * axisLength + axisIndex) *
              innerCount + column;
          output[outputIndex] = accumulated[columnLane];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    carry += tileTotal;
  }
}


kernel void cumprod_scalar(device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant Descriptor &od [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid == 0)
    output[od.offset] = as_type<float>(as_type<uint>(input[id.offset]));
}

void cumprodOrderedLine(device const float *input, device float *output,
    ulong axisLength, long base, long axisStride, ulong outputBase,
    ulong innerCount) {
#pragma METAL fp math_mode(safe)
  float carry = 1.0f;
  ulong index = 0;
  // Prefetch across the dependency chain without reassociating its products.
  for (; index + 16 <= axisLength; index += 16) {
    const long source = base + long(index) * axisStride;
    float4 values[4], prefixes[4];
#pragma unroll
    for (uint block = 0; block < 4; ++block) {
      const long first = source + long(block * 4) * axisStride;
      if (axisStride == 1) {
        values[block] = float4(*reinterpret_cast<device const packed_float4 *>(input + first));
      } else {
        values[block] = float4(input[first], input[first + axisStride],
            input[first + 2 * axisStride], input[first + 3 * axisStride]);
      }
    }
#pragma unroll
    for (uint block = 0; block < 4; ++block) {
      carry *= values[block].x; prefixes[block].x = carry;
      carry *= values[block].y; prefixes[block].y = carry;
      carry *= values[block].z; prefixes[block].z = carry;
      carry *= values[block].w; prefixes[block].w = carry;
    }
    const ulong target = outputBase + index * innerCount;
#pragma unroll
    for (uint block = 0; block < 4; ++block) {
      const ulong first = target + ulong(block * 4) * innerCount;
      if (innerCount == 1) {
        *reinterpret_cast<device packed_float4 *>(output + first) = packed_float4(prefixes[block]);
      } else {
#pragma unroll
        for (uint lane = 0; lane < 4; ++lane)
          output[first + ulong(lane) * innerCount] = prefixes[block][lane];
      }
    }
  }
  for (; index < axisLength; ++index) {
    carry *= input[base + long(index) * axisStride];
    output[outputBase + index * innerCount] = carry;
  }
}

kernel void cumprod_lines(device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant int &axis [[buffer(3)]],
    constant ulong &axisLength [[buffer(4)]],
    constant ulong &innerCount [[buffer(5)]],
    constant ulong &lineCount [[buffer(6)]],
    constant uint &gridWidth [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const ulong line = (ulong(group.x) + ulong(group.y) * ulong(gridWidth)) * 8ul + ulong(tid);
  if (line >= lineCount) return;
  const ulong outer = axis == 0 ? 0 : (innerCount == 1 ? line : line / innerCount);
  const ulong inner = axis == 0 ? line : (innerCount == 1 ? 0 : line % innerCount);
  const long base = id.offset + scanOuterOffset(outer, id, axis) +
      scanInnerOffset(inner, id, axis);
  cumprodOrderedLine(input, output, axisLength, base, id.strides[axis],
      outer * axisLength * innerCount + inner, innerCount);
}

void cumprodShortLine(device const float *input, device float *output,
    ulong axisLength, long base, long axisStride, ulong outputBase,
    ulong innerCount, uint lane, uint width) {
#pragma METAL fp math_mode(safe)
  if (axisLength <= ulong(width) * 4ul) {
    const ulong start = ulong(lane) * 4ul;
    float4 values = 1.0f;
    bool normal = true;
    int positive = 0, negative = 0;
    for (uint part = 0; part < 4; ++part) {
      if (start + ulong(part) < axisLength) {
        const float value = input[base + long(start + ulong(part)) * axisStride];
        values[part] = value;
        const uint magnitude = as_type<uint>(value) & 0x7fffffffu;
        const int exponent = int(magnitude >> 23);
        normal &= exponent > 0 && exponent < 255;
        const int lower = exponent - 127;
        const int upper = lower + int((magnitude & 0x7fffffu) != 0);
        positive += max(upper, 0);
        negative += min(lower, 0);
      }
    }
    // Bound every subtree away from overflow/underflow before reassociation.
    if (simd_all(normal) && simd_sum(positive) <= 120 && simd_sum(negative) >= -120) {
      float4 prefixes = values;
      prefixes.y *= prefixes.x;
      prefixes.z *= prefixes.y;
      prefixes.w *= prefixes.z;
      const float preceding = simd_prefix_exclusive_product(prefixes.w);
      for (uint part = 0; part < 4; ++part) {
        if (start + ulong(part) < axisLength)
          output[outputBase + (start + ulong(part)) * innerCount] =
              preceding * prefixes[part];
      }
      return;
    }
  }
  if (lane == 0)
    cumprodOrderedLine(input, output, axisLength, base, axisStride, outputBase, innerCount);
}

kernel void cumprod_short_lines(device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant int &axis [[buffer(3)]],
    constant ulong &axisLength [[buffer(4)]],
    constant ulong &innerCount [[buffer(5)]],
    constant ulong &lineCount [[buffer(6)]],
    constant uint &gridWidth [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
  const ulong line = ulong(group.x) + ulong(group.y) * ulong(gridWidth);
  if (line >= lineCount) return;
  const ulong outer = axis == 0 ? 0 : (innerCount == 1 ? line : line / innerCount);
  const ulong inner = axis == 0 ? line : (innerCount == 1 ? 0 : line % innerCount);
  const long base = id.offset + scanOuterOffset(outer, id, axis) +
      scanInnerOffset(inner, id, axis);
  cumprodShortLine(input, output, axisLength, base, id.strides[axis],
      outer * axisLength * innerCount + inner, innerCount, lane, width);
}

kernel void cumprod_short_tasks(constant PackedReductionTask *tasks [[buffer(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
  constant PackedReductionTask &task = tasks[group.z];
  const ulong line = group.x;
  const ulong innerCount = task.reserved;
  const ulong outer = innerCount == 1 ? line : line / innerCount;
  const ulong inner = innerCount == 1 ? 0 : line % innerCount;
  const ulong base = outer * ulong(task.inner) * innerCount + inner;
  cumprodShortLine(task.input, task.output, task.inner, long(base), long(innerCount),
      base, innerCount, lane, width);
}

kernel void cumprod_gradient_scalar(device const float *seed [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &sd [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid == 0)
    output[0] = as_type<float>(as_type<uint>(seed[sd.offset]));
}

kernel void cumprod_gradient_singleton(device const float *seed [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &sd [[buffer(2)]],
    constant int &axis [[buffer(3)]],
    constant ulong &innerCount [[buffer(4)]],
    constant ulong &lineCount [[buffer(5)]],
    constant uint &gridWidth [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const ulong groupIndex = ulong(group.x) +
      ulong(group.y) * ulong(gridWidth);
  const ulong line = groupIndex * 16ul + ulong(tid);
  if (line >= lineCount) return;

  const ulong outer = line / innerCount;
  const ulong inner = line % innerCount;
  const long seedBase = sd.offset + scanOuterOffset(outer, sd, axis) +
      scanInnerOffset(inner, sd, axis);
  output[line] = as_type<float>(as_type<uint>(seed[seedBase]));
}

kernel void cumprod_gradient_lines(
    device const float *input [[buffer(0)]],
    device const float *prefixes [[buffer(1)]],
    device const float *seed [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Descriptor &id [[buffer(4)]],
    constant Descriptor &pd [[buffer(5)]],
    constant Descriptor &sd [[buffer(6)]],
    constant int &axis [[buffer(7)]],
    constant ulong &axisLength [[buffer(8)]],
    constant ulong &innerCount [[buffer(9)]],
    constant ulong &lineCount [[buffer(10)]],
    constant uint &gridWidth [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
  const ulong groupIndex = ulong(group.x) +
      ulong(group.y) * ulong(gridWidth);
  const ulong line = groupIndex * 16ul + ulong(tid);
  if (line >= lineCount) return;

  const ulong outer = line / innerCount;
  const ulong inner = line % innerCount;
  const long inputBase = id.offset + scanOuterOffset(outer, id, axis) +
      scanInnerOffset(inner, id, axis);
  const long prefixBase = pd.offset + scanOuterOffset(outer, pd, axis) +
      scanInnerOffset(inner, pd, axis);
  const long seedBase = sd.offset + scanOuterOffset(outer, sd, axis) +
      scanInnerOffset(inner, sd, axis);
  const long inputAxisStride = id.strides[axis];
  const long prefixAxisStride = pd.strides[axis];
  const long seedAxisStride = sd.strides[axis];

  ulong firstZero = axisLength;
  for (ulong index = 0; index < axisLength; ++index) {
    if (input[inputBase + long(index) * inputAxisStride] == 0.0f) {
      firstZero = index;
      break;
    }
  }

  if (firstZero == axisLength) {
    float accumulated = 0.0f;
    ulong remaining = axisLength;
    while (remaining > 0) {
      --remaining;
      const float weight = prefixes[prefixBase +
          long(remaining) * prefixAxisStride] *
          seed[seedBase + long(remaining) * seedAxisStride];
      accumulated = accumulated + weight;
      output[(outer * axisLength + remaining) * innerCount + inner] =
          accumulated / input[inputBase + long(remaining) * inputAxisStride];
    }
    return;
  }

  for (ulong index = firstZero; index < axisLength; ++index)
    output[(outer * axisLength + index) * innerCount + inner] = 0.0f;

  float accumulated = 0.0f;
  ulong remaining = firstZero;
  while (remaining > 0) {
    --remaining;
    const float weight = prefixes[prefixBase +
        long(remaining) * prefixAxisStride] *
        seed[seedBase + long(remaining) * seedAxisStride];
    accumulated = accumulated + weight;
    output[(outer * axisLength + remaining) * innerCount + inner] =
        accumulated / input[inputBase + long(remaining) * inputAxisStride];
  }

  float suffixProduct = 1.0f;
  float zeroSum = 0.0f;
  for (ulong index = firstZero; index < axisLength; ++index) {
    if (index > firstZero) {
      const float value = input[inputBase + long(index) * inputAxisStride];
      if (value == 0.0f) {
        const float maskedWeight = suffixProduct * 0.0f;
        zeroSum = zeroSum + maskedWeight;
        break;
      }
      suffixProduct = suffixProduct * value;
    }
    const float weight = suffixProduct *
        seed[seedBase + long(index) * seedAxisStride];
    zeroSum = zeroSum + weight;
  }
  const float beforeZero = firstZero == 0 ? 1.0f :
      prefixes[prefixBase + long(firstZero - 1ul) * prefixAxisStride];
  output[(outer * axisLength + firstZero) * innerCount + inner] =
      zeroSum * beforeZero;
}

long productGradientAxisBase(ulong slice, constant Descriptor &descriptor,
                             int axis) {
  long base = descriptor.offset;
  if (descriptor.rank == 0) return base;
  for (int dim = descriptor.rank - 1; dim >= 0; --dim) {
    if (dim == axis) continue;
    const ulong coordinate = slice % ulong(descriptor.shape[dim]);
    slice /= ulong(descriptor.shape[dim]);
    base += long(coordinate) * descriptor.strides[dim];
  }
  return base;
}

ulong productGradientInnerSize(constant Descriptor &descriptor, int axis) {
  ulong inner = 1;
  for (int dim = descriptor.rank - 1; dim > axis; --dim)
    inner *= ulong(descriptor.shape[dim]);
  return inner;
}

ulong productGradientSlice(ulong index, constant Descriptor &descriptor,
                           int axis) {
  if (descriptor.rank == 0) return 0;
  const ulong length = ulong(descriptor.shape[axis]);
  const ulong inner = productGradientInnerSize(descriptor, axis);
  return (index / (length * inner)) * inner + index % inner;
}

kernel void math_product_gradient_count_full(
    device const float *input [[buffer(0)]],
    device uint *counts [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant ulong &inputCount [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup uint partial[256];
  const ulong start = ulong(group) * 65536ul;
  const ulong end = min(inputCount, start + 65536ul);
  uint zeros = 0;
  for (ulong index = start + ulong(lane); index < end && zeros < 2;
       index += 256ul) {
    if (input[elementAddress(index, id)] == 0.0f) ++zeros;
  }
  partial[lane] = zeros;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride)
      partial[lane] = min(2u, partial[lane] + partial[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) counts[group] = partial[0];
}

kernel void math_product_gradient_finalize_full(
    device uint *counts [[buffer(0)]],
    constant ulong &partialCount [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint partial[256];
  uint zeros = 0;
  for (ulong index = ulong(lane); index < partialCount; index += 256ul)
    zeros = min(2u, zeros + counts[index]);
  partial[lane] = zeros;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride)
      partial[lane] = min(2u, partial[lane] + partial[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) counts[0] = partial[0];
}

kernel void math_product_gradient_count_axis(
    device const float *input [[buffer(0)]],
    device uint *counts [[buffer(1)]],
    constant Descriptor &id [[buffer(2)]],
    constant int &axis [[buffer(3)]],
    constant ulong &axisLength [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup uint partial[256];
  const long base = productGradientAxisBase(ulong(group), id, axis);
  uint zeros = 0;
  for (ulong index = ulong(lane); index < axisLength && zeros < 2;
       index += 256ul) {
    if (input[base + long(index) * id.strides[axis]] == 0.0f) ++zeros;
  }
  partial[lane] = zeros;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride)
      partial[lane] = min(2u, partial[lane] + partial[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) counts[group] = partial[0];
}

void productGradientFullScan(device const float *input,
    device const float *seed, device float *output,
    constant Descriptor &id, constant Descriptor &sd,
    constant Descriptor &od, ulong inputCount) {
  const float seedValue = seed[elementAddress(0, sd)];
  float prefix = 1.0f;
  for (ulong index = 0; index < inputCount; ++index) {
    output[elementAddress(index, od)] = prefix;
    prefix *= input[elementAddress(index, id)];
  }
  float suffix = 1.0f;
  for (ulong remaining = inputCount; remaining > 0; --remaining) {
    const ulong index = remaining - 1;
    const float exclusivePrefix = output[elementAddress(index, od)];
    output[elementAddress(index, od)] =
        exclusivePrefix * suffix * seedValue;
    suffix *= input[elementAddress(index, id)];
  }
}

kernel void math_product_gradient_full(
    device const float *input [[buffer(0)]],
    device const float *product [[buffer(1)]],
    device const float *seed [[buffer(2)]],
    device float *output [[buffer(3)]],
    device const uint *counts [[buffer(4)]],
    constant Descriptor &id [[buffer(5)]],
    constant Descriptor &pd [[buffer(6)]],
    constant Descriptor &sd [[buffer(7)]],
    constant Descriptor &od [[buffer(8)]],
    constant ulong &inputCount [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  const ulong index = ulong(tid);
  if (index >= inputCount) return;
  const uint zeros = counts[0];
  if (zeros == 1) {
    if (index == 0)
      productGradientFullScan(input, seed, output, id, sd, od, inputCount);
    return;
  }
  if (zeros > 1) {
    output[elementAddress(index, od)] = 0.0f;
    return;
  }
  const float productValue = product[elementAddress(0, pd)];
  const float seedValue = seed[elementAddress(0, sd)];
  output[elementAddress(index, od)] =
      (seedValue * productValue) / input[elementAddress(index, id)];
}

void productGradientAxisScan(device const float *input,
    device const float *seed, device float *output,
    constant Descriptor &id, constant Descriptor &sd,
    constant Descriptor &od, ulong slice, int axis, ulong axisLength) {
  const long inputBase = productGradientAxisBase(slice, id, axis);
  const long outputBase = productGradientAxisBase(slice, od, axis);
  const long inputStride = id.rank == 0 ? 0 : id.strides[axis];
  const long outputStride = od.rank == 0 ? 0 : od.strides[axis];
  const float seedValue = seed[elementAddress(slice, sd)];
  float prefix = 1.0f;
  for (ulong index = 0; index < axisLength; ++index) {
    output[outputBase + long(index) * outputStride] = prefix;
    prefix *= input[inputBase + long(index) * inputStride];
  }
  float suffix = 1.0f;
  for (ulong remaining = axisLength; remaining > 0; --remaining) {
    const ulong index = remaining - 1;
    const float exclusivePrefix = output[outputBase + long(index) * outputStride];
    output[outputBase + long(index) * outputStride] =
        exclusivePrefix * suffix * seedValue;
    suffix *= input[inputBase + long(index) * inputStride];
  }
}

kernel void math_product_gradient_axis(
    device const float *input [[buffer(0)]],
    device const float *product [[buffer(1)]],
    device const float *seed [[buffer(2)]],
    device float *output [[buffer(3)]],
    device const uint *counts [[buffer(4)]],
    constant Descriptor &id [[buffer(5)]],
    constant Descriptor &pd [[buffer(6)]],
    constant Descriptor &sd [[buffer(7)]],
    constant Descriptor &od [[buffer(8)]],
    constant ulong &inputCount [[buffer(9)]],
    constant ulong &outputCount [[buffer(10)]],
    constant int &axis [[buffer(11)]],
    constant ulong &axisLength [[buffer(12)]],
    uint3 tid [[thread_position_in_grid]]) {
  if (tid.y == 0) {
    const ulong index = ulong(tid.x);
    if (index >= inputCount) return;
    const ulong slice = productGradientSlice(index, id, axis);
    if (counts[slice] != 0) return;
    const float productValue = product[elementAddress(slice, pd)];
    const float seedValue = seed[elementAddress(slice, sd)];
    output[elementAddress(index, od)] =
        (seedValue * productValue) / input[elementAddress(index, id)];
    return;
  }
  const ulong slice = ulong(tid.x);
  if (slice >= outputCount || counts[slice] == 0) return;
  productGradientAxisScan(input, seed, output, id, sd, od, slice, axis,
                          axisLength);
}

kernel void math_product_gradient_scalar(
    device const float *seed [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant Descriptor &sd [[buffer(2)]],
    constant Descriptor &od [[buffer(3)]]) {
  output[elementAddress(0, od)] = seed[elementAddress(0, sd)];
}

kernel void math_product_gradient_axis_singleton(
    device const float *input [[buffer(0)]],
    device const float *product [[buffer(1)]],
    device const float *seed [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Descriptor &id [[buffer(4)]],
    constant Descriptor &pd [[buffer(5)]],
    constant Descriptor &sd [[buffer(6)]],
    constant Descriptor &od [[buffer(7)]],
    constant ulong &inputCount [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  const ulong index = ulong(tid);
  if (index >= inputCount) return;
  const float inputValue = input[elementAddress(index, id)];
  const float seedValue = seed[elementAddress(index, sd)];
  if (inputValue == 0.0f) {
    output[elementAddress(index, od)] = seedValue;
  } else {
    const float productValue = product[elementAddress(index, pd)];
    output[elementAddress(index, od)] =
        (seedValue * productValue) / inputValue;
  }
}
