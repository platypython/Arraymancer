// Copyright 2026 Arraymancer contributors. Apache-2.0.
// Provides native shared storage; producers belong to storage, not metadata aliases.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "apple_abi.h"
#include <atomic>
#include <cstring>
#include <cstdlib>

@interface AMBuffer : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, strong) id<MTLCommandBuffer> producer;
@property(nonatomic) AMEvidence evidence;
@property(nonatomic) uint64_t capacity;
@end
@implementation AMBuffer
@end

static AMStatus failure(int32_t code, const char *text) {
  return {code, strdup(text)};
}
static AMStatus ok();
static std::atomic<uint64_t> nextInvocation{1};
AMStatus am_buffer_evidence(void *buffer, AMEvidence *evidence) {
  if (!buffer || !evidence) return failure(2, "null evidence handle");
  AMBuffer *owner = (__bridge AMBuffer *)buffer;
  *evidence = owner.evidence;
  return ok();
}
static AMStatus ok() { return {0, nullptr}; }
void am_message_free(char *message) { free(message); }
void am_retain(void *handle) { if (handle) CFRetain(handle); }
void am_release(void *handle) { if (handle) CFRelease(handle); }
AMStatus am_device_count(int32_t *count) {
  @autoreleasepool {
    if (!count) return failure(2, "null count");
    *count = (int32_t)MTLCopyAllDevices().count;
    return ok();
  }
}
AMStatus am_device(int32_t index, void **device, uint64_t *identity) {
  @autoreleasepool {
    if (!device || !identity) return failure(2, "null device output");
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (index < 0 || (NSUInteger)index >= devices.count)
      return failure(1, "Metal device index unavailable");
    id<MTLDevice> selected = devices[index];
    *identity = selected.registryID;
    *device = (__bridge_retained void *)selected;
    return ok();
  }
}
AMStatus am_buffer(void *device, uint64_t bytes, void **buffer) {
  @autoreleasepool {
    if (!device || !buffer) return failure(2, "null allocation handle");
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    if (bytes > dev.maxBufferLength) return failure(4, "Metal allocation exceeds maxBufferLength");
    AMBuffer *owner = [AMBuffer new];
    owner.evidence = {nextInvocation.fetch_add(1), dev.registryID, 0, 0, 0, 1, 1};
    owner.buffer = [dev newBufferWithLength:MAX(bytes, 1) options:MTLResourceStorageModeShared];
    if (!owner.buffer) return failure(4, "Metal allocation failed");
    owner.capacity = bytes;
    *buffer = (__bridge_retained void *)owner;
    return ok();
  }
}
AMStatus am_transfer(void *buffer, AMTensorDescriptor d, void *host, int32_t upload) {
  @autoreleasepool {
    if (!buffer) return failure(2, "null storage");
    AMBuffer *owner = (__bridge AMBuffer *)buffer;
    if (d.rank < 0 || d.rank > 8 || d.dtype != 1 || d.capacity != owner.capacity)
      return failure(2, "invalid tensor descriptor");
    uint64_t count = 1;
    for (int i = 0; i < d.rank; ++i) {
      if (d.shape[i] < 0) return failure(2, "negative dimension");
      if (d.shape[i] && count > UINT64_MAX / (uint64_t)d.shape[i])
        return failure(2, "shape overflow");
      count *= d.shape[i];
    }
    if (!count) return ok();
    if (!host || d.offset < 0 || count > UINT64_MAX / 4)
      return failure(2, "invalid transfer bounds");
    // Check every logical address before any host memory is modified.
    __int128 lo = d.offset, hi = d.offset;
    for (int i = 0; i < d.rank; ++i) {
      __int128 span = (__int128)(d.shape[i] - 1) * d.strides[i];
      if (span < 0) lo += span; else hi += span;
    }
    if (lo < 0 || hi >= (__int128)(d.capacity / 4))
      return failure(2, "view exceeds native storage");
    [owner.producer waitUntilCompleted];
    if (owner.producer.status == MTLCommandBufferStatusError)
      return failure(5, owner.producer.error.localizedDescription.UTF8String);
    auto *native = (char *)owner.buffer.contents;
    auto *cpu = (char *)host;
    for (uint64_t n = 0; n < count; ++n) {
      uint64_t rest = n;
      int64_t index = d.offset;
      for (int axis = d.rank - 1; axis >= 0; --axis) {
        index += (rest % d.shape[axis]) * d.strides[axis];
        rest /= d.shape[axis];
      }
      if (upload) memcpy(native + index * 4, cpu + n * 4, 4);
      else memcpy(cpu + n * 4, native + index * 4, 4);
    }
    AMEvidence evidence = owner.evidence;
    evidence.invocation_id = nextInvocation.fetch_add(1);
    evidence.transfer_bytes += count * 4;
    evidence.completed = 1;
    owner.evidence = evidence;
    return ok();
  }
}
