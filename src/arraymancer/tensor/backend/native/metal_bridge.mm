// Copyright 2026 Arraymancer contributors. Apache-2.0.
// Provides native shared storage; producers belong to storage, not metadata aliases.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "apple_abi.h"
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <mutex>

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
static uint64_t dtypeWidth(int32_t dtype) {
  static const uint64_t widths[] = {0, 4, 2, 2, 1, 2, 4, 8, 1, 2, 4, 8, 1, 4, 8};
  return dtype > 0 && dtype < 15 ? widths[dtype] : 0;
}
static AMStatus validate(AMBuffer *owner, AMTensorDescriptor d, uint64_t &count) {
  const auto width = dtypeWidth(d.dtype);
  if (!width || d.rank < 0 || d.rank > 8 || d.capacity != owner.capacity)
    return failure(2, "invalid tensor descriptor");
  count = 1;
  for (int i = 0; i < d.rank; ++i) {
    if (d.shape[i] < 0 || (d.shape[i] && count > INT64_MAX / (uint64_t)d.shape[i]))
      return failure(2, "invalid shape or overflow");
    count *= d.shape[i];
  }
  if (!count) return ok();
  __int128 lo = d.offset, hi = d.offset;
  for (int i = 0; i < d.rank; ++i) {
    __int128 span = (__int128)(d.shape[i] - 1) * d.strides[i];
    if (span < -(__int128)(d.capacity / width) || span > (__int128)(d.capacity / width))
      return failure(2, "stride exceeds storage");
    if (span < 0) lo += span; else hi += span;
  }
  if (lo < 0 || hi >= (__int128)(d.capacity / width))
    return failure(2, "view exceeds native storage");
  return ok();
}
AMStatus am_transfer(void *buffer, AMTensorDescriptor d, void *host, int32_t upload) {
  @autoreleasepool {
    if (!buffer) return failure(2, "null storage");
    AMBuffer *owner = (__bridge AMBuffer *)buffer;
    uint64_t count;
    AMStatus status = validate(owner, d, count);
    if (status.code) return status;
    if (!count) return ok();
    if (!host) return failure(2, "null host transfer pointer");
    const uint64_t width = dtypeWidth(d.dtype);
    [owner.producer waitUntilCompleted];
    if (owner.producer.status == MTLCommandBufferStatusError)
      return failure(5, owner.producer.error.localizedDescription.UTF8String);
    auto *native = (char *)owner.buffer.contents;
    auto *cpu = (char *)host;
    for (uint64_t n = 0; n < count; ++n) {
      uint64_t rest = n;
      int64_t index = d.offset;
      for (int axis = d.rank - 1; axis >= 0; --axis) {
        index += (int64_t)(rest % d.shape[axis]) * d.strides[axis];
        rest /= d.shape[axis];
      }
      if (upload) memcpy(native + index * width, cpu + n * width, width);
      else memcpy(cpu + n * width, native + index * width, width);
    }
    AMEvidence evidence = owner.evidence;
    if (upload) {
      evidence.invocation_id = nextInvocation.fetch_add(1);
      evidence.kind = 1;
      evidence.kernel_bits = 0;
    }
    evidence.transfer_bytes += count * width;
    evidence.completed = 1;
    owner.evidence = evidence;
    return ok();
  }
}

@interface AMCompute : NSObject
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@end
@implementation AMCompute
@end

static AMCompute *compute(id<MTLDevice> device, const char *directory, NSError **error) {
  static NSMutableDictionary<NSNumber *, AMCompute *> *contexts = [NSMutableDictionary new];
  @synchronized(contexts) {
    AMCompute *context = contexts[@(device.registryID)];
    if (!context) {
      context = [AMCompute new];
      context.queue = [device newCommandQueue];
      NSURL *url = [NSURL fileURLWithPath:
        [[NSString stringWithUTF8String:directory] stringByAppendingPathComponent:@"arraymancer.metallib"]];
      context.library = [device newLibraryWithURL:url error:error];
      if (!context.library || !context.queue) return nil;
      context.pipelines = [NSMutableDictionary new];
      contexts[@(device.registryID)] = context;
    }
    return context;
  }
}

static id<MTLComputePipelineState> pipelineFor(AMCompute *context, NSString *name, NSError **error) {
  @synchronized(context) {
    id<MTLComputePipelineState> pipeline = context.pipelines[name];
    if (!pipeline) {
      id<MTLFunction> function = [context.library newFunctionWithName:name];
      if (!function) return nil;
      pipeline = [context.queue.device newComputePipelineStateWithFunction:function error:error];
      if (pipeline) context.pipelines[name] = pipeline;
    }
    return pipeline;
  }
}

AMStatus am_copy(void *source, AMTensorDescriptor input, void *destination,
                 AMTensorDescriptor output, const char *directory) {
  @autoreleasepool {
    if (!source || !destination || !directory) return failure(2, "null copy argument");
    AMBuffer *src = (__bridge AMBuffer *)source;
    AMBuffer *dst = (__bridge AMBuffer *)destination;
    if (src == dst || src.buffer.device.registryID != dst.buffer.device.registryID ||
        input.dtype != output.dtype) return failure(2, "copy requires distinct same-device buffers and dtype");
    uint64_t count, outCount;
    AMStatus status = validate(src, input, count);
    if (status.code) return status;
    status = validate(dst, output, outCount);
    if (status.code) return status;
    if (count != outCount || count > UINT32_MAX) return failure(2, "copy element count invalid");
    if (!count) return ok();
    NSError *error = nil;
    AMCompute *context = compute(src.buffer.device, directory, &error);
    if (!context) return failure(6, error ? error.localizedDescription.UTF8String : "Metal queue unavailable");
    NSString *name = [NSString stringWithFormat:@"copy%llu", dtypeWidth(input.dtype) * 8];
    id<MTLComputePipelineState> pipeline = pipelineFor(context, name, &error);
    if (!pipeline) return failure(8, error ? error.localizedDescription.UTF8String : "missing copy kernel");
    [src.producer waitUntilCompleted];
    [dst.producer waitUntilCompleted];
    id<MTLCommandBuffer> command = [context.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) return failure(5, "cannot create Metal copy command");
    command.label = name;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src.buffer offset:0 atIndex:0];
    [encoder setBuffer:dst.buffer offset:0 atIndex:1];
    [encoder setBytes:&input length:sizeof(input) atIndex:2];
    [encoder setBytes:&output length:sizeof(output) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(count, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding];
    dst.producer = command;
    // The command retains buffer resources; this lease additionally retains their owners.
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) { (void)src; (void)dst; }];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError)
      return failure(5, command.error.localizedDescription.UTF8String);
    dst.evidence = {nextInvocation.fetch_add(1), dst.buffer.device.registryID,
      0, count * dtypeWidth(input.dtype), 0, 1, 2, (int32_t)(dtypeWidth(input.dtype) * 8)};
    return ok();
  }
}

AMStatus am_set_element(void *destination, AMTensorDescriptor output,
                        const void *value, const char *directory) {
  @autoreleasepool {
    if (!destination || !value || !directory) return failure(2, "null mutation argument");
    AMBuffer *dst = (__bridge AMBuffer *)destination;
    uint64_t count;
    AMStatus status = validate(dst, output, count);
    if (status.code) return status;
    if (output.rank != 0 || count != 1) return failure(2, "element mutation needs scalar descriptor");
    NSError *error = nil;
    AMCompute *context = compute(dst.buffer.device, directory, &error);
    if (!context) return failure(6, error ? error.localizedDescription.UTF8String : "Metal queue unavailable");
    NSString *name = [NSString stringWithFormat:@"set%llu", dtypeWidth(output.dtype) * 8];
    id<MTLComputePipelineState> pipeline = pipelineFor(context, name, &error);
    if (!pipeline) return failure(8, error ? error.localizedDescription.UTF8String : "missing mutation kernel");
    [dst.producer waitUntilCompleted];
    id<MTLCommandBuffer> command = [context.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) return failure(5, "cannot create Metal mutation command");
    command.label = name;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dst.buffer offset:0 atIndex:0];
    [encoder setBytes:value length:dtypeWidth(output.dtype) atIndex:1];
    [encoder setBytes:&output length:sizeof(output) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    dst.producer = command;
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) { (void)dst; }];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError)
      return failure(5, command.error.localizedDescription.UTF8String);
    AMEvidence evidence = dst.evidence;
    evidence.invocation_id = nextInvocation.fetch_add(1);
    evidence.completed = 1;
    evidence.kind = 2;
    evidence.kernel_bits = -(int32_t)(dtypeWidth(output.dtype) * 8);
    dst.evidence = evidence;
    return ok();
  }
}
