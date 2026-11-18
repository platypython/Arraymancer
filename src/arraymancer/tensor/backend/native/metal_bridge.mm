// Copyright 2026 Arraymancer contributors. Apache-2.0.
// Provides native shared storage; producers belong to storage, not metadata aliases.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <dispatch/dispatch.h>
#include "apple_abi.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <deque>
#include <vector>
#include <stdlib.h>
#include <string.h>

@class AMDeviceState;
@class AMCapture;
@class AMGenerator;

@class AMBufferLease;
@class AMPendingUse;
@class AMPendingCommand;
@class AMStream;
// These adapter-private classes have no dynamic overrides or external selectors.
__attribute__((objc_direct_members))
@interface AMBuffer : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) uint64_t gpuAddress;
@property(nonatomic, strong) id<MTLCommandBuffer> producer;
@property(nonatomic) uint64_t producerInvocation;
@property(nonatomic, copy) NSString *producerError;
@property(nonatomic) AMEvidence evidence;
@property(nonatomic) uint64_t capacity;
@property(nonatomic, weak) AMDeviceState *state;
@property(nonatomic) uint64_t externalReferences;
@property(nonatomic) uint64_t inFlightLeases;
@property(nonatomic, strong) NSMutableArray *customLeases;
@property(nonatomic, strong) NSMutableArray<AMPendingUse *> *pendingUses;
@property(nonatomic, strong) NSHashTable<AMBufferLease *> *nativeLeases;
@property(nonatomic) BOOL cached;
@end
@implementation AMBuffer
@end

__attribute__((objc_direct_members))
@interface AMGenerator : NSObject {
 @public
  AMRngState rng;
}
@property(nonatomic, strong) AMDeviceState *deviceState;
@end
@implementation AMGenerator
@end

__attribute__((objc_direct_members))
@interface AMPendingCommand : NSObject {
@public
  // Unlike the claimed flag, this publishes evidence, leases and stream retirement.
  std::atomic<bool> completionPublished;
}
// Synchronization snapshots race completion handlers releasing the command.
@property(atomic, strong) id<MTLCommandBuffer> command;
// Invocation handles stay private; shared batch leases grow only before commit.
@property(nonatomic, strong) NSArray *retainedHandles;
@property(nonatomic, strong) id<MTLComputePipelineState> builtinPipeline;
@property(nonatomic, strong) NSMutableSet<AMBuffer *> *bufferLeases;
// Built-in math retains its one output without allocating an array per dispatch.
@property(nonatomic, strong) AMBuffer *singleOutput;
@property(nonatomic, strong) NSArray<AMBuffer *> *outputs;
@property(nonatomic, copy) NSArray *customLeases;
@property(nonatomic) uint64_t sequence;
@property(nonatomic) uint64_t invocation;
@property(nonatomic) uint64_t materializationBytes;
@property(nonatomic) BOOL finalized;
@property(nonatomic, copy) NSString *submissionError;
@property(nonatomic, strong) AMPendingUse *pendingUse;
// Sealed math records share one completion winner instead of racing per record.
@property(atomic, strong) AMPendingUse *completionUse;
@end
@implementation AMPendingCommand
@end

__attribute__((objc_direct_members))
@interface AMPendingUse : NSObject {
@public
  std::atomic<bool> completionPublished;
  // Borrow identities from the command's strong buffer leases.
  std::vector<AMBuffer *__unsafe_unretained> writes;
}
@property(nonatomic, weak) AMStream *stream;
// Eager commands use one record; batches retain their complete record array.
@property(nonatomic, weak) AMPendingCommand *record;
@property(nonatomic, strong) id<MTLCommandBuffer> command;
@property(nonatomic, weak) NSArray<AMPendingCommand *> *records;
@end
@implementation AMPendingUse
@end

__attribute__((objc_direct_members))
@interface AMBufferLease : NSObject
@property(nonatomic, strong) AMBuffer *owner;
@property(nonatomic, copy) NSArray<AMPendingUse *> *uses;
@property(nonatomic, copy) NSString *savedError;
@property(nonatomic) uint64_t producerCount;
@property(nonatomic) int32_t access;
@property(nonatomic) uint64_t externalReferences;
@property(nonatomic) BOOL active;
@end
@implementation AMBufferLease
@end

__attribute__((objc_direct_members))
@interface AMCapture : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MTLCaptureManager *manager;
@property(nonatomic, copy) NSString *path;
@property(nonatomic) uint64_t externalReferences;
@property(nonatomic) BOOL active;
@end
@implementation AMCapture
@end

static NSObject *captureLock;
static NSObject *customLeaseLock;
static AMCapture *activeCapture;
static void ensureCustomLeaseLock(void);
static BOOL releaseBufferLease(AMBufferLease *lease);

// Matrix/reduction kernels consume inner/output counts; unary kernels consume count/zero.
struct AMPackedMathTask {
  uint64_t a, b, output;
  uint32_t count, columns;
};
static_assert(sizeof(AMPackedMathTask) == 32, "Metal packed task layout");

struct AMMatrixDescriptorEntry {
  NSUInteger rows = 0, columns = 0, rowBytes = 0;
  __strong MPSMatrixDescriptor *descriptor = nil;
};

__attribute__((objc_direct_members))
@interface AMStream : NSObject {
@public
  // Slots preserve admission order, including holes from out-of-order completion.
  std::deque<AMPendingCommand *__strong> pending;
  uint64_t pendingBase;
  AMPackedMathTask packedTasks[32];
  NSUInteger packedTaskCount;
  bool packedRows;
  bool packedRepeatedWrites;
  MTLSize packedGroups, packedThreads;
  // Three operand layouts cover one matrix product without holding buffer storage.
  AMMatrixDescriptorEntry matrixDescriptors[3];
  NSUInteger matrixDescriptorCursor;
}
@property(nonatomic, weak) AMDeviceState *state;
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
// Driver allocation may wait for completion callbacks; keep their monitor free.
@property(nonatomic, strong) NSObject *encodingLock;
@property(nonatomic) uint64_t identity;
@property(nonatomic) uint64_t acceptedSequence;
@property(nonatomic) uint64_t completedSequence;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSString *> *failures;
@property(nonatomic) uint64_t synchronizedSequence;
@property(nonatomic) BOOL lastAcceptedComplete;
@property(nonatomic) BOOL lastAcceptedFailed;
@property(nonatomic, copy) NSString *lastAcceptedError;
@property(nonatomic, strong) NSMutableDictionary<NSString *, MPSKernel *> *matrixKernels;
@property(nonatomic) BOOL aggregateMath;
@property(nonatomic) BOOL foreignQueueExposed;
@property(nonatomic) BOOL encoderConcurrent;
// Admission owns every key until these encoding-only identity sets are cleared.
@property(nonatomic) CFMutableSetRef encoderReads;
@property(nonatomic) CFMutableSetRef encoderWrites;
@property(nonatomic, strong) id<MTLCommandBuffer> batchCommand;
@property(nonatomic, strong) id<MTLComputeCommandEncoder> batchEncoder;
@property(nonatomic, strong) id<MTLComputePipelineState> packedPipeline;
@property(nonatomic, strong) NSMutableArray<AMPendingCommand *> *batchRecords;
@property(nonatomic, strong) AMPendingCommand *batchLeaseRecord;
@property(nonatomic) uint64_t batchBytes;
@property(nonatomic) uint64_t batchStarted;
@property(nonatomic) uint64_t batchGeneration;
@end
@implementation AMStream
- (void)dealloc {
  if (_encoderReads) CFRelease(_encoderReads);
  if (_encoderWrites) CFRelease(_encoderWrites);
}
@end
__attribute__((objc_direct_members))
@interface AMStreamFence : NSObject
@property(nonatomic, strong) AMStream *stream;
@property(nonatomic, copy) NSArray<AMPendingCommand *> *snapshot;
@property(nonatomic) uint64_t sequence;
@property(nonatomic) BOOL completed;
@property(nonatomic) int32_t resultCode;
@property(nonatomic, copy) NSString *resultMessage;
@end
@implementation AMStreamFence
@end

__attribute__((objc_direct_members))
@interface AMCustomLease : NSObject
@property(nonatomic, weak) AMBuffer *owner;
@property(nonatomic) uint64_t byteStart;
@property(nonatomic) uint64_t byteEnd;
@property(nonatomic) uint64_t streamIdentity;
@property(nonatomic) int32_t access;
@end
@implementation AMCustomLease
@end

__attribute__((objc_direct_members))
@interface AMEvent : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLSharedEvent> shared;
@property(nonatomic) uint64_t lastScheduledValue;
@property(nonatomic) uint64_t lastRecordedValue;
@end
@implementation AMEvent
@end

__attribute__((objc_direct_members))
@interface AMDeviceState : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) AMGenerator *defaultGenerator;
@property(nonatomic, strong) AMStream *defaultStream;
@property(nonatomic, strong) NSHashTable<AMStream *> *streams;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSMutableArray<AMBuffer *> *> *bufferCache;
@property(nonatomic, strong) NSMutableArray<id<MTLHeap>> *bufferHeaps;
@property(nonatomic) BOOL supportsSharedHeaps;
@property(nonatomic) BOOL supportsIndirectMath;
@property(nonatomic) uint64_t liveBytes;
@property(nonatomic) uint64_t cachedBytes;
@property(nonatomic) uint64_t inFlightBytes;
@property(nonatomic) uint64_t reservedBytes;
@property(nonatomic) uint64_t hostCopyBytes;
@property(nonatomic) uint64_t materializationBytes;
@property(nonatomic) double memoryFraction;
@end
@implementation AMDeviceState
@end

__attribute__((objc_direct_members))
@interface AMCompute : NSObject
@property(nonatomic, strong) AMDeviceState *state;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@end
@implementation AMCompute
@end
__attribute__((objc_direct_members))
@interface AMLibrary : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, copy) NSString *label;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@end
@implementation AMLibrary
@end

__attribute__((objc_direct_members))
@interface AMKernel : NSObject
@property(nonatomic, strong) AMLibrary *owner;
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property(nonatomic, copy) NSString *name;
@end
@implementation AMKernel
@end
static AMStatus failure(int32_t code, const char *text) {
  return {code, strdup(text ? text : "native Metal operation failed")};
}
static AMStatus ok();
static std::atomic<uint64_t> nextInvocation{1};
static AMStatus ok() { return {0, nullptr}; }
static AMStatus flushMathBatch(AMStream *stream);
static AMStatus flushDeviceMathBatches(id<MTLDevice> device);
static AMStatus flushOwnerMathBatches(AMBuffer *owner);
static uint64_t saturatedAdd(uint64_t left, uint64_t right) {
  return right > UINT64_MAX - left ? UINT64_MAX : left + right;
}

static uint64_t memoryLimitBytes(AMDeviceState *state) {
  if (state.memoryFraction == 0) return UINT64_MAX;
  const long double bytes =
      (long double)state.device.recommendedMaxWorkingSetSize *
      (long double)state.memoryFraction;
  if (bytes >= (long double)UINT64_MAX) return UINT64_MAX;
  return (uint64_t)bytes;
}

// All cache/accounting helpers run under @synchronized(state).
static BOOL removeCachedOwnerLocked(AMDeviceState *state, AMBuffer *owner,
                                    NSUInteger index = NSNotFound) {
  if (!owner.cached) return NO;
  NSNumber *key = @(owner.buffer.length);
  NSMutableArray<AMBuffer *> *bin = state.bufferCache[key];
  if (index == NSNotFound) index = [bin indexOfObjectIdenticalTo:owner];
  if (index == NSNotFound) return NO;
  [bin removeObjectAtIndex:index];
  if (!bin.count) [state.bufferCache removeObjectForKey:key];
  owner.cached = NO;
  state.cachedBytes -= owner.buffer.length;
  return YES;
}

static void cacheOwnerLocked(AMDeviceState *state, AMBuffer *owner) {
  if (owner.cached || owner.externalReferences || owner.inFlightLeases ||
      !owner.buffer) return;
  NSNumber *key = @(owner.buffer.length);
  NSMutableArray<AMBuffer *> *bin = state.bufferCache[key];
  if (!bin) {
    bin = [NSMutableArray new];
    state.bufferCache[key] = bin;
  }
  owner.cached = YES;
  [bin addObject:owner];
  state.cachedBytes = saturatedAdd(state.cachedBytes, owner.buffer.length);
}

static BOOL releaseEmptyHeapsLocked(AMDeviceState *state) {
  BOOL released = NO;
  for (NSUInteger index = state.bufferHeaps.count; index > 0;) {
    id<MTLHeap> heap = state.bufferHeaps[--index];
    if (heap.usedSize) continue;
    state.reservedBytes -= heap.size;
    [state.bufferHeaps removeObjectAtIndex:index];
    released = YES;
  }
  return released;
}

static BOOL evictOneCachedOwnerLocked(AMDeviceState *state) {
  for (NSNumber *key in state.bufferCache.allKeys) {
    NSMutableArray<AMBuffer *> *bin = state.bufferCache[key];
    if (!bin.count) continue;
    AMBuffer *owner = bin.lastObject;
    const uint64_t length = owner.buffer.length;
    id<MTLHeap> heap = owner.buffer.heap;
    [bin removeLastObject];
    if (!bin.count) [state.bufferCache removeObjectForKey:key];
    owner.cached = NO;
    owner.buffer = nil;
    state.cachedBytes -= length;
    if (heap) releaseEmptyHeapsLocked(state);
    else state.reservedBytes -= length;
    return YES;
  }
  return releaseEmptyHeapsLocked(state);
}

static void ensureCaptureLock(void) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{ captureLock = [NSObject new]; });
}

static NSString *stopCaptureLocked(AMCapture *capture) {
  if (!capture || !capture.active || activeCapture != capture)
    return @"Metal capture is not the active capture owned by this handle";
  if (!capture.manager.isCapturing) {
    capture.active = NO;
    activeCapture = nil;
    return @"Metal capture is no longer active";
  }
  AMStatus status = flushDeviceMathBatches(capture.device);
  NSString *batchError = status.code
      ? [NSString stringWithUTF8String:status.message ?: "Metal batch flush failed"] : nil;
  if (status.message) free(status.message);
  @try {
    [capture.manager stopCapture];
  } @catch (NSException *exception) {
    NSString *message = exception.reason ?: @"Metal capture stop failed";
    capture.active = NO;
    activeCapture = nil;
    return message;
  }
  capture.active = NO;
  activeCapture = nil;
  return batchError;
}

void am_message_free(char *message) { free(message); }

void am_retain(void *handle) {
  if (!handle) return;
  id object = (__bridge id)handle;
  if ([object isKindOfClass:AMBuffer.class]) {
    AMBuffer *owner = (AMBuffer *)object;
    AMDeviceState *state = owner.state;
    @synchronized(state) {
      if (owner.cached) {
        (void)removeCachedOwnerLocked(state, owner);
        state.liveBytes = saturatedAdd(state.liveBytes, owner.capacity);
      } else if (owner.externalReferences == 0) {
        state.liveBytes = saturatedAdd(state.liveBytes, owner.capacity);
      }
      if (owner.externalReferences != UINT64_MAX) ++owner.externalReferences;
    }
  } else if ([object isKindOfClass:AMCapture.class]) {
    AMCapture *capture = (AMCapture *)object;
    ensureCaptureLock();
    @synchronized(captureLock) {
      if (capture.externalReferences != UINT64_MAX)
        ++capture.externalReferences;
    }
  } else if ([object isKindOfClass:AMBufferLease.class]) {
    AMBufferLease *lease = (AMBufferLease *)object;
    ensureCustomLeaseLock();
    @synchronized(customLeaseLock) {
      if (lease.externalReferences != UINT64_MAX)
        ++lease.externalReferences;
    }
  }
  CFRetain((__bridge CFTypeRef)object);
}

void am_release(void *handle) {
  if (!handle) return;
  id object = (__bridge id)handle;
  if ([object isKindOfClass:AMBuffer.class]) {
    AMBuffer *owner = (AMBuffer *)object;
    AMDeviceState *state = owner.state;
    @synchronized(state) {
      if (!owner.externalReferences) return;
      --owner.externalReferences;
      if (!owner.externalReferences) {
        state.liveBytes -= owner.capacity;
        cacheOwnerLocked(state, owner);
      }
    }
  } else if ([object isKindOfClass:AMCapture.class]) {
    AMCapture *capture = (AMCapture *)object;
    ensureCaptureLock();
    @synchronized(captureLock) {
      if (!capture.externalReferences) return;
      --capture.externalReferences;
      if (!capture.externalReferences)
        (void)stopCaptureLocked(capture);
    }
  } else if ([object isKindOfClass:AMBufferLease.class]) {
    if (!releaseBufferLease((AMBufferLease *)object)) return;
  }
  CFRelease((__bridge CFTypeRef)object);
}
struct AMThreadState {
  __strong AMStream *stream = nil;
  __strong AMDeviceState *deviceState = nil;
  __strong AMCompute *compute = nil;
  uint64_t registryID = 0;
  __strong AMCompute *pipelineContext = nil;
  __strong NSString *pipelineName = nil;
  __strong id<MTLComputePipelineState> pipeline = nil;
};
// Selection and immutable compiled objects keep ownership until thread exit.
static thread_local AMThreadState threadState;

static AMStream *newStream(AMDeviceState *state, id<MTLCommandQueue> queue) {
  if (!state || !state.device || !queue) return nil;
  AMStream *stream = [AMStream new];
  stream.device = state.device;
  stream.state = state;
  stream.queue = queue;
  stream.encodingLock = [NSObject new];
  stream.identity = (uint64_t)(uintptr_t)(__bridge void *)queue;
  stream->pendingBase = 1;
  stream.failures = [NSMutableDictionary new];
  stream.matrixKernels = [NSMutableDictionary new];
  stream.lastAcceptedComplete = YES;
  return stream;
}

static AMDeviceState *deviceState(id<MTLDevice> device, NSError **error) {
  if (!device) {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    device = devices.firstObject;
  }
  if (!device) return nil;
  // Canonical states stay process-owned; repeated allocations need no global lookup.
  AMDeviceState *cached = threadState.deviceState;
  if (cached && cached.device == device) return cached;

  static NSMutableDictionary<NSNumber *, AMDeviceState *> *states;
  static dispatch_once_t once;
  dispatch_once(&once, ^{ states = [NSMutableDictionary new]; });
  @synchronized(states) {
    NSNumber *key = @(device.registryID);
    AMDeviceState *state = states[key];
    if (state) {
      threadState.deviceState = state;
      return state;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      if (error) *error = [NSError errorWithDomain:@"Arraymancer.Metal"
          code:1 userInfo:@{NSLocalizedDescriptionKey: @"cannot create default Metal command queue"}];
      return nil;
    }
    state = [AMDeviceState new];
    state.device = device;
    state.supportsSharedHeaps = device.hasUnifiedMemory && [device supportsFamily:MTLGPUFamilyApple1];
    state.supportsIndirectMath = device.argumentBuffersSupport == MTLArgumentBuffersTier2;
    state.streams = [NSHashTable weakObjectsHashTable];
    state.bufferCache = [NSMutableDictionary new];
    state.memoryFraction = 1.0;
    state.defaultStream = newStream(state, queue);
    state.defaultStream.aggregateMath = YES;
    [state.streams addObject:state.defaultStream];
    states[key] = state;
    threadState.deviceState = state;
    return state;
  }
}

static AMStatus selectedStream(id<MTLDevice> device, AMDeviceState *state,
                               AMStream *__strong *result) {
  if (!device || !state || !result) return failure(2, "invalid Metal stream request");
  AMStream *selected = threadState.stream;
  if (selected && selected.device.registryID != device.registryID)
    return failure(2, "selected Metal stream belongs to a different device");
  *result = selected ?: state.defaultStream;
  return *result ? ok() : failure(1, "Metal command queue unavailable");
}
static AMGenerator *newGenerator(AMDeviceState *state, uint64_t seed) {
  AMGenerator *generator = [AMGenerator new];
  generator.deviceState = state;
  generator->rng = {1, 0, seed, {0, 0, 0, 0}};
  return generator;
}

AMStatus am_generator(void *deviceHandle, uint64_t seed, int32_t mode,
                      void **out) {
  @autoreleasepool {
    if (!out) return failure(2, "null Metal generator output");
    *out = nullptr;
    if (mode != 0 && mode != 1)
      return failure(2, "Metal generator mode must be explicit or default");
    id<MTLDevice> device = deviceHandle
        ? (__bridge id<MTLDevice>)deviceHandle : MTLCopyAllDevices().firstObject;
    NSError *error = nil;
    AMDeviceState *state = deviceState(device, &error);
    if (!state)
      return failure(1, error.localizedDescription.UTF8String ?: "Metal device unavailable");

    AMGenerator *generator = nil;
    if (mode == 0) {
      generator = newGenerator(state, seed);
    } else {
      @synchronized(state) {
        generator = state.defaultGenerator;
        if (!generator) {
          uint64_t entropySeed = 0;
          arc4random_buf(&entropySeed, sizeof(entropySeed));
          generator = newGenerator(state, entropySeed);
          state.defaultGenerator = generator;
        }
      }
    }
    *out = (__bridge_retained void *)generator;
    return ok();
  }
}

AMStatus am_rng_state(void *generatorHandle, AMRngState *state, int32_t mode) {
  if (!generatorHandle || !state)
    return failure(2, "null Metal RNG state argument");
  if (mode != 0 && mode != 1)
    return failure(2, "Metal RNG state mode must be get or set");
  AMGenerator *generator = (__bridge AMGenerator *)generatorHandle;
  if (mode == 0) {
    @synchronized(generator) { *state = generator->rng; }
    return ok();
  }
  const AMRngState requested = *state;
  if (requested.version != 1 || requested.reserved != 0)
    return failure(6, "incompatible Metal RNG state version or reserved field");
  @synchronized(generator) { generator->rng = requested; }
  return ok();
}

AMStatus am_rng_seed(void *generatorHandle, uint64_t *seed, int32_t entropy) {
  if (!generatorHandle || !seed || (entropy != 0 && entropy != 1))
    return failure(2, "invalid Metal RNG seed argument");
  AMGenerator *generator = (__bridge AMGenerator *)generatorHandle;
  uint64_t nextSeed = entropy ? 0 : *seed;
  @synchronized(generator) {
    if (entropy) arc4random_buf(&nextSeed, sizeof(nextSeed));
    generator->rng = {1, 0, nextSeed, {0, 0, 0, 0}};
    *seed = nextSeed;
  }
  return ok();
}

static bool addRngBlocks(const uint32_t current[4], uint64_t blocks,
                         uint32_t next[4]) {
  uint64_t sum = (uint64_t)current[0] + (uint32_t)blocks;
  next[0] = (uint32_t)sum;
  uint64_t carry = sum >> 32;
  sum = (uint64_t)current[1] + (uint32_t)(blocks >> 32) + carry;
  next[1] = (uint32_t)sum;
  carry = sum >> 32;
  for (int limb = 2; limb < 4; ++limb) {
    sum = (uint64_t)current[limb] + carry;
    next[limb] = (uint32_t)sum;
    carry = sum >> 32;
  }
  return carry == 0;
}


static void prepareBufferProducer(AMBuffer *owner, id<MTLCommandBuffer> command,
                                  uint64_t invocation, AMEvidence evidence);

static void ensureCustomLeaseLock(void);
static void finishStreamCommand(AMStream *stream, AMPendingCommand *record,
                                NSString *submissionError) {
  id<MTLCommandBuffer> command = record.command;
  const uint64_t sequence = record.sequence;
  NSString *message = submissionError ?: (command.status == MTLCommandBufferStatusError
      ? (command.error.localizedDescription ?: @"Metal command failed") : nil);
  @synchronized(stream) {
    if (sequence > stream.completedSequence) stream.completedSequence = sequence;
    if (sequence >= stream->pendingBase) {
      const uint64_t index = sequence - stream->pendingBase;
      if (index < stream->pending.size()) stream->pending[index] = nil;
    }
    while (!stream->pending.empty() && !stream->pending.front()) {
      stream->pending.pop_front();
      ++stream->pendingBase;
    }
    if (sequence == stream.acceptedSequence) {
      stream.lastAcceptedComplete = YES;
      stream.lastAcceptedFailed = message != nil;
      stream.lastAcceptedError = message;
    }
    if (message && sequence > stream.synchronizedSequence)
      stream.failures[@(sequence)] = message;
  }
}

static void finishBufferProducer(AMBuffer *owner, uint64_t invocation,
                                 id<MTLCommandBuffer> command, NSString *submissionError) {
  NSString *message = submissionError ?: (command.status == MTLCommandBufferStatusError
      ? (command.error.localizedDescription ?: @"Metal command failed") : nil);
  @synchronized(owner) {
    if (owner.producerInvocation == invocation) {
      owner.producer = nil;
      owner.producerError = message;
    }
    if (owner.evidence.invocation_id == invocation && owner.evidence.kind == 2) {
      AMEvidence evidence = owner.evidence;
      evidence.completed = message ? 0 : 1;
      owner.evidence = evidence;
    }
  }
}

static void finalizeSingleCommand(AMStream *stream, AMPendingCommand *record) {
  NSString *submissionError = nil;
  @synchronized(record) {
    id<MTLCommandBuffer> command = record.command;
    submissionError = record.submissionError;
    if (record.finalized ||
        (command.status != MTLCommandBufferStatusCompleted &&
         command.status != MTLCommandBufferStatusError &&
         !(submissionError && command.status == MTLCommandBufferStatusNotEnqueued))) return;
    record.finalized = YES;
    AMBuffer *singleOutput = record.singleOutput;
    NSArray<AMBuffer *> *outputs = record.outputs;
    const NSUInteger outputCount = singleOutput ? 1 : outputs.count;
    auto outputAt = [&](NSUInteger index) -> AMBuffer * {
      return singleOutput ?: outputs[index];
    };
    if (command.status == MTLCommandBufferStatusCompleted &&
        record.materializationBytes) {
      for (NSUInteger index = 0; index < outputCount; ++index) {
        AMBuffer *output = outputAt(index);
        @synchronized(output) {
          if (output.evidence.invocation_id == record.invocation) {
            AMEvidence evidence = output.evidence;
            evidence.materialization_bytes = record.materializationBytes;
            output.evidence = evidence;
          }
        }
      }
    }

    AMDeviceState *state = record.bufferLeases.count || record.materializationBytes
        ? stream.state : nil;
    if (record.pendingUse || record.customLeases.count || state) {
      ensureCustomLeaseLock();
      @synchronized(customLeaseLock) {
        for (AMCustomLease *lease in record.customLeases) {
          AMBuffer *owner = lease.owner;
          [owner.customLeases removeObjectIdenticalTo:lease];
        }
        for (AMBuffer *owner in record.bufferLeases)
          [owner.pendingUses removeObjectIdenticalTo:record.pendingUse];
        record.pendingUse = nil;
        record.customLeases = @[];
        if (state) {
          @synchronized(state) {
            for (AMBuffer *owner in record.bufferLeases) {
              if (!owner.inFlightLeases) continue;
              --owner.inFlightLeases;
              if (!owner.inFlightLeases) state.inFlightBytes -= owner.capacity;
              if (!owner.externalReferences && !owner.inFlightLeases)
                cacheOwnerLocked(state, owner);
            }
            if (command.status == MTLCommandBufferStatusCompleted &&
                record.materializationBytes)
              state.materializationBytes = saturatedAdd(
                  state.materializationBytes, record.materializationBytes);
          }
        }
      }
    }
    // A read lease must find either its producer or fully retired ownership.
    for (NSUInteger index = 0; index < outputCount; ++index) {
      AMBuffer *output = outputAt(index);
      finishBufferProducer(output, record.invocation, command, submissionError);
    }
    finishStreamCommand(stream, record, submissionError);
    record.command = nil;
    record.retainedHandles = @[];
    record.builtinPipeline = nil;
    record.bufferLeases = nil;
    record.singleOutput = nil;
    record.outputs = @[];
    record->completionPublished.store(true, std::memory_order_release);
  }
}

static void finalizeCommand(AMStream *stream, AMPendingCommand *record) {
  if (record->completionPublished.load(std::memory_order_acquire)) return;
  AMPendingUse *use = record.completionUse;
  if (!use) {
    finalizeSingleCommand(stream, record);
    return;
  }
  if (use->completionPublished.load(std::memory_order_acquire)) return;
  id<MTLCommandBuffer> command = record.command;
  const MTLCommandBufferStatus status = command.status;
  if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError) {
    // Uncommitted submission failures still publish one record at a time.
    finalizeSingleCommand(stream, record);
    return;
  }
  @synchronized(use) {
    if (use->completionPublished.load(std::memory_order_relaxed)) return;
    // The callback owns this array until every member has published.
    NSArray<AMPendingCommand *> *records = use.records;
    for (AMPendingCommand *member in records) finalizeSingleCommand(stream, member);
    use->completionPublished.store(true, std::memory_order_release);
  }
}

static uint64_t monotonicNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void clearMathAccess(AMStream *stream) {
  if (stream.encoderReads) CFSetRemoveAllValues(stream.encoderReads);
  if (stream.encoderWrites) CFSetRemoveAllValues(stream.encoderWrites);
}

// Batch indirect residency declarations instead of calling the driver per owner.
static void useMathResources(id<MTLComputeCommandEncoder> encoder,
                             CFSetRef owners, MTLResourceUsage usage) {
  if (!owners || !CFSetGetCount(owners)) return;
  struct Group {
    id<MTLComputeCommandEncoder> __unsafe_unretained encoder;
    id<MTLResource> __unsafe_unretained resources[32];
    MTLResourceUsage usage;
    NSUInteger count;
  } group;
  group.encoder = encoder;
  group.usage = usage;
  group.count = 0;
  // Admission already owns these buffers; residency needs only borrowed pointers.
  CFSetApplyFunction(owners, [](const void *value, void *context) {
    Group *group = static_cast<Group *>(context);
    group->resources[group->count++] = ((__bridge AMBuffer *)value).buffer;
    if (group->count == 32) {
      [group->encoder useResources:group->resources count:group->count usage:group->usage];
      group->count = 0;
    }
  }, &group);
  if (group.count) [encoder useResources:group.resources count:group.count usage:usage];
}

// Each independent packed operation owns its grid slice and result.
static void flushPackedMath(AMStream *stream, id<MTLComputeCommandEncoder> encoder) {
  const NSUInteger count = stream->packedTaskCount;
  if (!count) return;
  id<MTLComputePipelineState> pipeline = stream.packedPipeline;
  MTLSize groups = stream->packedGroups;
  const bool ordered = !stream->packedRows && stream->packedRepeatedWrites;
  groups.depth = ordered ? 1 : count;
  if (!stream->packedRows) stream->packedTasks[0].b = ordered ? count : 0;
  MTLSize threads = stream->packedThreads;
  stream->packedTaskCount = 0;
  stream.packedPipeline = nil;
  useMathResources(encoder, stream.encoderReads, MTLResourceUsageRead);
  useMathResources(encoder, stream.encoderWrites, MTLResourceUsageWrite);
  [encoder setComputePipelineState:pipeline];
  [encoder setBytes:stream->packedTasks length:count * sizeof(AMPackedMathTask) atIndex:0];
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

// The encoding monitor excludes producers without blocking driver completion.
static AMStatus flushMathBatch(AMStream *stream) {
  if (!stream) return ok();
  @synchronized(stream.encodingLock) {
    id<MTLCommandBuffer> command = stream.batchCommand;
    if (!command) return ok();
    id<MTLComputeCommandEncoder> encoder = stream.batchEncoder;
    NSArray<AMPendingCommand *> *records = stream.batchRecords;
    AMPendingCommand *leaseRecord = stream.batchLeaseRecord;
    AMPendingCommand *lastRecord = records.lastObject;
    if (leaseRecord && leaseRecord != lastRecord) {
      // Keep one owner while encoding; release only after every invocation publishes.
      lastRecord.bufferLeases = leaseRecord.bufferLeases;
      lastRecord.pendingUse = leaseRecord.pendingUse;
      leaseRecord.bufferLeases = nil;
      leaseRecord.pendingUse = nil;
    }
    AMPendingUse *completionUse = lastRecord.pendingUse;
    for (AMPendingCommand *record in records) record.completionUse = completionUse;
    stream.batchEncoder = nil;
    stream.batchCommand = nil;
    stream.batchRecords = nil;
    stream.batchLeaseRecord = nil;
    stream.batchBytes = 0;
    @try {
      flushPackedMath(stream, encoder);
      clearMathAccess(stream);
      [encoder endEncoding];
      if (!records.count) return ok();
      [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        @autoreleasepool {
          (void)completed;
          for (AMPendingCommand *record in records) finalizeCommand(stream, record);
          for (AMPendingCommand *record in records) record.completionUse = nil;
        }
      }];
      [command commit];
      return ok();
    } @catch (NSException *exception) {
      clearMathAccess(stream);
      NSString *message = exception.reason ?: @"Metal math batch submission failed";
      for (AMPendingCommand *record in records) {
        @synchronized(record) { record.submissionError = message; }
        finalizeCommand(stream, record);
        @synchronized(stream) {
          if (record.sequence > stream.synchronizedSequence)
            stream.failures[@(record.sequence)] = message;
        }
      }
      for (AMPendingCommand *record in records) record.completionUse = nil;
      @synchronized(stream) {
        stream.lastAcceptedFailed = YES;
        stream.lastAcceptedError = message;
      }
      return failure(5, message.UTF8String);
    }
  }
}

static id<MTLCommandBuffer> mathCommandBuffer(AMStream *stream) {
  if (!stream.batchCommand) {
    // Admission retains buffers and pipelines through completion; avoid duplicate driver leases.
    stream.batchCommand = [stream.queue commandBufferWithUnretainedReferences];
    if (!stream.batchCommand) return nil;
    stream.batchCommand.label = @"arraymancer.math.batch";
    stream.batchRecords = [[NSMutableArray alloc] initWithCapacity:32];
    stream.batchBytes = 0;
    stream.batchStarted = 0;
    ++stream.batchGeneration;
  }
  return stream.batchCommand;
}

static AMStatus finishMathEncoding(AMStream *stream) {
  const NSUInteger count = stream.batchRecords.count;
  if (!stream.aggregateMath || count >= 32 || stream.batchBytes >= 4 * 1024 * 1024)
    return flushMathBatch(stream);
  constexpr uint64_t deadlineNanoseconds = 100000;
  if (monotonicNanoseconds() - stream.batchStarted >= deadlineNanoseconds)
    return flushMathBatch(stream);
  if (count == 1) {
    const uint64_t generation = stream.batchGeneration;
    const uint64_t elapsed = monotonicNanoseconds() - stream.batchStarted;
    const int64_t delay = elapsed < deadlineNanoseconds ? deadlineNanoseconds - elapsed : 0;
    __weak AMStream *weakStream = stream;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, delay),
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      @autoreleasepool {
        AMStream *current = weakStream;
        @synchronized(current.encodingLock) {
          if (current.batchGeneration == generation) {
            AMStatus status = flushMathBatch(current);
            if (status.message) free(status.message);
          }
        }
      }
    });
  }
  return ok();
}
static id<MTLComputeCommandEncoder> mathComputeEncoder(
    AMStream *stream, id<MTLCommandBuffer> command, AMBuffer *output,
    bool favorSerial) {
  if (favorSerial && stream.encoderConcurrent && stream.batchEncoder) {
    flushPackedMath(stream, stream.batchEncoder);
    [stream.batchEncoder endEncoding];
    stream.batchEncoder = nil;
  }
  if (!stream.batchEncoder) {
    BOOL concurrent = NO;
    if (stream.aggregateMath && !favorSerial) {
      // Reused destinations favor serial ordering over one barrier per write.
      @synchronized(output) { concurrent = output.producerInvocation == 0; }
    }
    stream.encoderConcurrent = concurrent;
    stream.batchEncoder = [command computeCommandEncoderWithDispatchType:
        concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    clearMathAccess(stream);
  }
  return stream.batchEncoder;
}

static void orderMathAccess(AMStream *stream, AMBuffer *output,
                             AMBuffer *a, AMBuffer *b = nil, AMBuffer *c = nil,
                             bool indirect = false, bool orderedWrites = false) {
  if (!stream.encoderConcurrent && (!indirect || !stream.aggregateMath)) {
    if (indirect) {
      for (AMBuffer *input : {a, b, c})
        if (input) [stream.batchEncoder useResource:input.buffer usage:MTLResourceUsageRead];
      [stream.batchEncoder useResource:output.buffer usage:MTLResourceUsageWrite];
    }
    return;
  }
  if (!stream.encoderReads)
    stream.encoderReads = CFSetCreateMutable(kCFAllocatorDefault, 32, nullptr);
  if (!stream.encoderWrites)
    stream.encoderWrites = CFSetCreateMutable(kCFAllocatorDefault, 32, nullptr);
  AMBuffer *__unsafe_unretained inputs[] = {a, b, c};
  // Unary packet threads preserve element ownership across repeated writes.
  BOOL conflict = CFSetContainsValue(stream.encoderReads, (__bridge const void *)output) ||
      (!orderedWrites && CFSetContainsValue(stream.encoderWrites, (__bridge const void *)output));
  for (AMBuffer *input : inputs)
    if (input && CFSetContainsValue(stream.encoderWrites, (__bridge const void *)input)) conflict = YES;
  if (conflict) {
    flushPackedMath(stream, stream.batchEncoder);
    // Concurrent passes need explicit read/write and write/write ordering.
    if (stream.encoderConcurrent)
      [stream.batchEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    clearMathAccess(stream);
  }
  for (AMBuffer *input : inputs) {
    if (!input) continue;
    CFSetAddValue(stream.encoderReads, (__bridge const void *)input);
  }
  CFSetAddValue(stream.encoderWrites, (__bridge const void *)output);
}


static AMStatus flushDeviceMathBatches(id<MTLDevice> device) {
  AMDeviceState *state = deviceState(device, nil);
  if (!state) return failure(1, "Metal device unavailable");
  NSArray<AMStream *> *streams = nil;
  @synchronized(state) { streams = state.streams.allObjects; }
  for (AMStream *stream in streams) {
    AMStatus status = flushMathBatch(stream);
    if (status.code) return status;
  }
  return ok();
}

static AMStatus flushOwnerMathBatches(AMBuffer *owner) {
  ensureCustomLeaseLock();
  NSMutableSet<AMStream *> *streams = [NSMutableSet new];
  @synchronized(customLeaseLock) {
    for (AMPendingUse *use in owner.pendingUses) {
      AMStream *stream = use.stream;
      if (stream) [streams addObject:stream];
    }
  }
  for (AMStream *stream in streams) {
    AMStatus status = flushMathBatch(stream);
    if (status.code) return status;
  }
  return ok();
}

static NSArray<AMBuffer *> *uniqueBufferOwners(NSArray *handles) {
  NSMutableArray<AMBuffer *> *owners = [NSMutableArray new];
  for (id handle in handles) {
    if (![handle isKindOfClass:AMBuffer.class]) continue;
    AMBuffer *owner = (AMBuffer *)handle;
    BOOL found = NO;
    for (AMBuffer *previous in owners) {
      if (previous == owner) { found = YES; break; }
    }
    if (!found) [owners addObject:owner];
  }
  return owners;
}

static void ensureCustomLeaseLock(void) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{ customLeaseLock = [NSObject new]; });
}
static BOOL releaseBufferLease(AMBufferLease *lease) {
  ensureCustomLeaseLock();
  @synchronized(customLeaseLock) {
    if (!lease.externalReferences) return NO;
    --lease.externalReferences;
    if (lease.externalReferences) return YES;
    if (!lease.active) return YES;
    lease.active = NO;
    AMBuffer *owner = lease.owner;
    [owner.nativeLeases removeObject:lease];
    AMDeviceState *state = owner.state;
    @synchronized(state) {
      if (owner.inFlightLeases) {
        --owner.inFlightLeases;
        if (!owner.inFlightLeases)
          state.inFlightBytes -= owner.capacity;
      }
      if (!owner.externalReferences && !owner.inFlightLeases)
        cacheOwnerLocked(state, owner);
    }
    lease.uses = @[];
    lease.owner = nil;
  }
  return YES;
}

static AMStatus submitCommandDetailed(
    AMStream *stream, id<MTLCommandBuffer> command, NSArray *leases,
    NSArray<AMBuffer *> *outputs, NSArray<AMCustomLease *> *customLeases,
    uint64_t invocation, AMEvidence evidence, uint64_t materializationBytes,
    uint64_t *submittedSequence, bool deferred = false,
    id<MTLComputePipelineState> builtinPipeline = nil,
    AMBuffer *__unsafe_unretained const *mathBuffers = nullptr,
    NSUInteger mathBufferCount = 0, AMBuffer *singleOutput = nil) {
  if (!stream || !command) return failure(5, "cannot submit Metal command");
  NSArray<AMBuffer *> *retainedOutputs = singleOutput ? nil :
      (outputs.count <= 1 ? (outputs ?: @[]) : uniqueBufferOwners(outputs));
  const NSUInteger outputCount = singleOutput ? 1 : retainedOutputs.count;
  auto outputAt = [&](NSUInteger index) -> AMBuffer * {
    return singleOutput ?: retainedOutputs[index];
  };
  NSArray *ownedHandles = leases ?: @[];
  NSMutableArray *retainedHandles = nil;
  if (!mathBuffers) {
    for (NSUInteger index = 0; index < outputCount; ++index) {
      AMBuffer *output = outputAt(index);
      if ([ownedHandles containsObject:output]) continue;
      if (!retainedHandles) {
        retainedHandles = [ownedHandles mutableCopy];
        ownedHandles = retainedHandles;
      }
      [retainedHandles addObject:output];
    }
  }
  // Stack lists stay borrowed only until shared ownership records every buffer.
  NSArray<AMBuffer *> *bufferLeases = mathBuffers ? nil : uniqueBufferOwners(ownedHandles);
  const NSUInteger bufferCount = mathBuffers ? mathBufferCount : bufferLeases.count;
  auto bufferAt = [&](NSUInteger index) -> AMBuffer * {
    return mathBuffers ? mathBuffers[index] : bufferLeases[index];
  };
  AMDeviceState *state = stream.state;
  if (!state) return failure(1, "Metal device accounting state unavailable");
  for (NSUInteger index = 0; index < bufferCount; ++index) {
    AMBuffer *owner = bufferAt(index);
    // Both owners bind their canonical device accounting state at creation.
    if (owner.state != state || !owner.buffer)
      return failure(2, "Metal command buffers belong to different devices");
  }

  for (AMCustomLease *lease in customLeases) {
    if (!lease.owner || ![bufferLeases containsObject:lease.owner] ||
        lease.byteEnd < lease.byteStart || lease.access < 0 ||
        lease.access > 2 || lease.streamIdentity != stream.identity)
      return failure(2, "invalid custom Metal buffer lease");
  }

  AMPendingCommand *record = [AMPendingCommand new];
  record->completionPublished.store(false, std::memory_order_relaxed);
  record.command = command;
  record.retainedHandles = ownedHandles;
  record.builtinPipeline = builtinPipeline;
  record.singleOutput = singleOutput;
  record.outputs = retainedOutputs;
  record.customLeases = [customLeases copy] ?: @[];
  record.invocation = invocation;
  record.materializationBytes = materializationBytes;

  ensureCustomLeaseLock();
  @synchronized(stream.encodingLock) {
    if (!deferred) {
      AMStatus status = flushMathBatch(stream);
      if (status.code) return status;
    } else if (command != stream.batchCommand) {
      return failure(5, "Metal math command does not belong to the open batch");
    }
    if (stream.acceptedSequence == UINT64_MAX)
      return failure(5, "Metal stream command sequence exhausted");
    @synchronized(customLeaseLock) {
      @synchronized(state) {
        for (NSUInteger index = 0; index < bufferCount; ++index) {
          AMBuffer *owner = bufferAt(index);
          if (!owner.externalReferences || owner.cached ||
              owner.inFlightLeases == UINT64_MAX)
            return failure(2, "Metal command cannot lease an unowned buffer");
        }
      }

      for (NSUInteger index = 0; index < bufferCount; ++index) {
        AMBuffer *owner = bufferAt(index);
        const BOOL writes = owner == singleOutput || [retainedOutputs containsObject:owner];
        const int32_t access = writes ? 1 : 0;
        if (owner.nativeLeases.count) {
          for (AMBufferLease *active in owner.nativeLeases.allObjects) {
            if (!active.active || (active.access == 0 && access == 0))
              continue;
            return failure(10, "Metal buffer has a conflicting Core AI borrow lease");
          }
        }
        for (AMCustomLease *active in owner.customLeases) {
          if (active.streamIdentity == stream.identity ||
              (active.access == 0 && access == 0))
            continue;
          return failure(10, "Metal buffer has a conflicting queued custom-kernel lease");
        }
      }

      for (AMCustomLease *lease in customLeases) {
        AMBuffer *owner = lease.owner;
        for (AMCustomLease *previous in owner.customLeases) {
          if (previous.streamIdentity == lease.streamIdentity ||
              previous.byteStart >= lease.byteEnd ||
              lease.byteStart >= previous.byteEnd ||
              (previous.access == 0 && lease.access == 0))
            continue;
          return failure(10, "custom Metal buffer has a conflicting queued lease on another stream");
        }
      }

      // A GPU command needs one ownership lease per buffer, not per invocation.
      // Every invocation still publishes its own producer evidence and failure.
      AMPendingCommand *leaseRecord = deferred ? stream.batchLeaseRecord : nil;
      if (!leaseRecord) leaseRecord = record;
      AMPendingUse *use = leaseRecord.pendingUse;
      if (!use && bufferCount) {
        use = [AMPendingUse new];
        use->completionPublished.store(false, std::memory_order_relaxed);
        use.stream = stream;
        if (!deferred) use.record = leaseRecord;
        use.command = command;
        use.records = deferred ? stream.batchRecords : nil;
        use->writes.reserve(deferred && stream.aggregateMath ? 32 : outputCount);
        leaseRecord.pendingUse = use;
        const NSUInteger spareOutputs = deferred && stream.aggregateMath ? 31 : 0;
        leaseRecord.bufferLeases =
            [[NSMutableSet alloc] initWithCapacity:bufferCount + spareOutputs];
      }
      @synchronized(state) {
        for (NSUInteger index = 0; index < bufferCount; ++index) {
          AMBuffer *owner = bufferAt(index);
          if ([leaseRecord.bufferLeases containsObject:owner]) continue;
          if (!owner.inFlightLeases)
            state.inFlightBytes = saturatedAdd(state.inFlightBytes, owner.capacity);
          ++owner.inFlightLeases;
          [leaseRecord.bufferLeases addObject:owner];
          if (!owner.pendingUses) owner.pendingUses = [NSMutableArray new];
          [owner.pendingUses addObject:use];
        }
      }
      for (NSUInteger index = 0; index < outputCount; ++index) {
        AMBuffer *output = outputAt(index);
        const bool firstWrite = deferred &&
            std::find(use->writes.begin(), use->writes.end(), output) == use->writes.end();
        use->writes.push_back(output);
        if (firstWrite)
          stream.batchBytes = saturatedAdd(stream.batchBytes, output.capacity);
      }
      if (deferred && !stream.batchLeaseRecord) stream.batchLeaseRecord = leaseRecord;

      for (AMCustomLease *lease in customLeases) {
        if (!lease.owner.customLeases)
          lease.owner.customLeases = [NSMutableArray new];
        [lease.owner.customLeases addObject:lease];
      }

      uint64_t sequence = 0;
      @synchronized(stream) {
        sequence = ++stream.acceptedSequence;
        record.sequence = sequence;
        for (NSUInteger index = 0; index < outputCount; ++index)
          prepareBufferProducer(outputAt(index), command, invocation, evidence);
        stream.lastAcceptedComplete = NO;
        stream.lastAcceptedFailed = NO;
        stream.lastAcceptedError = nil;
        stream->pending.push_back(record);
      }
      if (deferred) {
        // Charge the deadline from first admission, not encoder/wrapper preparation.
        if (!stream.batchRecords.count) stream.batchStarted = monotonicNanoseconds();
        [stream.batchRecords addObject:record];
      } else {
        [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
          @autoreleasepool {
            (void)completed;
            finalizeCommand(stream, record);
          }
        }];
        [command commit];
      }
      if (submittedSequence) *submittedSequence = sequence;
    }
  }
  return ok();
}

static AMStatus admitMathCommand(AMStream *stream, id<MTLCommandBuffer> command,
                                 NSArray *leases, AMBuffer *output, AMEvidence evidence,
                                 id<MTLComputePipelineState> builtinPipeline = nil,
                                 AMBuffer *__unsafe_unretained const *mathBuffers = nullptr,
                                 NSUInteger mathBufferCount = 0,
                                 uint64_t materializationBytes = 0) {
  AMStatus status = submitCommandDetailed(stream, command, leases, nil, @[],
      evidence.invocation_id, evidence, materializationBytes, nullptr, true, builtinPipeline,
      mathBuffers, mathBufferCount, output);
  if (status.code && !stream.batchRecords.count) {
    AMStatus cleanup = flushMathBatch(stream);
    if (cleanup.message) free(cleanup.message);
  }
  return status;
}

static AMStatus failMathEncoding(AMStream *stream, uint64_t invocation, NSString *message,
                                 int32_t code = 5) {
  message = message ?: @"Metal math encoding failed";
  AMPendingCommand *record = stream.batchRecords.lastObject;
  if (record.invocation == invocation)
    @synchronized(record) { record.submissionError = message; }
  AMStatus status = flushMathBatch(stream);
  if (status.message) free(status.message);
  return failure(code, message.UTF8String);
}

static AMStatus submitCommand(AMStream *stream, id<MTLCommandBuffer> command,
                              NSArray *leases, AMBuffer *output,
                              uint64_t invocation, AMEvidence evidence,
                              uint64_t materializationBytes,
                              uint64_t *submittedSequence) {
  NSArray<AMBuffer *> *outputs = output ? @[output] : @[];
  return submitCommandDetailed(stream, command, leases, outputs, @[],
                               invocation, evidence, materializationBytes,
                               submittedSequence);
}

static void prepareBufferProducer(AMBuffer *owner, id<MTLCommandBuffer> command,
                                  uint64_t invocation, AMEvidence evidence) {
  @synchronized(owner) {
    owner.producer = command;
    owner.producerInvocation = invocation;
    owner.producerError = nil;
    owner.evidence = evidence;
  }
}

static void abandonBufferProducer(AMBuffer *owner, uint64_t invocation) {
  @synchronized(owner) {
    if (owner.producerInvocation == invocation) {
      owner.producer = nil;
      owner.producerError = @"Metal command was not submitted";
      if (owner.evidence.invocation_id == invocation) {
        AMEvidence evidence = owner.evidence;
        evidence.completed = 0;
        owner.evidence = evidence;
      }
    }
  }
}

static AMStatus streamFailureThrough(AMStream *stream, uint64_t target) {
  NSMutableArray<NSString *> *messages = nil;
  NSMutableArray<NSNumber *> *handled = nil;
  @synchronized(stream) {
    for (NSNumber *key in stream.failures) {
      if (key.unsignedLongLongValue > target) continue;
      if (!messages) {
        messages = [NSMutableArray new];
        handled = [NSMutableArray new];
      }
      [messages addObject:[NSString stringWithFormat:@"Metal command %llu failed: %@",
        key.unsignedLongLongValue, stream.failures[key]]];
      [handled addObject:key];
    }
    if (handled) [stream.failures removeObjectsForKeys:handled];
    if (target > stream.synchronizedSequence)
      stream.synchronizedSequence = target;
  }
  return messages
      ? failure(5, [[messages componentsJoinedByString:@"; "] UTF8String]) : ok();
}

static NSArray<AMPendingCommand *> *pendingThrough(AMStream *stream,
                                                   uint64_t target) {
  NSMutableArray<AMPendingCommand *> *snapshot = [NSMutableArray new];
  @synchronized(stream) {
    for (AMPendingCommand *__unsafe_unretained record : stream->pending) {
      if (!record) continue;
      if (record.sequence > target) break;
      [snapshot addObject:record];
    }
  }
  return snapshot;
}

static void waitAndFinalize(AMStream *stream,
                            NSArray<AMPendingCommand *> *snapshot) {
  AMStatus status = flushMathBatch(stream);
  if (status.message) free(status.message);
  for (AMPendingCommand *record in snapshot) {
    if (record->completionPublished.load(std::memory_order_acquire)) continue;
    [record.command waitUntilCompleted];
    finalizeCommand(stream, record);
  }
}

static bool bufferAccessConflicts(int32_t left, int32_t right) {
  return left != 0 || right != 0;
}

static AMStatus beginBufferLease(AMBuffer *owner, int32_t access,
                                 AMBufferLease **out) {
  if (!owner || !out || access < 0 || access > 2)
    return failure(2, "invalid Metal buffer lease argument");
  *out = nil;
  ensureCustomLeaseLock();
  AMBufferLease *lease = [AMBufferLease new];
  if (!lease) return failure(7, "cannot allocate Metal buffer lease");

  @synchronized(customLeaseLock) {
    AMDeviceState *state = owner.state;
    if (!state || !owner.buffer)
      return failure(2, "invalid Metal buffer lease owner");
    @synchronized(state) {
      if (owner.cached || !owner.externalReferences)
        return failure(2, "Metal buffer lease owner is not externally retained");
      if (owner.inFlightLeases == UINT64_MAX)
        return failure(2, "Metal buffer lease count exhausted");
    }
    if (!owner.nativeLeases)
      owner.nativeLeases = [NSHashTable weakObjectsHashTable];
    for (AMBufferLease *active in owner.nativeLeases.allObjects)
      if (active.active && bufferAccessConflicts(access, active.access))
        return failure(10, "Metal buffer has a conflicting active borrow lease");

    NSMutableArray<AMPendingUse *> *snapshot = [NSMutableArray new];
    NSString *savedError = nil;
    @synchronized(owner) {
      savedError = owner.producerError;
      if (access == 0) {
        id<MTLCommandBuffer> producer = owner.producer;
        if (producer) {
          for (AMPendingUse *use in owner.pendingUses) {
            if (use.command == producer) {
              [snapshot addObject:use];
              break;
            }
          }
        }
      } else {
        [snapshot addObjectsFromArray:owner.pendingUses];
      }
    }

    @synchronized(state) {
      if (owner.cached || !owner.externalReferences)
        return failure(2, "Metal buffer lease owner is no longer retained");
      if (owner.inFlightLeases == UINT64_MAX)
        return failure(2, "Metal buffer lease count exhausted");
      if (!owner.inFlightLeases)
        state.inFlightBytes = saturatedAdd(state.inFlightBytes, owner.capacity);
      ++owner.inFlightLeases;
    }
    lease.owner = owner;
    lease.access = access;
    lease.savedError = savedError;
    lease.uses = [snapshot copy];
    uint64_t producerCount = 0;
    for (AMPendingUse *use in lease.uses) {
      if (access == 0)
        producerCount += std::find(use->writes.begin(), use->writes.end(), owner) != use->writes.end();
      else
        producerCount += std::count(use->writes.begin(), use->writes.end(), owner);
    }
    lease.producerCount = producerCount;
    lease.active = YES;
    lease.externalReferences = 1;
    [owner.nativeLeases addObject:lease];
  }
  *out = lease;
  return ok();
}

static AMStatus waitBufferLease(AMBufferLease *lease) {
  if (!lease) return failure(2, "null Metal buffer lease");
  for (AMPendingUse *use in lease.uses) {
    id<MTLCommandBuffer> command = use.command;
    AMStatus status = flushMathBatch(use.stream);
    if (status.code) return status;
    if (!command) continue;
    if (command.status == MTLCommandBufferStatusNotEnqueued)
      return failure(5, "Metal producer command was not submitted");
    [command waitUntilCompleted];
    AMStream *stream = use.stream;
    // The shared lease may represent an earlier invocation in the same batch.
    // Publish every invocation before observing the latest producer's error.
    NSArray<AMPendingCommand *> *records = use.records;
    if (stream && records) {
      for (AMPendingCommand *record in records) finalizeCommand(stream, record);
    } else {
      AMPendingCommand *record = use.record;
      if (record && stream) finalizeCommand(stream, record);
    }
    if (command.status == MTLCommandBufferStatusError)
      return failure(5, command.error.localizedDescription.UTF8String
          ?: "preceding Metal buffer use failed");
  }
  NSString *message = lease.savedError;
  AMBuffer *owner = lease.owner;
  @synchronized(owner) {
    if (!message) message = owner.producerError;
  }
  return message ? failure(5, message.UTF8String) : ok();
}

AMStatus am_buffer_lease_begin(void *buffer, int32_t access, void **lease,
                               uint64_t *producerCount) {
  if (!buffer || !lease || !producerCount)
    return failure(2, "null Metal buffer lease output");
  *lease = nullptr;
  *producerCount = 0;
  id object = (__bridge id)buffer;
  if (![object isKindOfClass:AMBuffer.class])
    return failure(2, "invalid Metal buffer lease handle");
  AMBufferLease *nativeLease = nil;
  AMStatus status = beginBufferLease((AMBuffer *)object, access, &nativeLease);
  if (status.code) return status;
  *producerCount = nativeLease.producerCount;
  *lease = (__bridge_retained void *)nativeLease;
  return ok();
}

AMStatus am_buffer_lease_wait(void *leaseHandle) {
  if (!leaseHandle) return failure(2, "null Metal buffer lease");
  id object = (__bridge id)leaseHandle;
  if (![object isKindOfClass:AMBufferLease.class])
    return failure(2, "invalid Metal buffer lease handle");
  return waitBufferLease((AMBufferLease *)object);
}

AMStatus am_buffer_evidence(void *buffer, AMEvidence *evidence) {
  if (!buffer || !evidence) return failure(2, "null evidence handle");
  AMBuffer *owner = (__bridge AMBuffer *)buffer;
  AMStatus status = flushOwnerMathBatches(owner);
  if (status.code) return status;
  NSString *message = nil;
  @synchronized(owner) {
    id<MTLCommandBuffer> producer = owner.producer;
    if (producer &&
        (producer.status == MTLCommandBufferStatusCompleted ||
         producer.status == MTLCommandBufferStatusError)) {
      const uint64_t invocation = owner.producerInvocation;
      if (owner.evidence.invocation_id == invocation &&
          owner.evidence.kind == 2) {
        AMEvidence current = owner.evidence;
        current.completed = producer.status == MTLCommandBufferStatusCompleted ? 1 : 0;
        owner.evidence = current;
        if (producer.status == MTLCommandBufferStatusError)
          owner.producerError =
              producer.error.localizedDescription ?: @"Metal command failed";
      }
    }
    *evidence = owner.evidence;
    if (owner.producerError &&
        owner.producerInvocation == owner.evidence.invocation_id)
      message = owner.producerError;
  }
  return message ? failure(5, message.UTF8String) : ok();
}

static AMStatus enqueueMarker(AMStream *stream, id<MTLCommandBuffer> *marker,
                              uint64_t *sequence) {
  if (!stream) return failure(2, "null Metal stream");
  id<MTLCommandBuffer> command = [stream.queue commandBuffer];
  if (!command) return failure(5, "cannot create Metal stream marker");
  command.label = @"arraymancer.stream.marker";
  AMStatus status = submitCommand(stream, command, @[], nil, 0, {}, 0, sequence);
  if (status.code) return status;
  if (marker) *marker = command;
  return ok();
}
static AMStatus synchronizationPoint(AMStream *stream, uint64_t *sequence) {
  @synchronized(stream.encodingLock) {
    // Foreign clients can enqueue commands outside our accepted-sequence ledger.
    if (stream.foreignQueueExposed) return enqueueMarker(stream, nil, sequence);
    AMStatus status = flushMathBatch(stream);
    if (status.code) return status;
    @synchronized(stream) { *sequence = stream.acceptedSequence; }
    return ok();
  }
}

AMStatus am_stream_info(void *streamHandle, void **metalQueue,
                        uint64_t *identity) {
  if (!metalQueue || !identity)
    return failure(2, "null Metal stream information output");
  *metalQueue = nullptr;
  *identity = 0;
  if (!streamHandle) return failure(2, "null Metal stream");
  id object = (__bridge id)streamHandle;
  if (![object isKindOfClass:AMStream.class])
    return failure(2, "invalid Metal stream handle");
  AMStream *stream = (AMStream *)object;
  if (!stream.queue) return failure(1, "Metal stream queue unavailable");
  @synchronized(stream.encodingLock) {
    AMStatus status = flushMathBatch(stream);
    if (status.code) return status;
    stream.foreignQueueExposed = YES;
    *metalQueue = (__bridge void *)stream.queue;
    *identity = stream.identity;
    return ok();
  }
}

AMStatus am_stream_fence(void *streamHandle, void **fenceHandle) {
  @autoreleasepool {
    if (!fenceHandle) return failure(2, "null Metal stream fence output");
    *fenceHandle = nullptr;
    if (!streamHandle) return failure(2, "null Metal stream");
    id object = (__bridge id)streamHandle;
    if (![object isKindOfClass:AMStream.class])
      return failure(2, "invalid Metal stream handle");
    AMStream *stream = (AMStream *)object;
    AMStreamFence *fence = [AMStreamFence new];
    if (!fence) return failure(7, "cannot allocate Metal stream fence");
    uint64_t sequence = 0;
    AMStatus status = enqueueMarker(stream, nil, &sequence);
    if (status.code) return status;
    fence.stream = stream;
    fence.sequence = sequence;
    fence.snapshot = pendingThrough(stream, sequence);
    *fenceHandle = (__bridge_retained void *)fence;
    return ok();
  }
}

AMStatus am_stream_fence_wait(void *fenceHandle) {
  @autoreleasepool {
    if (!fenceHandle) return failure(2, "null Metal stream fence");
    id object = (__bridge id)fenceHandle;
    if (![object isKindOfClass:AMStreamFence.class])
      return failure(2, "invalid Metal stream fence handle");
    AMStreamFence *fence = (AMStreamFence *)object;
    @synchronized(fence) {
      if (!fence.completed) {
        @try {
          waitAndFinalize(fence.stream, fence.snapshot);
          AMStatus status = streamFailureThrough(fence.stream, fence.sequence);
          fence.resultCode = status.code;
          if (status.message) {
            fence.resultMessage =
                [NSString stringWithUTF8String:status.message]
                ?: @"Metal stream command failed";
            free(status.message);
          }
          fence.snapshot = @[];
        } @catch (NSException *exception) {
          fence.resultCode = 5;
          fence.resultMessage =
              exception.reason ?: @"Metal stream fence wait failed";
          fence.snapshot = @[];
        }
        fence.completed = YES;
      }
      return fence.resultCode
          ? failure(fence.resultCode, fence.resultMessage.UTF8String)
          : ok();
    }
  }
}

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
AMStatus am_stream(void *deviceHandle, int32_t mode, void **out,
                   uint64_t *identity) {
  @autoreleasepool {
    if (!out || !identity) return failure(2, "null Metal stream output");
    *out = nullptr;
    *identity = 0;
    if (mode != 0 && mode != 1)
      return failure(2, "Metal stream mode must be new or default");
    id<MTLDevice> device = deviceHandle
        ? (__bridge id<MTLDevice>)deviceHandle : MTLCopyAllDevices().firstObject;
    NSError *error = nil;
    AMDeviceState *state = deviceState(device, &error);
    if (!state)
      return failure(1, error.localizedDescription.UTF8String ?: "Metal device unavailable");
    AMStream *stream = state.defaultStream;
    if (mode == 0) {
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (!queue) return failure(1, "cannot create Metal command queue");
      stream = newStream(state, queue);
      @synchronized(state) { [state.streams addObject:stream]; }
    }
    *identity = stream.identity;
    *out = (__bridge_retained void *)stream;
    return ok();
  }
}

AMStatus am_stream_set(void *streamHandle) {
  @autoreleasepool {
    threadState.stream = (__bridge AMStream *)streamHandle;
  }
  return ok();
}

AMStatus am_stream_sync(void *streamHandle) {
  @autoreleasepool {
    if (!streamHandle) return failure(2, "null Metal stream");
    AMStream *stream = (__bridge AMStream *)streamHandle;
    uint64_t sequence = 0;
    AMStatus status = synchronizationPoint(stream, &sequence);
    if (status.code) return status;
    // Finalize every captured record, including handlers still publishing leases.
    NSArray<AMPendingCommand *> *snapshot = pendingThrough(stream, sequence);
    waitAndFinalize(stream, snapshot);
    return streamFailureThrough(stream, sequence);
  }
}

AMStatus am_device_sync(void *deviceHandle) {
  @autoreleasepool {
    id<MTLDevice> device = deviceHandle
        ? (__bridge id<MTLDevice>)deviceHandle : MTLCopyAllDevices().firstObject;
    NSError *error = nil;
    AMDeviceState *state = deviceState(device, &error);
    if (!state)
      return failure(1, error.localizedDescription.UTF8String ?: "Metal device unavailable");

    NSArray<AMStream *> *streams = nil;
    @synchronized(state) { streams = state.streams.allObjects; }
    NSMutableArray<NSNumber *> *sequences = [NSMutableArray new];
    for (AMStream *stream in streams) {
      uint64_t sequence = 0;
      AMStatus status = synchronizationPoint(stream, &sequence);
      if (status.code) return status;
      [sequences addObject:@(sequence)];
    }
    // Snapshot all queues before waiting, then explicitly wait each prior
    // command so its lease/accounting handler is complete on return.
    NSMutableArray<NSArray<AMPendingCommand *> *> *snapshots = [NSMutableArray new];
    for (NSUInteger index = 0; index < streams.count; ++index)
      [snapshots addObject:pendingThrough(
          streams[index], sequences[index].unsignedLongLongValue)];
    for (NSUInteger index = 0; index < streams.count; ++index)
      waitAndFinalize(streams[index], snapshots[index]);

    AMStatus first = ok();
    for (NSUInteger index = 0; index < streams.count; ++index) {
      AMStatus status = streamFailureThrough(streams[index],
                                              sequences[index].unsignedLongLongValue);
      if (status.code && !first.code) first = status;
      else if (status.message) free(status.message);
    }
    return first;
  }
}


AMStatus am_stream_complete(void *streamHandle, int32_t *complete) {
  if (!streamHandle || !complete)
    return failure(2, "null Metal stream completion argument");
  AMStream *stream = (__bridge AMStream *)streamHandle;
  AMStatus status = flushMathBatch(stream);
  if (status.code) return status;
  AMPendingCommand *latest = nil;
  @synchronized(stream) {
    if (!stream.lastAcceptedComplete && stream.acceptedSequence)
      latest = stream->pending.empty() ? nil : stream->pending.back();
  }
  if (latest &&
      (latest.command.status == MTLCommandBufferStatusCompleted ||
       latest.command.status == MTLCommandBufferStatusError))
    for (AMPendingCommand *record in pendingThrough(stream, latest.sequence))
      finalizeCommand(stream, record);
  NSString *message = nil;
  @synchronized(stream) {
    *complete = stream.acceptedSequence == 0 || stream.lastAcceptedComplete ? 1 : 0;
    if (*complete && stream.lastAcceptedFailed)
      message = stream.lastAcceptedError;
  }
  return message ? failure(5, message.UTF8String) : ok();
}

AMStatus am_event(void *deviceHandle, void **out) {
  @autoreleasepool {
    if (!out) return failure(2, "null Metal event output");
    *out = nullptr;
    id<MTLDevice> device = deviceHandle
        ? (__bridge id<MTLDevice>)deviceHandle : MTLCopyAllDevices().firstObject;
    if (!device) return failure(1, "Metal device unavailable");
    id<MTLSharedEvent> shared = [device newSharedEvent];
    if (!shared) return failure(1, "Metal shared events are unavailable");
    AMEvent *event = [AMEvent new];
    event.device = device;
    event.shared = shared;
    *out = (__bridge_retained void *)event;
    return ok();
  }
}

AMStatus am_event_record(void *eventHandle, void *streamHandle,
                         uint64_t *recordedValue) {
  @autoreleasepool {
    if (!eventHandle || !streamHandle || !recordedValue)
      return failure(2, "null Metal event record argument");
    AMEvent *event = (__bridge AMEvent *)eventHandle;
    AMStream *stream = (__bridge AMStream *)streamHandle;
    if (event.device.registryID != stream.device.registryID)
      return failure(2, "Metal event and stream use different devices");

    uint64_t value = 0;
    @synchronized(event) {
      const uint64_t current = MAX(event.lastScheduledValue, event.shared.signaledValue);
      if (current == UINT64_MAX) return failure(2, "Metal event timeline exhausted");
      value = current + 1;
      event.lastScheduledValue = value;
      id<MTLCommandBuffer> command = [stream.queue commandBuffer];
      if (!command) return failure(5, "cannot create Metal event signal command");
      command.label = @"arraymancer.event.record";
      [command encodeSignalEvent:event.shared value:value];
      AMStatus status = submitCommand(stream, command, @[event], nil, 0, {}, 0, nullptr);
      if (status.code) return status;
      event.lastRecordedValue = value;
    }
    *recordedValue = value;
    return ok();
  }
}

AMStatus am_event_wait(void *streamHandle, void *eventHandle, uint64_t value,
                       int32_t explicitValue) {
  @autoreleasepool {
    if (!streamHandle || !eventHandle ||
        (explicitValue != 0 && explicitValue != 1))
      return failure(2, "invalid Metal event wait argument");
    AMStream *stream = (__bridge AMStream *)streamHandle;
    AMEvent *event = (__bridge AMEvent *)eventHandle;
    if (event.device.registryID != stream.device.registryID)
      return failure(2, "Metal event and stream use different devices");
    uint64_t target = value;
    if (!explicitValue) {
      @synchronized(event) { target = event.lastRecordedValue; }
      if (!target) return failure(2, "cannot wait for an unrecorded Metal event");
    }
    id<MTLCommandBuffer> command = [stream.queue commandBuffer];
    if (!command) return failure(5, "cannot create Metal event wait command");
    command.label = @"arraymancer.event.wait";
    [command encodeWaitForEvent:event.shared value:target];
    return submitCommand(stream, command, @[event], nil, 0, {}, 0, nullptr);
  }
}

AMStatus am_event_signal(void *eventHandle, uint64_t value) {
  if (!eventHandle) return failure(2, "null Metal event");
  AMEvent *event = (__bridge AMEvent *)eventHandle;
  @synchronized(event) {
    const uint64_t current = MAX(event.lastScheduledValue, event.shared.signaledValue);
    if (value <= current)
      return failure(2, "Metal event signal must advance its timeline monotonically");
    event.lastScheduledValue = value;
    event.shared.signaledValue = value;
  }
  return ok();
}
static BOOL wouldExceed(uint64_t current, uint64_t addition, uint64_t limit) {
  return current > limit || addition > limit - current;
}

static AMBuffer *cachedOwnerForRequestLocked(AMDeviceState *state,
                                             uint64_t bytes,
                                             uint64_t nativeLength) {
  NSMutableArray<AMBuffer *> *bin = state.bufferCache[@(nativeLength)];
  // Pop recent storage without shifting the other cached owners.
  for (NSUInteger index = bin.count; index > 0;) {
    AMBuffer *owner = bin[--index];
    if (owner.capacity == bytes && removeCachedOwnerLocked(state, owner, index))
      return owner;
  }
  return nil;
}

// Cached storage keeps its containers, but every allocation gets fresh evidence.
static void initializeBufferOwnerLocked(AMDeviceState *state, AMBuffer *owner) {
  owner.externalReferences = 1;
  owner.producer = nil;
  [owner.nativeLeases removeAllObjects];
  owner.producerInvocation = 0;
  owner.producerError = nil;
  owner.evidence = {nextInvocation.fetch_add(1), state.device.registryID, 0, 0, 0, 1, 1};
  state.liveBytes = saturatedAdd(state.liveBytes, owner.capacity);
}

static id<MTLBuffer> bufferFromHeapLocked(AMDeviceState *state, uint64_t length,
                                         const MTLSizeAndAlign &layout) {
  for (NSUInteger index = state.bufferHeaps.count; index > 0;) {
    id<MTLHeap> heap = state.bufferHeaps[--index];
    if ([heap maxAvailableSizeWithAlignment:layout.align] < layout.size) continue;
    id<MTLBuffer> buffer = [heap newBufferWithLength:length
        options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
    if (buffer) return buffer;
  }
  return nil;
}

static AMBuffer *newBufferOwnerLocked(AMDeviceState *state, id<MTLBuffer> native,
                                      uint64_t bytes) {
  AMBuffer *owner = [AMBuffer new];
  owner.buffer = native;
  owner.gpuAddress = state.supportsIndirectMath ? native.gpuAddress : 0;
  owner.state = state;
  owner.capacity = bytes;
  initializeBufferOwnerLocked(state, owner);
  return owner;
}

AMStatus am_buffer(void *deviceHandle, uint64_t bytes, void **buffer) {
  @autoreleasepool {
    if (!deviceHandle || !buffer) return failure(2, "null allocation handle");
    *buffer = nullptr;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)deviceHandle;
    // Empty tensors still need one packed vector for typed no-read bindings.
    const uint64_t requestedLength = MAX(bytes, 4 * sizeof(float));
    if (requestedLength > dev.maxBufferLength)
      return failure(4, "Metal allocation exceeds maxBufferLength");

    NSError *error = nil;
    AMDeviceState *state = deviceState(dev, &error);
    if (!state)
      return failure(1, error.localizedDescription.UTF8String ?: "Metal device unavailable");
    const BOOL pooled = requestedLength <= 1024 * 1024 && state.supportsSharedHeaps;
    MTLSizeAndAlign layout = {requestedLength, 1};
    BOOL pressure = NO;
    AMBuffer *owner = nil;
    @synchronized(state) {
      const uint64_t limit = memoryLimitBytes(state);
      const BOOL withinLimit = state.reservedBytes <= limit && requestedLength <= limit;
      if (withinLimit) {
        owner = cachedOwnerForRequestLocked(state, bytes, requestedLength);
        if (owner) {
          initializeBufferOwnerLocked(state, owner);
          *buffer = (__bridge_retained void *)owner;
          return ok();
        }
      }
      if (pooled) {
        layout = [dev heapBufferSizeAndAlignWithLength:requestedLength
            options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
        if (!layout.size || !layout.align) return failure(4, "Metal heap layout unavailable");
        if (withinLimit) {
          id<MTLBuffer> native = bufferFromHeapLocked(state, requestedLength, layout);
          if (native) {
            owner = newBufferOwnerLocked(state, native, bytes);
            *buffer = (__bridge_retained void *)owner;
            return ok();
          }
        }
      }
      pressure = wouldExceed(state.reservedBytes, layout.size, limit);
    }
    if (pressure) {
      AMStatus status = flushDeviceMathBatches(dev);
      if (status.code) return status;
    }
    @synchronized(state) {
      const uint64_t limit = memoryLimitBytes(state);
      if (requestedLength > limit)
        return failure(4, "Metal allocation exceeds the configured memory fraction");
      while (state.reservedBytes > limit)
        if (!evictOneCachedOwnerLocked(state))
          return failure(4, "Metal reserved memory exceeds the configured limit");

      owner = cachedOwnerForRequestLocked(state, bytes, requestedLength);
      if (owner) initializeBufferOwnerLocked(state, owner);
      else {
        id<MTLBuffer> native = pooled
            ? bufferFromHeapLocked(state, requestedLength, layout) : nil;
        while (!native && wouldExceed(state.reservedBytes, layout.size, limit)) {
          if (!evictOneCachedOwnerLocked(state))
            return failure(4, "Metal allocation exceeds the configured memory fraction");
          if (pooled) native = bufferFromHeapLocked(state, requestedLength, layout);
        }
        if (!native) {
          const uint64_t available = limit - state.reservedBytes;
          const NSUInteger alignment = MAX(layout.align, NSPageSize());
          NSUInteger heapLength = pooled ? MIN(4 * 1024 * 1024, available) : 0;
          heapLength -= heapLength % alignment;
          if (pooled && heapLength >= layout.size) {
            // Automatic placement never aliases resources still owned by GPU leases.
            MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
            descriptor.size = heapLength;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
            descriptor.type = MTLHeapTypeAutomatic;
            id<MTLHeap> heap = [dev newHeapWithDescriptor:descriptor];
            if (!heap) return failure(4, "Metal heap allocation failed");
            if (heap.size > available)
              return failure(4, "Metal heap exceeds the configured memory fraction");
            native = [heap newBufferWithLength:requestedLength
                options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
            if (!native) return failure(4, "Metal heap buffer allocation failed");
            if (!state.bufferHeaps) state.bufferHeaps = [NSMutableArray new];
            [state.bufferHeaps addObject:heap];
            state.reservedBytes = saturatedAdd(state.reservedBytes, heap.size);
          } else {
            native = [dev newBufferWithLength:requestedLength options:MTLResourceStorageModeShared];
            if (!native) return failure(4, "Metal allocation failed");
            while (wouldExceed(state.reservedBytes, native.length, limit))
              if (!evictOneCachedOwnerLocked(state))
                return failure(4, "Metal allocation exceeds the configured memory fraction");
            state.reservedBytes = saturatedAdd(state.reservedBytes, native.length);
          }
        }
        owner = newBufferOwnerLocked(state, native, bytes);
      }
    }
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
  const __int128 storageElements = d.capacity / width;
  if (!count) {
    if (d.offset < 0 || (__int128)d.offset > storageElements)
      return failure(2, "empty view offset exceeds native storage");
    return ok();
  }
  __int128 lo = d.offset, hi = d.offset;
  for (int i = 0; i < d.rank; ++i) {
    __int128 span = (__int128)(d.shape[i] - 1) * d.strides[i];
    if (span < -storageElements || span > storageElements)
      return failure(2, "stride exceeds storage");
    if (span < 0) lo += span; else hi += span;
  }
  if (lo < 0 || hi >= storageElements)
    return failure(2, "view exceeds native storage");
  return ok();
}
AMStatus am_buffer_info(void *buffer, AMTensorDescriptor d,
                        void **metalBuffer, uint64_t *identity) {
  if (!buffer || !metalBuffer || !identity)
    return failure(2, "null Metal buffer information argument");
  *metalBuffer = nullptr;
  *identity = 0;
  id object = (__bridge id)buffer;
  if (![object isKindOfClass:AMBuffer.class])
    return failure(2, "invalid Metal buffer information handle");
  AMBuffer *owner = (AMBuffer *)object;
  AMDeviceState *state = owner.state;
  if (!state)
    return failure(2, "Metal buffer has no device state");
  @synchronized(state) {
    if (owner.cached || !owner.externalReferences)
      return failure(2, "Core AI borrowing requires an externally retained Metal buffer");
  }
  if (!owner.buffer || owner.buffer.storageMode != MTLStorageModeShared)
    return failure(2, "Core AI borrowing requires shared Metal storage");

  uint64_t count = 0;
  AMStatus status = validate(owner, d, count);
  if (status.code) return status;
  const uint64_t width = dtypeWidth(d.dtype);
  if (!width || owner.capacity > owner.buffer.length)
    return failure(2, "Metal descriptor exceeds physical buffer capacity");

  uint64_t axisStrides[8];
  uint64_t axisDimensions[8];
  uint32_t axes = 0;
  for (int axis = 0; axis < d.rank; ++axis) {
    const int64_t dimension = d.shape[axis];
    const int64_t stride = d.strides[axis];
    if (dimension <= 1) continue;
    if (stride <= 0)
      return failure(2, "Core AI borrowing requires positive strides");
    uint32_t insert = axes;
    while (insert && axisStrides[insert - 1] > (uint64_t)stride) {
      axisStrides[insert] = axisStrides[insert - 1];
      axisDimensions[insert] = axisDimensions[insert - 1];
      --insert;
    }
    axisStrides[insert] = (uint64_t)stride;
    axisDimensions[insert] = (uint64_t)dimension;
    ++axes;
  }
  __uint128_t coveredElements = 1;
  for (uint32_t axis = 0; axis < axes; ++axis) {
    if ((__uint128_t)axisStrides[axis] < coveredElements)
      return failure(2, "Core AI borrowing rejects overlapping Metal strides");
    coveredElements +=
        (__uint128_t)(axisDimensions[axis] - 1) * axisStrides[axis];
  }

  const __uint128_t byteOffset = (__uint128_t)(uint64_t)d.offset * width;
  if (byteOffset > UINT64_MAX || byteOffset > owner.capacity ||
      byteOffset > owner.buffer.length)
    return failure(2, "Metal descriptor byte offset exceeds physical storage");
  if (count) {
    __int128 high = d.offset;
    for (int axis = 0; axis < d.rank; ++axis)
      high += (__int128)(d.shape[axis] - 1) * d.strides[axis];
    if (high < 0)
      return failure(2, "Metal descriptor view exceeds physical buffer bounds");
    const __uint128_t byteEnd = ((__uint128_t)high + 1) * width;
    if (byteEnd > owner.capacity || byteEnd > owner.buffer.length)
      return failure(2, "Metal descriptor view exceeds physical buffer bounds");
  }
  status = flushOwnerMathBatches(owner);
  if (status.code) return status;

  void *native = (__bridge void *)owner.buffer;
  *metalBuffer = native;
  *identity = (uint64_t)(uintptr_t)native;
  return ok();
}


static AMStatus validateFloatTensor(AMBuffer *owner, AMTensorDescriptor d,
                                    uint64_t &count) {
  if (!owner) return failure(2, "null math storage");
  if (d.dtype != 1) return failure(3, "Metal math supports float32 only");
  return validate(owner, d, count);
}

static bool sameDevice(AMBuffer *left, AMBuffer *right) {
  return left && right && left.buffer.device.registryID == right.buffer.device.registryID;
}

static bool sameShape(AMTensorDescriptor left, AMTensorDescriptor right) {
  if (left.rank != right.rank) return false;
  for (int axis = 0; axis < left.rank; ++axis)
    if (left.shape[axis] != right.shape[axis]) return false;
  return true;
}

static bool broadcastAligned(AMTensorDescriptor input, AMTensorDescriptor output) {
  if (input.rank > output.rank) return false;
  const int leading = output.rank - input.rank;
  for (int axis = 0; axis < input.rank; ++axis) {
    const int outputAxis = leading + axis;
    if (input.shape[axis] == output.shape[outputAxis]) continue;
    if (input.shape[axis] != 1 || input.strides[axis] != 0) return false;
  }
  return true;
}

static bool packedMathInput(AMTensorDescriptor descriptor) {
  int64_t stride = 1;
  for (int axis = descriptor.rank - 1; axis >= 0; --axis) {
    const int64_t dimension = descriptor.shape[axis];
    if (!dimension) return false;
    if (dimension == 1) continue;
    if (descriptor.strides[axis] != stride || stride > INT64_MAX / dimension)
      return false;
    stride *= dimension;
  }
  return true;
}

static bool innerSpanMathInput(AMTensorDescriptor input, AMTensorDescriptor output) {
  if (!sameShape(input, output) || !input.rank) return false;
  const int axis = input.rank - 1;
  return input.shape[axis] == 1 || input.strides[axis] == 0 || input.strides[axis] == 1;
}

static bool vectorInnerSpanMathInput(AMTensorDescriptor input) {
  if (input.shape[input.rank - 1] % 4) return false;
  if (input.strides[input.rank - 1] == 0) return true;
  if (input.offset % 4) return false;
  for (int axis = 0; axis < input.rank - 1; ++axis)
    if (input.shape[axis] > 1 && input.strides[axis] % 4) return false;
  return true;
}

static AMCompute *compute(id<MTLDevice> device, const char *directory, NSError **error) {
  if (!device || !directory) return nil;
  const uint64_t registryID = device.registryID;
  if (threadState.compute && threadState.registryID == registryID)
    return threadState.compute;
  static NSMutableDictionary<NSNumber *, AMCompute *> *contexts = [NSMutableDictionary new];
  @synchronized(contexts) {
    AMCompute *context = contexts[@(registryID)];
    if (!context) {
      context = [AMCompute new];
      context.state = deviceState(device, error);
      NSURL *url = [NSURL fileURLWithPath:
        [[NSString stringWithUTF8String:directory] stringByAppendingPathComponent:@"arraymancer.metallib"]];
      if (!context.state) return nil;
      context.library = [device newLibraryWithURL:url error:error];
      if (!context.library) return nil;
      context.pipelines = [NSMutableDictionary new];
      contexts[@(registryID)] = context;
    }
    threadState.compute = context;
    threadState.registryID = registryID;
    return context;
  }
}

static id<MTLComputePipelineState> pipelineFor(AMCompute *context, NSString *name, NSError **error) {
  if (threadState.pipeline && threadState.pipelineContext == context && threadState.pipelineName == name)
    return threadState.pipeline;
  @synchronized(context) {
    id<MTLComputePipelineState> pipeline = context.pipelines[name];
    if (!pipeline) {
      id<MTLFunction> function = [context.library newFunctionWithName:name];
      if (!function) return nil;
      pipeline = [context.state.device newComputePipelineStateWithFunction:function error:error];
      if (pipeline) context.pipelines[name] = pipeline;
    }
    threadState.pipelineContext = context;
    threadState.pipelineName = name;
    threadState.pipeline = pipeline;
    return pipeline;
  }
}

struct AMPackedUnaryParameters {
  uint32_t count;
  int64_t inputOffset, outputOffset;
};
static_assert(sizeof(AMPackedUnaryParameters) == 24, "direct unary parameter layout");

struct AMDotParameters {
  uint64_t inner;
  int64_t leftOffset, rightOffset, outputOffset, leftStride, rightStride;
};
static_assert(sizeof(AMDotParameters) == 48, "dot parameter ABI mismatch");

struct AMVectorMatrixParameters {
  uint64_t inner, columns;
  int64_t inputOffset, matrixOffset, outputOffset;
  int64_t inputStride, matrixRowStride, matrixColumnStride, outputStride;
};
static_assert(sizeof(AMVectorMatrixParameters) == 72, "vector-matrix parameter ABI mismatch");

static AMStatus runMathCommand(AMCompute *context,
                               id<MTLComputePipelineState> pipeline,
                               AMBuffer *a, AMBuffer *b,
                               AMBuffer *output, const AMTensorDescriptor &ad,
                               const AMTensorDescriptor &bd, const AMTensorDescriptor &od,
                               const AMMathParameters &parameters,
                               uint64_t inputCount, uint64_t outputCount, MTLSize groups,
                               MTLSize threads, int32_t operation,
                               bool packedUnary, bool packedUnaryTasks,
                               bool packedDot, bool packedVectorMatrix, bool packedReduction) {
  AMStream *stream = nil;
  AMStatus status = selectedStream(output.buffer.device, context.state, &stream);
  if (status.code) return status;
  @synchronized(stream.encodingLock) {
    id<MTLCommandBuffer> command = mathCommandBuffer(stream);
    if (!command) return failure(5, "cannot create Metal math command");
    // Single-output contractions trade no useful work for concurrent scheduling.
    const bool dot = operation == 20 && od.rank == 2 && ad.shape[ad.rank - 2] == 1 &&
        bd.shape[bd.rank - 1] == 1;
    const bool packedMath = packedUnaryTasks || packedVectorMatrix || packedReduction;
    // Packed tasks already expose their parallel work inside one dispatch.
    id<MTLComputeCommandEncoder> encoder = mathComputeEncoder(stream, command, output,
        dot || packedMath);
    if (!encoder) return failure(5, "cannot create Metal math encoder");
    const uint64_t invocation = nextInvocation.fetch_add(1);
    AMEvidence evidence = {invocation, output.buffer.device.registryID,
                           0, 0, 0, 0, 2, 100 + operation};
    AMBuffer *__unsafe_unretained leases[3];
    NSUInteger leaseCount = 0;
    if (a) leases[leaseCount++] = a;
    if (b && b != a) leases[leaseCount++] = b;
    if (output != a && output != b) leases[leaseCount++] = output;
    status = admitMathCommand(stream, command, nil, output, evidence, pipeline,
        leases, leaseCount);
    if (status.code) return status;
    @try {
      const uint64_t packedOutputAddress = packedMath ?
          output.gpuAddress + (uint64_t)od.offset * sizeof(float) : 0;
      if (!packedMath || (stream->packedTaskCount &&
          (stream->packedTaskCount == 32 || stream.packedPipeline != pipeline ||
           stream->packedGroups.width != groups.width ||
           stream->packedThreads.width != threads.width)))
        flushPackedMath(stream, encoder);
      const bool orderedWrites = packedUnaryTasks && stream->packedTaskCount &&
          stream->packedTasks[0].output == packedOutputAddress;
      orderMathAccess(stream, output, a, b, nil, packedMath, orderedWrites);
      if (packedMath) {
        if (!stream->packedTaskCount) {
          stream.packedPipeline = pipeline;
          stream->packedGroups = groups;
          stream->packedThreads = threads;
          stream->packedRows = packedVectorMatrix || packedReduction;
          stream->packedRepeatedWrites = false;
        }
        if (orderedWrites && stream->packedTaskCount)
          stream->packedRepeatedWrites = true;
        stream->packedTasks[stream->packedTaskCount++] = {
          a.gpuAddress + (uint64_t)ad.offset * sizeof(float),
          packedVectorMatrix ? b.gpuAddress + (uint64_t)bd.offset * sizeof(float) : 0,
          packedOutputAddress,
          packedVectorMatrix ? (uint32_t)ad.shape[1] :
              (packedReduction ? (uint32_t)(operation == 25 ? inputCount :
                  ad.shape[parameters.axis]) : (uint32_t)outputCount),
          packedVectorMatrix ? (uint32_t)bd.shape[1] :
              (packedReduction ? (uint32_t)outputCount : 0)};
        if (!stream.aggregateMath) flushPackedMath(stream, encoder);
        return finishMathEncoding(stream);
      }
      [encoder setComputePipelineState:pipeline];
      if (packedUnary) {
        AMPackedUnaryParameters packed = {(uint32_t)outputCount, ad.offset, od.offset};
        [encoder setBuffer:a.buffer offset:0 atIndex:0];
        [encoder setBuffer:output.buffer offset:0 atIndex:1];
        [encoder setBytes:&packed length:sizeof(packed) atIndex:2];
      } else switch (operation) {
        case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: case 9:
        case 10: case 11: case 12: case 13: case 14: case 15: case 16: case 17: case 18: case 19: case 23: case 24:
          [encoder setBuffer:(a ? a.buffer : output.buffer) offset:0 atIndex:0];
          [encoder setBuffer:(b ? b.buffer : output.buffer) offset:0 atIndex:1];
          [encoder setBuffer:output.buffer offset:0 atIndex:2];
          [encoder setBytes:&ad length:sizeof(ad) atIndex:3];
          [encoder setBytes:&bd length:sizeof(bd) atIndex:4];
          [encoder setBytes:&od length:sizeof(od) atIndex:5];
          [encoder setBytes:&parameters length:sizeof(parameters) atIndex:6];
          break;
        case 20:
          [encoder setBuffer:a.buffer offset:(packedDot ? (NSUInteger)ad.offset * sizeof(float) : 0) atIndex:0];
          [encoder setBuffer:b.buffer offset:(packedDot ? (NSUInteger)bd.offset * sizeof(float) : 0) atIndex:1];
          [encoder setBuffer:output.buffer offset:(packedDot ? (NSUInteger)od.offset * sizeof(float) : 0) atIndex:2];
          if (packedDot) {
            const uint32_t inner = (uint32_t)ad.shape[1];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:3];
          } else if (od.rank == 2 && ad.shape[0] == 1 && bd.shape[1] == 1) {
            AMDotParameters compact = {(uint64_t)ad.shape[1], ad.offset, bd.offset,
                od.offset, ad.strides[1], bd.strides[0]};
            [encoder setBytes:&compact length:sizeof(compact) atIndex:3];
          } else if (od.rank == 2 && ad.shape[0] == 1 && bd.shape[1] > 1) {
            AMVectorMatrixParameters compact = {
              (uint64_t)ad.shape[1], (uint64_t)bd.shape[1], ad.offset, bd.offset, od.offset,
              ad.strides[1], bd.strides[0], bd.strides[1], od.strides[1]};
            [encoder setBytes:&compact length:sizeof(compact) atIndex:3];
          } else {
            [encoder setBytes:&ad length:sizeof(ad) atIndex:3];
            [encoder setBytes:&bd length:sizeof(bd) atIndex:4];
            [encoder setBytes:&od length:sizeof(od) atIndex:5];
          }
          break;
        case 21: case 22: case 25: case 26: case 27:
          [encoder setBuffer:a.buffer offset:0 atIndex:0];
          [encoder setBuffer:output.buffer offset:0 atIndex:1];
          [encoder setBytes:&ad length:sizeof(ad) atIndex:2];
          [encoder setBytes:&od length:sizeof(od) atIndex:3];
          [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
          break;
      }
      // Direct tracked bindings cover inter-pass hazards; orderMathAccess covers this pass.
      [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    } @catch (NSException *exception) {
      return failMathEncoding(stream, invocation, exception.reason);
    }
    return finishMathEncoding(stream);
  }
}
struct MatrixLayout {
  NSUInteger rows, columns, rowBytes;
  bool transpose;
};

static bool matrixLayout(AMTensorDescriptor descriptor, MatrixLayout &layout) {
  if (descriptor.rank != 2 || descriptor.shape[0] <= 0 || descriptor.shape[1] <= 0)
    return false;
  const int64_t rows = descriptor.shape[0], columns = descriptor.shape[1];
  int64_t rowStride;
  if ((columns == 1 || descriptor.strides[1] == 1) &&
      (rows == 1 || descriptor.strides[0] >= columns)) {
    rowStride = rows == 1 ? columns : descriptor.strides[0];
    layout = {(NSUInteger)rows, (NSUInteger)columns, 0, false};
  } else if ((rows == 1 || descriptor.strides[0] == 1) &&
             (columns == 1 || descriptor.strides[1] >= rows)) {
    rowStride = columns == 1 ? rows : descriptor.strides[1];
    layout = {(NSUInteger)columns, (NSUInteger)rows, 0, true};
  } else {
    return false;
  }
  const __uint128_t bytes = (__uint128_t)rowStride * sizeof(float);
  if (bytes > NSUIntegerMax) return false;
  layout.rowBytes = (NSUInteger)bytes;
  return true;
}

static MPSMatrix *matrixView(AMStream *stream, AMBuffer *owner,
                             const AMTensorDescriptor &descriptor,
                             const MatrixLayout &layout) {
  MPSMatrixDescriptor *shape = nil;
  for (const auto &entry : stream->matrixDescriptors) {
    if (entry.rows == layout.rows && entry.columns == layout.columns &&
        entry.rowBytes == layout.rowBytes && entry.descriptor) {
      shape = entry.descriptor;
      break;
    }
  }
  if (!shape) {
    shape = [MPSMatrixDescriptor matrixDescriptorWithRows:layout.rows
        columns:layout.columns rowBytes:layout.rowBytes dataType:MPSDataTypeFloat32];
    auto &entry = stream->matrixDescriptors[stream->matrixDescriptorCursor];
    entry.rows = layout.rows;
    entry.columns = layout.columns;
    entry.rowBytes = layout.rowBytes;
    entry.descriptor = shape;
    stream->matrixDescriptorCursor = (stream->matrixDescriptorCursor + 1) % 3;
  }
  return [[MPSMatrix alloc] initWithBuffer:owner.buffer
      offset:(NSUInteger)descriptor.offset * sizeof(float) descriptor:shape];
}

static MPSVector *vectorView(AMBuffer *owner, const AMTensorDescriptor &descriptor,
                             NSUInteger length) {
  MPSVectorDescriptor *shape = [MPSVectorDescriptor
      vectorDescriptorWithLength:length dataType:MPSDataTypeFloat32];
  return [[MPSVector alloc] initWithBuffer:owner.buffer
      offset:(NSUInteger)descriptor.offset * sizeof(float) descriptor:shape];
}

static bool mpsMatmulSupported(AMTensorDescriptor ad, AMTensorDescriptor bd,
                               AMTensorDescriptor od) {
  if (ad.rank != 2 || bd.rank != 2 || od.rank != 2 || ad.shape[1] == 0)
    return false;
  const int64_t rows = ad.shape[0], columns = bd.shape[1];
  MatrixLayout left, right, output;
  if (rows == 1) return false;
  if (columns == 1)
    return matrixLayout(ad, left) && bd.strides[0] == 1 && od.strides[0] == 1;
  // Small matrix products do not amortize MPS descriptor and wrapper setup.
  if ((uint64_t)rows <= (1ull << 22) / (uint64_t)columns / (uint64_t)ad.shape[1])
    return false;
  return matrixLayout(ad, left) && matrixLayout(bd, right) &&
      matrixLayout(od, output) && !output.transpose;
}

static AMStatus runMpsMatmul(AMCompute *context, AMBuffer *a, AMBuffer *b,
                             AMBuffer *output, AMTensorDescriptor ad,
                             AMTensorDescriptor bd, AMTensorDescriptor od) {
  AMStream *stream = nil;
  AMStatus status = selectedStream(output.buffer.device, context.state, &stream);
  if (status.code) return status;
  const NSUInteger rows = ad.shape[0], inner = ad.shape[1], columns = bd.shape[1];
  const uint64_t invocation = nextInvocation.fetch_add(1);
  AMEvidence evidence = {invocation, output.buffer.device.registryID,
                         0, 0, 0, 0, 2, 120};
  // MPS wrappers keep GPU objects; the shared lease union keeps canonical owners.
  AMBuffer *__unsafe_unretained buffers[3];
  NSUInteger bufferCount = 0;
  buffers[bufferCount++] = a;
  if (b != a) buffers[bufferCount++] = b;
  if (output != a && output != b) buffers[bufferCount++] = output;
  // Kernels are stream-local; serialize their cache and encode-time state.
  @synchronized(stream.encodingLock) {
    id<MTLCommandBuffer> command = mathCommandBuffer(stream);
    if (!command) return failure(5, "cannot create MPS matrix command");
    @try {
      if (columns == 1) {
        MatrixLayout layout;
        matrixLayout(ad, layout);
        NSString *key = [NSString stringWithFormat:@"mv:%lu:%lu:%d",
            (unsigned long)rows, (unsigned long)inner, (int)layout.transpose];
        MPSMatrixVectorMultiplication *kernel =
            (MPSMatrixVectorMultiplication *)stream.matrixKernels[key];
        if (!kernel) {
          kernel = [[MPSMatrixVectorMultiplication alloc]
              initWithDevice:stream.device transpose:layout.transpose rows:rows columns:inner
              alpha:1.0 beta:0.0];
          if (!kernel) return failMathEncoding(stream, invocation, @"cannot create MPS matrix/vector kernel", 7);
          stream.matrixKernels[key] = kernel;
        }
        MPSMatrix *matrixValue = matrixView(stream, a, ad, layout);
        MPSVector *vectorValue = vectorView(b, bd, inner);
        MPSVector *resultValue = vectorView(output, od, rows);
        NSArray *leases = @[kernel, matrixValue, vectorValue, resultValue];
        status = admitMathCommand(stream, command, leases, output, evidence,
            nil, buffers, bufferCount);
        if (status.code) return status;
        flushPackedMath(stream, stream.batchEncoder);
        [stream.batchEncoder endEncoding];
        stream.batchEncoder = nil;
        [kernel encodeToCommandBuffer:command inputMatrix:matrixValue
            inputVector:vectorValue resultVector:resultValue];
      } else {
        MatrixLayout left, right, result;
        matrixLayout(ad, left);
        matrixLayout(bd, right);
        matrixLayout(od, result);
        NSString *key = [NSString stringWithFormat:@"mm:%lu:%lu:%lu:%d:%d",
            (unsigned long)rows, (unsigned long)columns, (unsigned long)inner,
            (int)left.transpose, (int)right.transpose];
        MPSMatrixMultiplication *kernel =
            (MPSMatrixMultiplication *)stream.matrixKernels[key];
        if (!kernel) {
          kernel = [[MPSMatrixMultiplication alloc]
              initWithDevice:stream.device transposeLeft:left.transpose
              transposeRight:right.transpose resultRows:rows resultColumns:columns
              interiorColumns:inner alpha:1.0 beta:0.0];
          if (!kernel) return failMathEncoding(stream, invocation, @"cannot create MPS matrix kernel", 7);
          stream.matrixKernels[key] = kernel;
        }
        MPSMatrix *leftValue = matrixView(stream, a, ad, left);
        MPSMatrix *rightValue = matrixView(stream, b, bd, right);
        MPSMatrix *resultValue = matrixView(stream, output, od, result);
        NSArray *leases = @[kernel, leftValue, rightValue, resultValue];
        status = admitMathCommand(stream, command, leases, output, evidence,
            nil, buffers, bufferCount);
        if (status.code) return status;
        flushPackedMath(stream, stream.batchEncoder);
        [stream.batchEncoder endEncoding];
        stream.batchEncoder = nil;
        [kernel encodeToCommandBuffer:command leftMatrix:leftValue
            rightMatrix:rightValue resultMatrix:resultValue];
      }
    } @catch (NSException *exception) {
      return failMathEncoding(stream, invocation, exception.reason);
    }
    status = finishMathEncoding(stream);
  }
  return status;
}

typedef struct {
  uint32_t counter[4];
  uint64_t seed;
} AMRngParameters;
static_assert(sizeof(AMRngParameters) == 24, "RNG shader parameter ABI mismatch");


static bool packedRandomOutput(AMTensorDescriptor descriptor,
                               uint64_t count) {
  if (descriptor.offset != 0) return false;
  if (!count) return true;
  int64_t expectedStride = 1;
  for (int axis = descriptor.rank - 1; axis >= 0; --axis) {
    if (descriptor.strides[axis] < 0) return false;
    if (descriptor.shape[axis] > 1 &&
        descriptor.strides[axis] != expectedStride) return false;
    expectedStride *= descriptor.shape[axis];
  }
  return true;
}

AMStatus am_rand(void *generatorHandle, void *outputHandle,
                 AMTensorDescriptor descriptor, const char *directory) {
  @autoreleasepool {
    if (!generatorHandle || !outputHandle)
      return failure(2, "null Metal random-generation handle");
    AMGenerator *generator = (__bridge AMGenerator *)generatorHandle;
    AMBuffer *output = (__bridge AMBuffer *)outputHandle;
    AMDeviceState *state = generator.deviceState;
    if (!state || !state.device)
      return failure(1, "Metal random generator has no device");
    if (output.state != state ||
        output.buffer.device.registryID != state.device.registryID)
      return failure(2, "Metal RNG generator and output use different devices");

    uint64_t count = 0;
    AMStatus status = validateFloatTensor(output, descriptor, count);
    if (status.code) return status;
    if (!packedRandomOutput(descriptor, count))
      return failure(2, "Metal random output must be a packed mutable tensor");
    if (!count) return ok();
    if (count > UINT32_MAX)
      return failure(2, "Metal random output exceeds thread-grid limits");
    if (!directory) return failure(9, "null Metal library directory");

    NSError *error = nil;
    AMCompute *context = compute(state.device, directory, &error);
    if (!context)
      return failure(9, error.localizedDescription.UTF8String ?: "Metal queue or library unavailable");
    NSString *name = @"rng_philox4x32_10";
    id<MTLComputePipelineState> pipeline = pipelineFor(context, name, &error);
    if (!pipeline)
      return failure(error ? 7 : 8,
                     error.localizedDescription.UTF8String ?: "missing Philox Metal kernel");
    const NSUInteger width = MIN((NSUInteger)256,
                                 pipeline.maxTotalThreadsPerThreadgroup);
    if (!width) return failure(7, "Philox Metal pipeline has no supported threads");

    AMStream *stream = nil;
    status = selectedStream(state.device, context.state, &stream);
    if (status.code) return status;
    id<MTLCommandBuffer> command = [stream.queue commandBuffer];
    if (!command) return failure(5, "cannot create Metal random command");

    const uint64_t blockCount = count / 4 + (count % 4 != 0);
    @synchronized(generator) {
      const AMRngState start = generator->rng;
      if (start.version != 1 || start.reserved != 0)
        return failure(6, "incompatible Metal RNG state version or reserved field");
      uint32_t nextCounter[4];
      if (!addRngBlocks(start.counter, blockCount, nextCounter))
        return failure(6, "Metal RNG 128-bit counter overflow");
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (!encoder) return failure(5, "cannot create Metal random encoder");
      command.label = name;
      [encoder setComputePipelineState:pipeline];

      AMRngParameters parameters = {};
      memcpy(parameters.counter, start.counter, sizeof(parameters.counter));
      parameters.seed = start.seed;
      [encoder setBuffer:output.buffer offset:0 atIndex:0];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
      [encoder setBytes:&count length:sizeof(count) atIndex:2];
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];

      memcpy(generator->rng.counter, nextCounter, sizeof(nextCounter));
      const uint64_t invocation = nextInvocation.fetch_add(1);
      AMEvidence evidence = {invocation, state.device.registryID,
                             0, 0, 0, 0, 2, 200};
      status = submitCommand(stream, command, @[generator, output, pipeline],
                             output, invocation, evidence, 0, nullptr);
      if (status.code) abandonBufferProducer(output, invocation);
    }
    return status;
  }
}


AMStatus am_math(void *aHandle, AMTensorDescriptor ad,
                 void *bHandle, AMTensorDescriptor bd,
                 void *outputHandle, AMTensorDescriptor od,
                 AMMathParameters parameters, const char *directory) {
  @autoreleasepool {
    if (!outputHandle) return failure(2, "null math output storage");
    if (!directory) return failure(9, "null Metal library directory");
    if (parameters.reserved != 0 ||
        (parameters.scalar_input != 0 && parameters.scalar_input != 1))
      return failure(2, "invalid Metal math parameters");

    AMBuffer *a = aHandle ? (__bridge AMBuffer *)aHandle : nil;
    AMBuffer *b = bHandle ? (__bridge AMBuffer *)bHandle : nil;
    AMBuffer *output = (__bridge AMBuffer *)outputHandle;
    uint64_t outputCount = 0;
    AMStatus status = validateFloatTensor(output, od, outputCount);
    if (status.code) return status;

    const int operation = parameters.operation;
    const bool elementwise = (operation >= 1 && operation <= 19) || operation == 23;
    if (!elementwise && (operation < 20 || operation > 27))
      return failure(2, "unsupported Metal math operation");

    uint64_t aCount = 0, bCount = 0;
    if (operation == 6) {
      if (a || b) return failure(2, "fill does not accept input buffers");
    } else {
      if (!a) return failure(2, "Metal math input A is required");
      status = validateFloatTensor(a, ad, aCount);
      if (status.code) return status;
      if (!sameDevice(a, output)) return failure(2, "Metal math buffers use different devices");
    }
    if (b) {
      status = validateFloatTensor(b, bd, bCount);
      if (status.code) return status;
      if (!sameDevice(b, output)) return failure(2, "Metal math buffers use different devices");
    }

    NSString *kernelName = nil;
    MTLSize groups = MTLSizeMake(0, 1, 1);
    MTLSize threads = MTLSizeMake(1, 1, 1);
    uint64_t dispatchCount = outputCount;
    bool packedUnary = false;
    bool packedUnaryTasks = false;
    bool packedDot = false;
    bool packedVectorMatrix = false;
    bool packedAxis = false;
    bool packedAxisColumns4 = false;
    bool packedAxisRows4 = false;
    bool packedFullProduct = false;

    if (elementwise) {
      if (operation == 13 || operation == 15 || operation == 17 || operation == 19 || operation == 23) {
        if (!b || parameters.scalar_input || !sameShape(ad, bd) || !sameShape(ad, od))
          return failure(2, "fused binary operation requires matching tensor shapes");
      } else if (operation == 5) {
        if (b || parameters.scalar_input || !sameShape(ad, od))
          return failure(2, "negation requires matching input and output shapes");
      } else if (operation == 6) {
        if (a || b) return failure(2, "fill does not accept input buffers");
      } else if (operation == 8 || operation == 9) {
        if (b || !parameters.scalar_input || !sameShape(ad, od))
          return failure(2, "scalar-left operation requires matching tensor shape");
      } else if ((operation >= 10 && operation <= 12) || operation == 14 || operation == 16 || operation == 18) {
        if (b || !parameters.scalar_input || !sameShape(ad, od))
          return failure(2, "unary operation requires matching input and output shapes");
      } else {
        if (!broadcastAligned(ad, od))
          return failure(2, "input A is not broadcast-aligned to output");
        if (parameters.scalar_input) {
          if (b) return failure(2, "scalar input must not provide buffer B");
        } else {
          if (!b || !broadcastAligned(bd, od))
            return failure(2, "input B is missing or not broadcast-aligned");
        }
      }
      if (outputCount > UINT32_MAX)
        return failure(2, "elementwise output exceeds Metal grid limits");
      if (!outputCount) return ok();
      kernelName = @"math_elementwise";
      if (packedMathInput(od)) {
        if ((!a || (sameShape(ad, od) && packedMathInput(ad))) &&
            (!b || (sameShape(bd, od) && packedMathInput(bd)))) {
          kernelName = @"math_elementwise_packed";
          if (operation >= 10 && operation <= 12 && ad.offset % 4 == 0 && od.offset % 4 == 0) {
            packedUnary = true;
            dispatchCount = (outputCount + 3) / 4;
            packedUnaryTasks = output.state.supportsIndirectMath;
            if (packedUnaryTasks)
              kernelName = operation == 10 ? @"math_exp_tasks" :
                  operation == 11 ? @"math_ln_tasks" : @"math_sqrt_tasks";
            else
              kernelName = operation == 10 ? @"math_exp_packed" :
                  operation == 11 ? @"math_ln_packed" : @"math_sqrt_packed";
          }
        } else if ((!a || innerSpanMathInput(ad, od)) &&
                   (!b || innerSpanMathInput(bd, od))) {
          kernelName = @"math_elementwise_inner_span";
          if ((operation <= 4 || operation == 7) && a && od.offset % 4 == 0 &&
              vectorInnerSpanMathInput(ad) && (!b || vectorInnerSpanMathInput(bd))) {
            kernelName = @"math_binary_inner_span4";
            dispatchCount = outputCount / 4;
          }
        }
      }
      threads = MTLSizeMake(MIN((NSUInteger)256, (NSUInteger)outputCount), 1, 1);
      groups = MTLSizeMake((outputCount + threads.width - 1) / threads.width, 1, 1);
    } else if (operation == 20) {
      if (parameters.scalar_input || !b || ad.rank < 2 ||
          ad.rank != od.rank || bd.rank != od.rank)
        return failure(2, "matmul requires batch-aligned matrix descriptors");
      const int rowAxis = od.rank - 2, columnAxis = od.rank - 1;
      if (ad.shape[columnAxis] != bd.shape[rowAxis] ||
          od.shape[rowAxis] != ad.shape[rowAxis] ||
          od.shape[columnAxis] != bd.shape[columnAxis])
        return failure(2, "matmul dimensions are incompatible");
      for (int axis = 0; axis < rowAxis; ++axis) {
        if (ad.shape[axis] != od.shape[axis]) {
          if (ad.shape[axis] != 1 || ad.strides[axis] != 0)
            return failure(2, "matmul input A has incompatible batch dimensions");
          ad.shape[axis] = od.shape[axis];
        }
        if (bd.shape[axis] != od.shape[axis]) {
          if (bd.shape[axis] != 1 || bd.strides[axis] != 0)
            return failure(2, "matmul input B has incompatible batch dimensions");
          bd.shape[axis] = od.shape[axis];
        }
      }
      if (outputCount > UINT32_MAX)
        return failure(2, "matmul output exceeds Metal grid limits");
      if (!outputCount) return ok();
      const uint64_t rows = od.shape[rowAxis], columns = od.shape[columnAxis];
      const uint64_t batches = outputCount / rows / columns;
      if (rows == 1 && columns == 1) {
        kernelName = od.rank == 2 ? @"math_dot_unbatched" : @"math_dot";
        packedDot = od.rank == 2 && ad.shape[1] > 0 && ad.shape[1] <= UINT32_MAX &&
            ad.strides[1] == 1 && bd.strides[0] == 1 &&
            ad.offset % 4 == 0 && bd.offset % 4 == 0;
        if (packedDot) kernelName = @"math_dot_packed";
        threads = MTLSizeMake(256, 1, 1);
        groups = MTLSizeMake(batches, 1, 1);
      } else if (rows == 1) {
        packedVectorMatrix = od.rank == 2 && bCount <= UINT32_MAX &&
            packedMathInput(ad) && packedMathInput(bd) && packedMathInput(od) &&
            bd.shape[1] % 4 == 0 && bd.offset % 4 == 0 && od.offset % 4 == 0;
        kernelName = od.rank == 2 ? @"math_vector_matrix_unbatched" : @"math_vector_matrix";
        packedVectorMatrix = packedVectorMatrix &&
            output.state.supportsIndirectMath;
        if (packedVectorMatrix) kernelName = @"math_vector_matrix_packed";
        threads = MTLSizeMake(packedVectorMatrix ? 32 : 256, 1, 1);
        groups = MTLSizeMake((columns + 31) / 32, 1, batches);
      } else {
        kernelName = @"math_matmul_batched";
        threads = MTLSizeMake(8, 8, 1);
        groups = MTLSizeMake((columns + 15) / 16, (rows + 15) / 16, batches);
      }
      if (groups.width > UINT32_MAX || groups.height > UINT32_MAX ||
          groups.depth > UINT32_MAX)
        return failure(2, "matmul grid exceeds Metal limits");
    } else if (operation == 21 || operation == 25) {
      if (b || (operation == 25 && parameters.scalar_input))
        return failure(2, "full reduction accepts one tensor input");
      const bool small = aCount <= 65536;
      const uint64_t partialCount = small ? 1 : (aCount + 255) / 256;
      if (outputCount != partialCount || partialCount > UINT32_MAX ||
          (parameters.scalar_input && partialCount != 1))
        return failure(2, "full-reduction output must contain one value per group");
      kernelName = operation == 25 ? @"math_product" :
          (small ? @"math_reduce_small" : @"math_reduce");
      parameters.axis = small && packedMathInput(ad);
      threads = MTLSizeMake(256, 1, 1);
      groups = MTLSizeMake(partialCount, 1, 1);
      packedFullProduct = operation == 25 && parameters.axis &&
          output.state.supportsIndirectMath;
      if (packedFullProduct)
        kernelName = aCount % 4 == 0 ?
            @"math_product4_tasks" : @"math_product_tasks";
    } else if (operation == 24) {
      if (!b || parameters.scalar_input || !sameShape(ad, bd) ||
          aCount != bCount || od.rank != 0 || outputCount != 1)
        return failure(2, "fused mean MSE requires matching broadcast views and scalar output");
      kernelName = @"math_mse_reduce";
      parameters.axis = int(packedMathInput(ad)) | (int(packedMathInput(bd)) << 1);
      threads = MTLSizeMake(256, 1, 1);
      groups = MTLSizeMake(1, 1, 1);
    } else {
      if (b || parameters.scalar_input || parameters.axis < 0 ||
          parameters.axis >= ad.rank || od.rank != ad.rank)
        return failure(2, "invalid axis-reduction arguments");
      for (int axis = 0; axis < ad.rank; ++axis) {
        const int64_t expected = axis == parameters.axis ? 1 : ad.shape[axis];
        if (od.shape[axis] != expected)
          return failure(2, "axis-reduction output must keep the reduced dimension");
      }
      if (outputCount > UINT32_MAX)
        return failure(2, "axis-reduction output exceeds Metal grid limits");
      if (!outputCount) return ok();
      kernelName = @"math_axis_sum";
      threads = MTLSizeMake(256, 1, 1);
      groups = MTLSizeMake(outputCount, 1, 1);
      // Short axes share a SIMD group, or coalesce adjacent strided outputs.
      if (ad.shape[parameters.axis] <= 4096 &&
          (ad.strides[parameters.axis] == 1 || ad.strides[parameters.axis] == -1)) {
        kernelName = @"math_axis_simd";
        // Resolve the actual pipeline SIMD width after pipeline creation.
      } else if (ad.shape[parameters.axis] <= 256) {
        kernelName = @"math_axis_serial";
        groups.width = (outputCount + 255) / 256;
      }
      if (output.state.supportsIndirectMath && ad.rank == 2 &&
          packedMathInput(ad) && packedMathInput(od) &&
          ((parameters.axis == 1 && ad.shape[1] <= 4096) ||
           (parameters.axis == 0 && ad.shape[0] <= 256))) {
        packedAxis = true;
        if (parameters.axis == 1) {
          packedAxisRows4 = ad.shape[1] % 4 == 0;
          if (packedAxisRows4)
            kernelName = operation == 22 ? @"math_sum_rows4_tasks" :
                (operation == 26 ? @"math_prod_rows4_tasks" : @"math_mean_rows4_tasks");
          else
            kernelName = operation == 22 ? @"math_sum_rows_tasks" :
                (operation == 26 ? @"math_prod_rows_tasks" : @"math_mean_rows_tasks");
        } else
          kernelName = operation == 22 ? @"math_sum_columns_tasks" :
              (operation == 26 ? @"math_prod_columns_tasks" : @"math_mean_columns_tasks");
        packedAxisColumns4 = parameters.axis == 0 && ad.shape[1] % 4 == 0 &&
            ad.offset % 4 == 0 && od.offset % 4 == 0;
        if (packedAxisColumns4)
          kernelName = operation == 22 ? @"math_sum_columns4_tasks" :
              (operation == 26 ? @"math_prod_columns4_tasks" : @"math_mean_columns4_tasks");
      }
    }

    NSError *error = nil;
    AMCompute *context = compute(output.buffer.device, directory, &error);
    if (!context)
      return failure(9, error ? error.localizedDescription.UTF8String : "Metal queue or library unavailable");
    if (operation == 20 && output != a && output != b && mpsMatmulSupported(ad, bd, od))
      return runMpsMatmul(context, a, b, output, ad, bd, od);
    id<MTLComputePipelineState> pipeline = pipelineFor(context, kernelName, &error);
    if (!pipeline)
      return failure(error ? 7 : 8,
                     error ? error.localizedDescription.UTF8String : "missing Metal math kernel");
    if ([kernelName isEqualToString:@"math_axis_simd"] ||
        (packedAxis && parameters.axis == 1)) {
      const NSUInteger simdWidth = pipeline.threadExecutionWidth;
      if (!simdWidth || simdWidth > 256 || (simdWidth & (simdWidth - 1)))
        return failure(7, "axis reduction pipeline has an unsupported SIMD width");
      const uint64_t outputsPerGroup = 256 /
          (packedAxisRows4 ? MIN((NSUInteger)8, simdWidth) : simdWidth);
      groups.width = (outputCount + outputsPerGroup - 1) / outputsPerGroup;
      if (groups.width * threads.width > UINT32_MAX)
        return failure(2, "axis reduction grid exceeds Metal index limits");
    }
    if (packedAxisColumns4) {
      const NSUInteger simdWidth = pipeline.threadExecutionWidth;
      if (!simdWidth || simdWidth > 256 || (simdWidth & (simdWidth - 1)))
        return failure(7, "column reduction pipeline has an unsupported SIMD width");
      threads.width = simdWidth;
      const uint64_t outputsPerGroup = 4 * MIN((NSUInteger)4, simdWidth);
      groups.width = (outputCount + outputsPerGroup - 1) / outputsPerGroup;
    }
    if (([kernelName isEqualToString:@"math_axis_serial"] ||
         (packedAxis && parameters.axis == 0)) && groups.width * threads.width > UINT32_MAX)
      return failure(2, "axis reduction grid exceeds Metal index limits");
    if (operation >= 20 &&
        pipeline.maxTotalThreadsPerThreadgroup < threads.width * threads.height * threads.depth)
      return failure(7, "Metal math pipeline cannot run its required threadgroup");
    if (operation == 20 && od.shape[od.rank - 2] == 1 && od.shape[od.rank - 1] > 1) {
      const NSUInteger simdWidth = pipeline.threadExecutionWidth;
      if (!simdWidth || 256 % simdWidth)
        return failure(7, "row-vector pipeline has an unsupported SIMD width");
      if (packedVectorMatrix) threads.width = simdWidth;
      const uint64_t columnsPerGroup = 4 *
          (packedVectorMatrix ? MIN((NSUInteger)4, simdWidth) : 256 / simdWidth);
      groups.width = (od.shape[od.rank - 1] + columnsPerGroup - 1) / columnsPerGroup;
    }
    if (elementwise) {
      const NSUInteger supported = pipeline.maxTotalThreadsPerThreadgroup;
      const NSUInteger width = MIN((NSUInteger)dispatchCount, MIN((NSUInteger)256, supported));
      if (!width) return failure(7, "Metal elementwise pipeline has no supported threads");
      threads = MTLSizeMake(width, 1, 1);
      groups = MTLSizeMake((dispatchCount + width - 1) / width, 1, 1);
      if (groups.width > UINT32_MAX || groups.width * width > UINT32_MAX)
        return failure(2, "elementwise grid exceeds Metal index limits");
    }

    return runMathCommand(context, pipeline, a, b, output,
                          ad, bd, od, parameters, aCount, outputCount,
                          groups, threads, operation, packedUnary, packedUnaryTasks,
                          packedDot, packedVectorMatrix, packedAxis || packedFullProduct);
  }
}

AMStatus am_cumsum(void *inputHandle, AMTensorDescriptor inputDescriptor,
                   void *outputHandle, AMTensorDescriptor outputDescriptor,
                   int32_t axis, int32_t reverse, const char *directory) {
  @autoreleasepool {
    if (!inputHandle || !outputHandle || !directory)
      return failure(2, "null Metal cumulative-sum argument");
    if (reverse != 0 && reverse != 1)
      return failure(2, "Metal cumulative-sum reverse flag must be zero or one");

    AMBuffer *input = (__bridge AMBuffer *)inputHandle;
    AMBuffer *output = (__bridge AMBuffer *)outputHandle;
    uint64_t inputCount = 0, outputCount = 0;
    AMStatus status = validateFloatTensor(input, inputDescriptor, inputCount);
    if (status.code) return status;
    status = validateFloatTensor(output, outputDescriptor, outputCount);
    if (status.code) return status;
    if (!sameDevice(input, output))
      return failure(2, "Metal cumulative-sum buffers use different devices");
    if (input == output || input.buffer == output.buffer ||
        !sameShape(inputDescriptor, outputDescriptor) ||
        inputCount != outputCount)
      return failure(2, "Metal cumulative-sum output must be independent and input-shaped");
    if (axis < 0 || (inputDescriptor.rank == 0
            ? axis != 0 : axis >= inputDescriptor.rank))
      return failure(2, "Metal cumulative-sum axis is outside rank");
    if (outputCount && outputDescriptor.offset != 0)
      return failure(2, "Metal cumulative-sum output must be zero-offset contiguous storage");
    if (outputCount) {
      uint64_t expectedStride = 1;
      for (int dim = outputDescriptor.rank - 1; dim >= 0; --dim) {
        if (outputDescriptor.strides[dim] != (int64_t)expectedStride)
          return failure(2, "Metal cumulative-sum output must be contiguous");
        const __uint128_t next = (__uint128_t)expectedStride *
            (uint64_t)outputDescriptor.shape[dim];
        if (next > UINT64_MAX)
          return failure(2, "Metal cumulative-sum output shape overflows");
        expectedStride = (uint64_t)next;
      }
    }
    if (!outputCount) return ok();

    const bool scalar = inputDescriptor.rank == 0;
    uint64_t outerCount = 1, innerCount = 1, axisLength = 1;
    if (!scalar) {
      axisLength = (uint64_t)inputDescriptor.shape[axis];
      for (int dim = 0; dim < axis; ++dim)
        outerCount *= (uint64_t)inputDescriptor.shape[dim];
      for (int dim = axis + 1; dim < inputDescriptor.rank; ++dim)
        innerCount *= (uint64_t)inputDescriptor.shape[dim];
    }

    AMDeviceState *state = output.state;
    if (!state || !state.device)
      return failure(1, "Metal cumulative-sum output has no device state");
    NSError *error = nil;
    AMCompute *context = compute(output.buffer.device, directory, &error);
    if (!context)
      return failure(9, error.localizedDescription.UTF8String
          ?: "Metal queue or library unavailable");

    const uint64_t columnTileCount = innerCount / 4ul +
        (innerCount % 4ul != 0);
    const uint64_t groupCount = scalar ? 1 : (innerCount == 1
        ? outerCount : outerCount * columnTileCount);
    const __uint128_t maxGridGroups =
        (__uint128_t)UINT32_MAX * (__uint128_t)UINT32_MAX;
    if (!groupCount || (__uint128_t)groupCount > maxGridGroups)
      return failure(2, "Metal cumulative-sum grid exceeds index limits");
    const uint64_t gridWidth = MIN(groupCount, (uint64_t)UINT32_MAX);
    const uint64_t gridHeight = groupCount / gridWidth +
        (groupCount % gridWidth != 0);
    if (gridHeight > UINT32_MAX)
      return failure(2, "Metal cumulative-sum grid exceeds index limits");

    id<MTLComputePipelineState> pipeline = pipelineFor(context,
        scalar ? @"cumsum_scalar" :
        (innerCount == 1 ? @"cumsum_rows" : @"cumsum_columns"), &error);
    if (!pipeline)
      return failure(error ? 7 : 8, error.localizedDescription.UTF8String
          ?: "missing Metal cumulative-sum kernel");
    const NSUInteger threadCount = scalar ? 1 :
        (innerCount == 1 ? pipeline.threadExecutionWidth : 128);
    if (scalar) {
      if (pipeline.maxTotalThreadsPerThreadgroup < threadCount)
        return failure(7, "Metal scalar cumulative-sum kernel cannot run");
    } else {
      const NSUInteger simdWidth = pipeline.threadExecutionWidth;
      if (!simdWidth || pipeline.maxTotalThreadsPerThreadgroup < threadCount ||
          (innerCount != 1 && (128 % simdWidth || 128 / simdWidth > 4)))
        return failure(7, "Metal cumulative-sum pipeline has an unsupported SIMD width");
    }

    AMStream *stream = nil;
    status = selectedStream(output.buffer.device, context.state, &stream);
    if (status.code) return status;
    @synchronized(stream.encodingLock) {
      id<MTLCommandBuffer> command = mathCommandBuffer(stream);
      if (!command)
        return failure(5, "cannot create Metal cumulative-sum command");
      id<MTLComputeCommandEncoder> encoder =
          mathComputeEncoder(stream, command, output, true);
      if (!encoder)
        return failure(5, "cannot create Metal cumulative-sum encoder");

      const uint64_t invocation = nextInvocation.fetch_add(1);
      AMEvidence evidence = {invocation, output.buffer.device.registryID,
          0, 0, 0, 0, 2, reverse ? 132 : 131};
      AMBuffer *__unsafe_unretained mathBuffers[2] = {input, output};
      flushPackedMath(stream, encoder);
      orderMathAccess(stream, output, input, nil, nil, true);
      status = admitMathCommand(stream, command, @[], output, evidence,
          pipeline, mathBuffers, 2);
      if (status.code) return status;

      @try {
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer offset:0 atIndex:0];
        [encoder setBuffer:output.buffer offset:0 atIndex:1];
        if (scalar) {
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:2];
          [encoder setBytes:&outputDescriptor length:sizeof(outputDescriptor) atIndex:3];
          [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        } else {
          const uint64_t columns = innerCount;
          const uint32_t nativeGridWidth = (uint32_t)gridWidth;
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:2];
          const int32_t scanAxis = axis;
          [encoder setBytes:&scanAxis length:sizeof(scanAxis) atIndex:3];
          [encoder setBytes:&axisLength length:sizeof(axisLength) atIndex:4];
          [encoder setBytes:&columns length:sizeof(columns) atIndex:5];
          if (innerCount == 1) {
            [encoder setBytes:&nativeGridWidth length:sizeof(nativeGridWidth) atIndex:6];
            [encoder setBytes:&reverse length:sizeof(reverse) atIndex:7];
          } else {
            [encoder setBytes:&columnTileCount length:sizeof(columnTileCount) atIndex:6];
            [encoder setBytes:&nativeGridWidth length:sizeof(nativeGridWidth) atIndex:7];
            [encoder setBytes:&reverse length:sizeof(reverse) atIndex:8];
          }
          [encoder dispatchThreadgroups:MTLSizeMake(gridWidth, gridHeight, 1)
              threadsPerThreadgroup:MTLSizeMake(threadCount, 1, 1)];
        }
      } @catch (NSException *exception) {
        return failMathEncoding(stream, invocation, exception.reason);
      }
      return finishMathEncoding(stream);
    }
  }
}

AMStatus am_cumprod(void *inputHandle, AMTensorDescriptor inputDescriptor,
                    void *outputHandle, AMTensorDescriptor outputDescriptor,
                    int32_t axis, const char *directory) {
  @autoreleasepool {
    if (!inputHandle || !outputHandle || !directory)
      return failure(2, "null Metal cumulative-product argument");

    AMBuffer *input = (__bridge AMBuffer *)inputHandle;
    AMBuffer *output = (__bridge AMBuffer *)outputHandle;
    uint64_t inputCount = 0, outputCount = 0;
    AMStatus status = validateFloatTensor(input, inputDescriptor, inputCount);
    if (status.code) return status;
    status = validateFloatTensor(output, outputDescriptor, outputCount);
    if (status.code) return status;
    if (!sameDevice(input, output))
      return failure(2, "Metal cumulative-product buffers use different devices");
    if (input == output || input.buffer == output.buffer ||
        !sameShape(inputDescriptor, outputDescriptor) ||
        inputCount != outputCount)
      return failure(2, "Metal cumulative-product output must be independent and input-shaped");
    if (axis < 0 || (inputDescriptor.rank == 0
            ? axis != 0 : axis >= inputDescriptor.rank))
      return failure(2, "Metal cumulative-product axis is outside rank");
    if (outputCount) {
      if (outputDescriptor.offset != 0)
        return failure(2, "Metal cumulative-product output must be zero-offset contiguous storage");
      uint64_t expectedStride = 1;
      for (int dim = outputDescriptor.rank - 1; dim >= 0; --dim) {
        if (outputDescriptor.strides[dim] != (int64_t)expectedStride)
          return failure(2, "Metal cumulative-product output must be contiguous");
        const __uint128_t next = (__uint128_t)expectedStride *
            (uint64_t)outputDescriptor.shape[dim];
        if (next > UINT64_MAX)
          return failure(2, "Metal cumulative-product output shape overflows");
        expectedStride = (uint64_t)next;
      }
    }
    if (!outputCount) return ok();

    const bool scalar = inputDescriptor.rank == 0;
    uint64_t outerCount = 1, innerCount = 1, axisLength = 1;
    if (!scalar) {
      axisLength = (uint64_t)inputDescriptor.shape[axis];
      for (int dim = 0; dim < axis; ++dim)
        outerCount *= (uint64_t)inputDescriptor.shape[dim];
      for (int dim = axis + 1; dim < inputDescriptor.rank; ++dim)
        innerCount *= (uint64_t)inputDescriptor.shape[dim];
    }
    const uint64_t lineCount = outerCount * innerCount;
    const bool simdLines = !scalar && (innerCount == 1 || axisLength <= 128);
    constexpr uint64_t threadsPerGroup = 8;
    const uint64_t groupCount = simdLines ? lineCount :
        lineCount / threadsPerGroup + (lineCount % threadsPerGroup != 0);
    const __uint128_t maxGridGroups =
        (__uint128_t)UINT32_MAX * (__uint128_t)UINT32_MAX;
    if (!groupCount || (__uint128_t)groupCount > maxGridGroups)
      return failure(2, "Metal cumulative-product grid exceeds index limits");
    const uint64_t gridWidth = MIN(groupCount, (uint64_t)UINT32_MAX);
    const uint64_t gridHeight = groupCount / gridWidth +
        (groupCount % gridWidth != 0);
    if (gridHeight > UINT32_MAX)
      return failure(2, "Metal cumulative-product grid exceeds index limits");

    AMDeviceState *state = output.state;
    if (!state || !state.device)
      return failure(1, "Metal cumulative-product output has no device state");
    NSError *error = nil;
    AMCompute *context = compute(output.buffer.device, directory, &error);
    if (!context)
      return failure(9, error.localizedDescription.UTF8String
          ?: "Metal queue or library unavailable");

    const bool packedScan = !scalar && axisLength <= 128 &&
        lineCount <= UINT32_MAX && output.state.supportsIndirectMath &&
        packedMathInput(inputDescriptor);
    id<MTLComputePipelineState> pipeline = pipelineFor(context,
        scalar ? @"cumprod_scalar" : (packedScan ? @"cumprod_short_tasks" :
            (simdLines ? @"cumprod_short_lines" : @"cumprod_lines")), &error);
    if (!pipeline)
      return failure(error ? 7 : 8, error.localizedDescription.UTF8String
          ?: "missing Metal cumulative-product kernel");
    const NSUInteger threadCount = scalar ? 1 :
        (simdLines ? pipeline.threadExecutionWidth : (NSUInteger)threadsPerGroup);
    if (pipeline.maxTotalThreadsPerThreadgroup < threadCount)
      return failure(7, "Metal cumulative-product pipeline cannot run its threadgroup");

    AMStream *stream = nil;
    status = selectedStream(output.buffer.device, context.state, &stream);
    if (status.code) return status;
    @synchronized(stream.encodingLock) {
      id<MTLCommandBuffer> command = mathCommandBuffer(stream);
      if (!command)
        return failure(5, "cannot create Metal cumulative-product command");
      id<MTLComputeCommandEncoder> encoder =
          mathComputeEncoder(stream, command, output, true);
      if (!encoder)
        return failure(5, "cannot create Metal cumulative-product encoder");

      const uint64_t invocation = nextInvocation.fetch_add(1);
      AMEvidence evidence = {invocation, output.buffer.device.registryID,
          0, 0, 0, 0, 2, 133};
      AMBuffer *__unsafe_unretained mathBuffers[2] = {input, output};
      if (!packedScan || (stream->packedTaskCount &&
          (stream->packedTaskCount == 32 || stream.packedPipeline != pipeline ||
           stream->packedGroups.width != gridWidth ||
           stream->packedThreads.width != threadCount)))
        flushPackedMath(stream, encoder);
      orderMathAccess(stream, output, input, nil, nil, true);
      status = admitMathCommand(stream, command, @[], output, evidence,
          pipeline, mathBuffers, 2);
      if (status.code) return status;

      @try {
        if (packedScan) {
          if (!stream->packedTaskCount) {
            stream.packedPipeline = pipeline;
            stream->packedGroups = MTLSizeMake(gridWidth, 1, 1);
            stream->packedThreads = MTLSizeMake(threadCount, 1, 1);
            stream->packedRows = true;
            stream->packedRepeatedWrites = false;
          }
          stream->packedTasks[stream->packedTaskCount++] = {
            input.gpuAddress + (uint64_t)inputDescriptor.offset * sizeof(float),
            innerCount, output.gpuAddress, (uint32_t)axisLength, (uint32_t)lineCount};
          if (!stream.aggregateMath) flushPackedMath(stream, encoder);
          return finishMathEncoding(stream);
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.buffer offset:0 atIndex:0];
        [encoder setBuffer:output.buffer offset:0 atIndex:1];
        if (scalar) {
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:2];
          [encoder setBytes:&outputDescriptor length:sizeof(outputDescriptor) atIndex:3];
          [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        } else {
          const int32_t scanAxis = axis;
          const uint32_t nativeGridWidth = (uint32_t)gridWidth;
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:2];
          [encoder setBytes:&scanAxis length:sizeof(scanAxis) atIndex:3];
          [encoder setBytes:&axisLength length:sizeof(axisLength) atIndex:4];
          [encoder setBytes:&innerCount length:sizeof(innerCount) atIndex:5];
          [encoder setBytes:&lineCount length:sizeof(lineCount) atIndex:6];
          [encoder setBytes:&nativeGridWidth length:sizeof(nativeGridWidth) atIndex:7];
          [encoder dispatchThreadgroups:MTLSizeMake(gridWidth, gridHeight, 1)
              threadsPerThreadgroup:MTLSizeMake(threadCount, 1, 1)];
        }
      } @catch (NSException *exception) {
        return failMathEncoding(stream, invocation, exception.reason);
      }
      return finishMathEncoding(stream);
    }
  }
}

AMStatus am_cumprod_gradient(
    void *inputHandle, AMTensorDescriptor inputDescriptor,
    void *prefixesHandle, AMTensorDescriptor prefixesDescriptor,
    void *seedHandle, AMTensorDescriptor seedDescriptor,
    void *outputHandle, AMTensorDescriptor outputDescriptor,
    int32_t axis, const char *directory) {
  @autoreleasepool {
    if (!inputHandle || !prefixesHandle || !seedHandle || !outputHandle ||
        !directory)
      return failure(2, "null Metal cumulative-product-gradient argument");

    AMBuffer *input = (__bridge AMBuffer *)inputHandle;
    AMBuffer *prefixes = (__bridge AMBuffer *)prefixesHandle;
    AMBuffer *seed = (__bridge AMBuffer *)seedHandle;
    AMBuffer *output = (__bridge AMBuffer *)outputHandle;
    uint64_t inputCount = 0, prefixesCount = 0, seedCount = 0, outputCount = 0;
    AMStatus status = validateFloatTensor(input, inputDescriptor, inputCount);
    if (status.code) return status;
    status = validateFloatTensor(prefixes, prefixesDescriptor, prefixesCount);
    if (status.code) return status;
    status = validateFloatTensor(seed, seedDescriptor, seedCount);
    if (status.code) return status;
    status = validateFloatTensor(output, outputDescriptor, outputCount);
    if (status.code) return status;

    if (!sameDevice(input, output) || !sameDevice(prefixes, output) ||
        !sameDevice(seed, output))
      return failure(2, "Metal cumulative-product-gradient buffers use different devices");
    if (output == input || output == prefixes || output == seed ||
        output.buffer == input.buffer || output.buffer == prefixes.buffer ||
        output.buffer == seed.buffer || !sameShape(inputDescriptor, outputDescriptor) ||
        inputCount != outputCount)
      return failure(2, "Metal cumulative-product-gradient output must be independent and input-shaped");
    if (!sameShape(inputDescriptor, prefixesDescriptor) ||
        !sameShape(inputDescriptor, seedDescriptor) ||
        prefixesCount != inputCount || seedCount != inputCount)
      return failure(2, "Metal cumulative-product prefixes and seed must be input-shaped");
    if (axis < 0 || (inputDescriptor.rank == 0
            ? axis != 0 : axis >= inputDescriptor.rank))
      return failure(2, "Metal cumulative-product-gradient axis is outside rank");
    if (outputCount) {
      if (outputDescriptor.offset != 0)
        return failure(2, "Metal cumulative-product-gradient output must be zero-offset contiguous storage");
      uint64_t expectedStride = 1;
      for (int dim = outputDescriptor.rank - 1; dim >= 0; --dim) {
        if (outputDescriptor.strides[dim] != (int64_t)expectedStride)
          return failure(2, "Metal cumulative-product-gradient output must be contiguous");
        const __uint128_t next = (__uint128_t)expectedStride *
            (uint64_t)outputDescriptor.shape[dim];
        if (next > UINT64_MAX)
          return failure(2, "Metal cumulative-product-gradient output shape overflows");
        expectedStride = (uint64_t)next;
      }
    }
    if (!outputCount) return ok();

    const bool scalar = inputDescriptor.rank == 0;
    uint64_t outerCount = 1, innerCount = 1, axisLength = 1;
    if (!scalar) {
      axisLength = (uint64_t)inputDescriptor.shape[axis];
      for (int dim = 0; dim < axis; ++dim)
        outerCount *= (uint64_t)inputDescriptor.shape[dim];
      for (int dim = axis + 1; dim < inputDescriptor.rank; ++dim)
        innerCount *= (uint64_t)inputDescriptor.shape[dim];
    }
    const bool singletonAxis = !scalar && axisLength == 1;
    const uint64_t lineCount = outerCount * innerCount;
    constexpr uint64_t threadsPerGroup = 16;
    const uint64_t groupCount = lineCount / threadsPerGroup +
        (lineCount % threadsPerGroup != 0);
    const __uint128_t maxGridGroups =
        (__uint128_t)UINT32_MAX * (__uint128_t)UINT32_MAX;
    if (!groupCount || (__uint128_t)groupCount > maxGridGroups)
      return failure(2, "Metal cumulative-product-gradient grid exceeds index limits");
    const uint64_t gridWidth = MIN(groupCount, (uint64_t)UINT32_MAX);
    const uint64_t gridHeight = groupCount / gridWidth +
        (groupCount % gridWidth != 0);
    if (gridHeight > UINT32_MAX)
      return failure(2, "Metal cumulative-product-gradient grid exceeds index limits");

    AMDeviceState *state = output.state;
    if (!state || !state.device)
      return failure(1, "Metal cumulative-product-gradient output has no device state");
    NSError *error = nil;
    AMCompute *context = compute(output.buffer.device, directory, &error);
    if (!context)
      return failure(9, error.localizedDescription.UTF8String
          ?: "Metal queue or library unavailable");

    NSString *kernelName = scalar ? @"cumprod_gradient_scalar" :
        (singletonAxis ? @"cumprod_gradient_singleton" : @"cumprod_gradient_lines");
    id<MTLComputePipelineState> pipeline = pipelineFor(context, kernelName, &error);
    if (!pipeline)
      return failure(error ? 7 : 8, error.localizedDescription.UTF8String
          ?: "missing Metal cumulative-product-gradient kernel");
    const NSUInteger threadCount = scalar ? 1 : (NSUInteger)threadsPerGroup;
    if (pipeline.maxTotalThreadsPerThreadgroup < threadCount)
      return failure(7, "Metal cumulative-product-gradient pipeline cannot run its threadgroup");

    AMStream *stream = nil;
    status = selectedStream(output.buffer.device, context.state, &stream);
    if (status.code) return status;
    @synchronized(stream.encodingLock) {
      id<MTLCommandBuffer> command = mathCommandBuffer(stream);
      if (!command)
        return failure(5, "cannot create Metal cumulative-product-gradient command");
      id<MTLComputeCommandEncoder> encoder =
          mathComputeEncoder(stream, command, output, true);
      if (!encoder)
        return failure(5, "cannot create Metal cumulative-product-gradient encoder");

      const uint64_t invocation = nextInvocation.fetch_add(1);
      AMEvidence evidence = {invocation, output.buffer.device.registryID,
          0, 0, 0, 0, 2, 134};
      AMBuffer *__unsafe_unretained mathBuffers[4];
      NSUInteger mathBufferCount = 0;
      AMBuffer *candidates[] = {input, prefixes, seed, output};
      for (AMBuffer *candidate : candidates) {
        bool alreadyPresent = false;
        for (NSUInteger index = 0; index < mathBufferCount; ++index)
          if (mathBuffers[index] == candidate) alreadyPresent = true;
        if (!alreadyPresent) mathBuffers[mathBufferCount++] = candidate;
      }
      flushPackedMath(stream, encoder);
      orderMathAccess(stream, output, input, prefixes, seed, true);
      status = admitMathCommand(stream, command, @[], output, evidence,
          pipeline, mathBuffers, mathBufferCount);
      if (status.code) return status;

      @try {
        [encoder setComputePipelineState:pipeline];
        if (scalar) {
          [encoder setBuffer:seed.buffer offset:0 atIndex:0];
          [encoder setBuffer:output.buffer offset:0 atIndex:1];
          [encoder setBytes:&seedDescriptor length:sizeof(seedDescriptor) atIndex:2];
          [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        } else if (singletonAxis) {
          const int32_t scanAxis = axis;
          const uint32_t nativeGridWidth = (uint32_t)gridWidth;
          [encoder setBuffer:seed.buffer offset:0 atIndex:0];
          [encoder setBuffer:output.buffer offset:0 atIndex:1];
          [encoder setBytes:&seedDescriptor length:sizeof(seedDescriptor) atIndex:2];
          [encoder setBytes:&scanAxis length:sizeof(scanAxis) atIndex:3];
          [encoder setBytes:&innerCount length:sizeof(innerCount) atIndex:4];
          [encoder setBytes:&lineCount length:sizeof(lineCount) atIndex:5];
          [encoder setBytes:&nativeGridWidth length:sizeof(nativeGridWidth) atIndex:6];
          [encoder dispatchThreadgroups:MTLSizeMake(gridWidth, gridHeight, 1)
              threadsPerThreadgroup:MTLSizeMake(threadCount, 1, 1)];
        } else {
          const int32_t scanAxis = axis;
          const uint32_t nativeGridWidth = (uint32_t)gridWidth;
          [encoder setBuffer:input.buffer offset:0 atIndex:0];
          [encoder setBuffer:prefixes.buffer offset:0 atIndex:1];
          [encoder setBuffer:seed.buffer offset:0 atIndex:2];
          [encoder setBuffer:output.buffer offset:0 atIndex:3];
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:4];
          [encoder setBytes:&prefixesDescriptor length:sizeof(prefixesDescriptor) atIndex:5];
          [encoder setBytes:&seedDescriptor length:sizeof(seedDescriptor) atIndex:6];
          [encoder setBytes:&scanAxis length:sizeof(scanAxis) atIndex:7];
          [encoder setBytes:&axisLength length:sizeof(axisLength) atIndex:8];
          [encoder setBytes:&innerCount length:sizeof(innerCount) atIndex:9];
          [encoder setBytes:&lineCount length:sizeof(lineCount) atIndex:10];
          [encoder setBytes:&nativeGridWidth length:sizeof(nativeGridWidth) atIndex:11];
          [encoder dispatchThreadgroups:MTLSizeMake(gridWidth, gridHeight, 1)
              threadsPerThreadgroup:MTLSizeMake(threadCount, 1, 1)];
        }
      } @catch (NSException *exception) {
        return failMathEncoding(stream, invocation, exception.reason);
      }
      return finishMathEncoding(stream);
    }
  }
}


AMStatus am_product_gradient(void *inputHandle, AMTensorDescriptor inputDescriptor,
                             void *productHandle, AMTensorDescriptor pd,
                             void *seedHandle, AMTensorDescriptor sd,
                             void *outputHandle, AMTensorDescriptor od,
                             int32_t axis, const char *directory) {
  @autoreleasepool {
    if (!inputHandle || !productHandle || !seedHandle || !outputHandle ||
        !directory)
      return failure(2, "null Metal product-gradient argument");

    AMBuffer *input = (__bridge AMBuffer *)inputHandle;
    AMBuffer *product = (__bridge AMBuffer *)productHandle;
    AMBuffer *seed = (__bridge AMBuffer *)seedHandle;
    AMBuffer *output = (__bridge AMBuffer *)outputHandle;
    uint64_t inputCount = 0, productCount = 0, seedCount = 0, outputCount = 0;
    AMStatus status = validateFloatTensor(input, inputDescriptor, inputCount);
    if (status.code) return status;
    status = validateFloatTensor(product, pd, productCount);
    if (status.code) return status;
    status = validateFloatTensor(seed, sd, seedCount);
    if (status.code) return status;
    status = validateFloatTensor(output, od, outputCount);
    if (status.code) return status;

    if (!sameDevice(input, output) || !sameDevice(product, output) ||
        !sameDevice(seed, output))
      return failure(2, "Metal product-gradient buffers use different devices");
    if (output == input || output == product || output == seed ||
        !sameShape(inputDescriptor, od) || inputCount != outputCount)
      return failure(2, "Metal product-gradient output must be independent and input-shaped");
    if (axis < -1 || (axis >= 0 &&
        (inputDescriptor.rank == 0 ? axis != 0 : axis >= inputDescriptor.rank)))
      return failure(2, "Metal product-gradient axis is outside rank");

    if (axis == -1) {
      if (pd.rank != 0 || sd.rank != 0 ||
          productCount != 1 || seedCount != 1)
        return failure(2, "full product gradient requires scalar product and seed");
    } else if (inputDescriptor.rank == 0) {
      if (pd.rank != 0 || sd.rank != 0 ||
          productCount != 1 || seedCount != 1)
        return failure(2, "scalar product gradient requires scalar product and seed");
    } else {
      if (pd.rank != inputDescriptor.rank || sd.rank != inputDescriptor.rank ||
          productCount != seedCount)
        return failure(2, "axis product gradient requires matching product and seed ranks");
      for (int dim = 0; dim < inputDescriptor.rank; ++dim) {
        const int64_t expected = dim == axis ? 1 : inputDescriptor.shape[dim];
        if (pd.shape[dim] != expected || sd.shape[dim] != expected)
          return failure(2, "axis product and seed must keep the reduced dimension");
      }
    }
    if (outputCount > UINT32_MAX)
      return failure(2, "Metal product-gradient output exceeds grid limits");
    if (!outputCount) return ok();

    AMDeviceState *state = output.state;
    if (!state || !state.device)
      return failure(1, "Metal product-gradient output has no device state");
    NSError *error = nil;
    AMCompute *context = compute(output.buffer.device, directory, &error);
    if (!context)
      return failure(9, error.localizedDescription.UTF8String
          ?: "Metal queue or library unavailable");

    const bool scalarInput = inputDescriptor.rank == 0;
    const bool fullReduction = axis == -1;
    const uint64_t sliceCount = inputDescriptor.rank == 0 ? 1 : productCount;
    const uint64_t axisLength = axis >= 0 && inputDescriptor.rank != 0
        ? (uint64_t)inputDescriptor.shape[axis] : 1;
    const bool singletonAxis = !scalarInput && !fullReduction && axisLength == 1;
    uint64_t partialCount = 0;
    id<MTLBuffer> scratch = nil;
    if (!scalarInput && !singletonAxis) {
      partialCount = fullReduction
          ? (inputCount + 65535ul) / 65536ul : sliceCount;
      const __uint128_t scratchBytes =
          (__uint128_t)partialCount * sizeof(uint32_t);
      if (!partialCount || scratchBytes > NSUIntegerMax)
        return failure(2, "Metal product-gradient scratch size overflows");
      scratch = [state.device newBufferWithLength:(NSUInteger)scratchBytes
          options:MTLResourceStorageModePrivate];
      if (!scratch)
        return failure(4, "cannot allocate Metal product-gradient statistics");
    }

    id<MTLComputePipelineState> countPipeline = nil;
    id<MTLComputePipelineState> finalizePipeline = nil;
    id<MTLComputePipelineState> outputPipeline = nil;
    id<MTLComputePipelineState> scalarPipeline = nil;
    id<MTLComputePipelineState> singletonPipeline = nil;
    if (scalarInput) {
      scalarPipeline = pipelineFor(context, @"math_product_gradient_scalar", &error);
      if (!scalarPipeline)
        return failure(error ? 7 : 8, error.localizedDescription.UTF8String
            ?: "missing Metal scalar product-gradient kernel");
    } else if (singletonAxis) {
      singletonPipeline = pipelineFor(context, @"math_product_gradient_axis_singleton", &error);
      if (!singletonPipeline)
        return failure(error ? 7 : 8, error.localizedDescription.UTF8String
            ?: "missing Metal singleton product-gradient kernel");
    } else if (fullReduction) {
      countPipeline = pipelineFor(context, @"math_product_gradient_count_full", &error);
      if (!countPipeline)
        return failure(error ? 7 : 8, error.localizedDescription.UTF8String
            ?: "missing Metal full product-gradient count kernel");
      if (partialCount > 1) {
        finalizePipeline = pipelineFor(context, @"math_product_gradient_finalize_full", &error);
        if (!finalizePipeline)
          return failure(error ? 7 : 8, error.localizedDescription.UTF8String
              ?: "missing Metal full product-gradient finalize kernel");
      }
      outputPipeline = pipelineFor(context, @"math_product_gradient_full", &error);
      if (!outputPipeline)
        return failure(error ? 7 : 8, error.localizedDescription.UTF8String
            ?: "missing Metal full product-gradient output kernel");
    } else {
      countPipeline = pipelineFor(context, @"math_product_gradient_count_axis", &error);
      if (!countPipeline)
        return failure(error ? 7 : 8, error.localizedDescription.UTF8String
            ?: "missing Metal axis product-gradient count kernel");
      outputPipeline = pipelineFor(context, @"math_product_gradient_axis", &error);
      if (!outputPipeline)
        return failure(error ? 7 : 8, error.localizedDescription.UTF8String
            ?: "missing Metal axis product-gradient output kernel");
    }
    if (scalarInput) {
      if (scalarPipeline.maxTotalThreadsPerThreadgroup < 1)
        return failure(7, "Metal scalar product-gradient pipeline cannot run");
    } else if (singletonAxis) {
      if (singletonPipeline.maxTotalThreadsPerThreadgroup < 256)
        return failure(7, "Metal singleton product-gradient pipeline cannot run");
    } else if (countPipeline.maxTotalThreadsPerThreadgroup < 256 ||
               outputPipeline.maxTotalThreadsPerThreadgroup < 256 ||
               (fullReduction && partialCount > 1 &&
                finalizePipeline.maxTotalThreadsPerThreadgroup < 256)) {
      return failure(7, "Metal product-gradient pipeline cannot run its threadgroup");
    }

    if (partialCount > UINT32_MAX)
      return failure(2, "Metal product-gradient grid exceeds index limits");
    AMStream *stream = nil;
    status = selectedStream(output.buffer.device, context.state, &stream);
    if (status.code) return status;
    @synchronized(stream.encodingLock) {
      id<MTLCommandBuffer> command = mathCommandBuffer(stream);
      if (!command)
        return failure(5, "cannot create Metal product-gradient command");
      id<MTLComputeCommandEncoder> encoder =
          mathComputeEncoder(stream, command, output, true);
      if (!encoder)
        return failure(5, "cannot create Metal product-gradient encoder");

      const uint64_t invocation = nextInvocation.fetch_add(1);
      AMEvidence evidence = {invocation, output.buffer.device.registryID,
                             0, 0, 0, 0, 2, 128};
      AMBuffer *__unsafe_unretained mathBuffers[4];
      NSUInteger mathBufferCount = 0;
      AMBuffer *candidates[] = {input, product, seed, output};
      for (AMBuffer *candidate : candidates) {
        bool alreadyPresent = false;
        for (NSUInteger index = 0; index < mathBufferCount; ++index)
          if (mathBuffers[index] == candidate) alreadyPresent = true;
        if (!alreadyPresent) mathBuffers[mathBufferCount++] = candidate;
      }
      flushPackedMath(stream, encoder);
      orderMathAccess(stream, output, input, product, seed, true);
      id<MTLComputePipelineState> leasedPipeline = scalarInput
          ? scalarPipeline : (singletonAxis ? singletonPipeline : outputPipeline);
      status = admitMathCommand(stream, command,
          scratch ? @[scratch] : @[], output, evidence, leasedPipeline,
          mathBuffers, mathBufferCount);
      if (status.code) return status;
      if (scratch)
        [encoder useResource:scratch
            usage:MTLResourceUsageRead | MTLResourceUsageWrite];

      @try {
        const MTLSize threads = MTLSizeMake(256, 1, 1);
        if (scalarInput) {
          [encoder setComputePipelineState:scalarPipeline];
          [encoder setBuffer:seed.buffer offset:0 atIndex:0];
          [encoder setBuffer:output.buffer offset:0 atIndex:1];
          [encoder setBytes:&sd length:sizeof(sd) atIndex:2];
          [encoder setBytes:&od length:sizeof(od) atIndex:3];
          [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        } else if (singletonAxis) {
          [encoder setComputePipelineState:singletonPipeline];
          [encoder setBuffer:input.buffer offset:0 atIndex:0];
          [encoder setBuffer:product.buffer offset:0 atIndex:1];
          [encoder setBuffer:seed.buffer offset:0 atIndex:2];
          [encoder setBuffer:output.buffer offset:0 atIndex:3];
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:4];
          [encoder setBytes:&pd length:sizeof(pd) atIndex:5];
          [encoder setBytes:&sd length:sizeof(sd) atIndex:6];
          [encoder setBytes:&od length:sizeof(od) atIndex:7];
          [encoder setBytes:&inputCount length:sizeof(inputCount) atIndex:8];
          [encoder dispatchThreads:MTLSizeMake(inputCount, 1, 1)
              threadsPerThreadgroup:threads];
        } else if (fullReduction) {
          [encoder setComputePipelineState:countPipeline];
          [encoder setBuffer:input.buffer offset:0 atIndex:0];
          [encoder setBuffer:scratch offset:0 atIndex:1];
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:2];
          [encoder setBytes:&inputCount length:sizeof(inputCount) atIndex:3];
          [encoder dispatchThreadgroups:MTLSizeMake(partialCount, 1, 1)
              threadsPerThreadgroup:threads];
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

          if (partialCount > 1) {
            [encoder setComputePipelineState:finalizePipeline];
            [encoder setBuffer:scratch offset:0 atIndex:0];
            [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:threads];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
          }

          [encoder setComputePipelineState:outputPipeline];
          [encoder setBuffer:input.buffer offset:0 atIndex:0];
          [encoder setBuffer:product.buffer offset:0 atIndex:1];
          [encoder setBuffer:seed.buffer offset:0 atIndex:2];
          [encoder setBuffer:output.buffer offset:0 atIndex:3];
          [encoder setBuffer:scratch offset:0 atIndex:4];
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:5];
          [encoder setBytes:&pd length:sizeof(pd) atIndex:6];
          [encoder setBytes:&sd length:sizeof(sd) atIndex:7];
          [encoder setBytes:&od length:sizeof(od) atIndex:8];
          [encoder setBytes:&inputCount length:sizeof(inputCount) atIndex:9];
          [encoder dispatchThreads:MTLSizeMake(inputCount, 1, 1)
              threadsPerThreadgroup:threads];
        } else {
          const int32_t reducedAxis = axis;
          [encoder setComputePipelineState:countPipeline];
          [encoder setBuffer:input.buffer offset:0 atIndex:0];
          [encoder setBuffer:scratch offset:0 atIndex:1];
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:2];
          [encoder setBytes:&reducedAxis length:sizeof(reducedAxis) atIndex:3];
          [encoder setBytes:&axisLength length:sizeof(axisLength) atIndex:4];
          [encoder dispatchThreadgroups:MTLSizeMake(sliceCount, 1, 1)
              threadsPerThreadgroup:threads];
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

          [encoder setComputePipelineState:outputPipeline];
          [encoder setBuffer:input.buffer offset:0 atIndex:0];
          [encoder setBuffer:product.buffer offset:0 atIndex:1];
          [encoder setBuffer:seed.buffer offset:0 atIndex:2];
          [encoder setBuffer:output.buffer offset:0 atIndex:3];
          [encoder setBuffer:scratch offset:0 atIndex:4];
          [encoder setBytes:&inputDescriptor length:sizeof(inputDescriptor) atIndex:5];
          [encoder setBytes:&pd length:sizeof(pd) atIndex:6];
          [encoder setBytes:&sd length:sizeof(sd) atIndex:7];
          [encoder setBytes:&od length:sizeof(od) atIndex:8];
          [encoder setBytes:&inputCount length:sizeof(inputCount) atIndex:9];
          [encoder setBytes:&sliceCount length:sizeof(sliceCount) atIndex:10];
          [encoder setBytes:&reducedAxis length:sizeof(reducedAxis) atIndex:11];
          [encoder setBytes:&axisLength length:sizeof(axisLength) atIndex:12];
          [encoder dispatchThreads:MTLSizeMake(inputCount, 2, 1)
              threadsPerThreadgroup:threads];
        }
      } @catch (NSException *exception) {
        return failMathEncoding(stream, invocation, exception.reason);
      }
      return finishMathEncoding(stream);
    }
  }
}
AMStatus am_mse_backward(void *upstreamHandle, AMTensorDescriptor ud,
                         void *predictionHandle, AMTensorDescriptor pd,
                         void *targetHandle, AMTensorDescriptor td,
                         void *outputHandle, AMTensorDescriptor od,
                         const char *directory) {
  @autoreleasepool {
    if (!upstreamHandle || !predictionHandle || !targetHandle ||
        !outputHandle || !directory)
      return failure(2, "null Metal MSE backward argument");

    AMBuffer *upstream = (__bridge AMBuffer *)upstreamHandle;
    AMBuffer *prediction = (__bridge AMBuffer *)predictionHandle;
    AMBuffer *target = (__bridge AMBuffer *)targetHandle;
    AMBuffer *output = (__bridge AMBuffer *)outputHandle;
    uint64_t upstreamCount = 0, predictionCount = 0, targetCount = 0, outputCount = 0;
    AMStatus status = validateFloatTensor(upstream, ud, upstreamCount);
    if (status.code) return status;
    status = validateFloatTensor(prediction, pd, predictionCount);
    if (status.code) return status;
    status = validateFloatTensor(target, td, targetCount);
    if (status.code) return status;
    status = validateFloatTensor(output, od, outputCount);
    if (status.code) return status;

    if (ud.rank != 0 || upstreamCount != 1)
      return failure(2, "Metal MSE upstream gradient must be a scalar");
    if (!sameShape(pd, td) || predictionCount != targetCount || od.rank > pd.rank)
      return failure(2, "Metal MSE prediction and target shapes must match");
    const int leading = pd.rank - od.rank;
    for (int axis = 0; axis < od.rank; ++axis)
      if (od.shape[axis] != 1 && od.shape[axis] != pd.shape[leading + axis])
        return failure(2, "Metal MSE output must reduce to an operand shape");
    const bool reduced = !sameShape(pd, od);
    if (!sameDevice(upstream, output) || !sameDevice(prediction, output) ||
        !sameDevice(target, output))
      return failure(2, "Metal MSE buffers use different devices");
    if (!outputCount) return ok();
    if (outputCount > UINT32_MAX)
      return failure(2, "Metal MSE backward output exceeds grid limits");

    NSError *error = nil;
    AMCompute *context = compute(output.buffer.device, directory, &error);
    if (!context)
      return failure(9, error ? error.localizedDescription.UTF8String : "Metal queue or library unavailable");
    NSString *name = reduced ? @"math_mse_backward_reduce" : @"math_mse_backward";
    id<MTLComputePipelineState> pipeline = pipelineFor(context, name, &error);
    if (!pipeline)
      return failure(error ? 7 : 8,
                     error ? error.localizedDescription.UTF8String : "missing Metal MSE backward kernel");
    const NSUInteger width = reduced ? 256 : MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    if (!width || width > pipeline.maxTotalThreadsPerThreadgroup)
      return failure(7, "Metal MSE backward pipeline cannot run its threadgroup");
    const NSUInteger groupCount = reduced ? outputCount : (outputCount + width - 1) / width;
    if (groupCount > UINT32_MAX || (!reduced && groupCount * width > UINT32_MAX))
      return failure(2, "Metal MSE backward grid exceeds index limits");

    AMStream *stream = nil;
    status = selectedStream(output.buffer.device, context.state, &stream);
    if (status.code) return status;
    @synchronized(stream.encodingLock) {
      id<MTLCommandBuffer> command = mathCommandBuffer(stream);
      if (!command) return failure(5, "cannot create Metal MSE backward command");
      id<MTLComputeCommandEncoder> encoder = mathComputeEncoder(stream, command, output, false);
      if (!encoder) return failure(5, "cannot create Metal MSE backward encoder");
      const uint64_t invocation = nextInvocation.fetch_add(1);
      AMEvidence evidence = {invocation, output.buffer.device.registryID,
                             0, 0, 0, 0, 2, 124};
      AMBuffer *__unsafe_unretained leases[4];
      NSUInteger leaseCount = 0;
      leases[leaseCount++] = upstream;
      if (prediction != upstream) leases[leaseCount++] = prediction;
      if (target != prediction && target != upstream) leases[leaseCount++] = target;
      if (output != upstream && output != prediction && output != target)
        leases[leaseCount++] = output;
      status = admitMathCommand(stream, command, nil, output, evidence, pipeline,
          leases, leaseCount);
      if (status.code) return status;
      @try {
        flushPackedMath(stream, encoder);
        orderMathAccess(stream, output, upstream, prediction, target);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:upstream.buffer offset:0 atIndex:0];
        [encoder setBuffer:prediction.buffer offset:0 atIndex:1];
        [encoder setBuffer:target.buffer offset:0 atIndex:2];
        [encoder setBuffer:output.buffer offset:0 atIndex:3];
        [encoder setBytes:&ud length:sizeof(ud) atIndex:4];
        [encoder setBytes:&pd length:sizeof(pd) atIndex:5];
        [encoder setBytes:&td length:sizeof(td) atIndex:6];
        [encoder setBytes:&od length:sizeof(od) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      } @catch (NSException *exception) {
        return failMathEncoding(stream, invocation, exception.reason);
      }
      return finishMathEncoding(stream);
    }
  }
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
    if (count > UINT64_MAX / width)
      return failure(2, "host transfer byte count overflows");
    const uint64_t transferBytes = count * width;
    AMBufferLease *lease = nil;
    status = beginBufferLease(owner, upload ? 1 : 0, &lease);
    if (status.code) return status;
    status = waitBufferLease(lease);
    if (status.code) {
      releaseBufferLease(lease);
      return status;
    }

    auto *native = (char *)owner.buffer.contents;
    auto *cpu = (char *)host;
    @synchronized(owner) {
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
      evidence.transfer_bytes = saturatedAdd(evidence.transfer_bytes, transferBytes);
      evidence.completed = 1;
      owner.evidence = evidence;
    }
    AMDeviceState *state = owner.state;
    @synchronized(state) {
      state.hostCopyBytes = saturatedAdd(state.hostCopyBytes, transferBytes);
    }
    releaseBufferLease(lease);
    return ok();
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
    if (!context) return failure(9, error ? error.localizedDescription.UTF8String : "Metal queue unavailable");
    NSString *name = [NSString stringWithFormat:@"copy%llu", dtypeWidth(input.dtype) * 8];
    id<MTLComputePipelineState> pipeline = pipelineFor(context, name, &error);
    if (!pipeline) return failure(8, error ? error.localizedDescription.UTF8String : "missing copy kernel");
    AMStream *stream = nil;
    status = selectedStream(src.buffer.device, context.state, &stream);
    if (status.code) return status;
    @synchronized(stream.encodingLock) {
      id<MTLCommandBuffer> command = mathCommandBuffer(stream);
      if (!command) return failure(5, "cannot create Metal copy command");
      id<MTLComputeCommandEncoder> encoder = mathComputeEncoder(stream, command, dst, false);
      if (!encoder) return failure(5, "cannot create Metal copy encoder");
      const uint64_t invocation = nextInvocation.fetch_add(1);
      AMEvidence evidence = {invocation, dst.buffer.device.registryID, 0,
        0, 0, 0, 2, (int32_t)(dtypeWidth(input.dtype) * 8)};
      AMBuffer *__unsafe_unretained leases[] = {src, dst};
      status = admitMathCommand(stream, command, nil, dst, evidence, pipeline,
          leases, 2, count * dtypeWidth(input.dtype));
      if (status.code) return status;
      @try {
        // A clone participates in the same producer/consumer ordering as arithmetic.
        flushPackedMath(stream, encoder);
        orderMathAccess(stream, dst, src);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:src.buffer offset:0 atIndex:0];
        [encoder setBuffer:dst.buffer offset:0 atIndex:1];
        [encoder setBytes:&input length:sizeof(input) atIndex:2];
        [encoder setBytes:&output length:sizeof(output) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(count, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
      } @catch (NSException *exception) {
        return failMathEncoding(stream, invocation, exception.reason);
      }
      return finishMathEncoding(stream);
    }
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
    if (!context) return failure(9, error ? error.localizedDescription.UTF8String : "Metal queue unavailable");
    NSString *name = [NSString stringWithFormat:@"set%llu", dtypeWidth(output.dtype) * 8];
    id<MTLComputePipelineState> pipeline = pipelineFor(context, name, &error);
    if (!pipeline) return failure(8, error ? error.localizedDescription.UTF8String : "missing mutation kernel");
    AMStream *stream = nil;
    status = selectedStream(dst.buffer.device, context.state, &stream);
    if (status.code) return status;
    id<MTLCommandBuffer> command = [stream.queue commandBuffer];
    if (!command) return failure(5, "cannot create Metal mutation command");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) return failure(5, "cannot create Metal mutation encoder");
    command.label = name;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dst.buffer offset:0 atIndex:0];
    [encoder setBytes:value length:dtypeWidth(output.dtype) atIndex:1];
    [encoder setBytes:&output length:sizeof(output) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    const uint64_t invocation = nextInvocation.fetch_add(1);
    AMEvidence evidence = {invocation, dst.buffer.device.registryID,
                           0, 0, 0, 0, 2, -(int32_t)(dtypeWidth(output.dtype) * 8)};
    status = submitCommand(stream, command, @[dst, pipeline], dst,
                           invocation, evidence, 0, nullptr);
    if (status.code) abandonBufferProducer(dst, invocation);
    return status;
  }
}
static AMDeviceState *memoryDeviceState(void *deviceHandle, AMStatus *status) {
  id<MTLDevice> device = deviceHandle
      ? (__bridge id<MTLDevice>)deviceHandle : MTLCopyAllDevices().firstObject;
  NSError *error = nil;
  AMDeviceState *state = deviceState(device, &error);
  if (!state && status)
    *status = failure(1, error.localizedDescription.UTF8String ?: "Metal device unavailable");
  return state;
}

AMStatus am_memory_stats(void *device, AMMemoryStats *out) {
  @autoreleasepool {
    if (!out) return failure(2, "null Metal memory statistics output");
    AMStatus status = ok();
    AMDeviceState *state = memoryDeviceState(device, &status);
    if (!state) return status;
    @synchronized(state) {
      out->live_bytes = state.liveBytes;
      out->cached_bytes = state.cachedBytes;
      out->in_flight_bytes = state.inFlightBytes;
      out->reserved_bytes = state.reservedBytes;
      out->host_copy_bytes = state.hostCopyBytes;
      out->materialization_bytes = state.materializationBytes;
      out->driver_bytes = state.device.currentAllocatedSize;
      out->recommended_bytes = state.device.recommendedMaxWorkingSetSize;
    }
    return ok();
  }
}

AMStatus am_memory_empty_cache(void *device) {
  @autoreleasepool {
    AMStatus status = ok();
    AMDeviceState *state = memoryDeviceState(device, &status);
    if (!state) return status;
    status = flushDeviceMathBatches(state.device);
    if (status.code) return status;
    @synchronized(state) {
      while (evictOneCachedOwnerLocked(state)) {}
    }
    return ok();
  }
}

AMStatus am_memory_fraction(void *device, double *value, int32_t set) {
  @autoreleasepool {
    if (!value || (set != 0 && set != 1))
      return failure(2, "invalid Metal memory-fraction argument");
    const double requested = *value;
    if (set && (!std::isfinite(requested) || requested < 0.0 || requested > 2.0))
      return failure(2, "Metal memory fraction must be finite and in [0, 2]");
    AMStatus status = ok();
    AMDeviceState *state = memoryDeviceState(device, &status);
    if (!state) return status;
    @synchronized(state) {
      if (set) state.memoryFraction = requested;
      *value = state.memoryFraction;
    }
    return ok();
  }
}

AMStatus am_capture_start(void *deviceHandle, const char *path, void **out) {
  @autoreleasepool {
    if (!path || !out) return failure(5, "null Metal capture path or output");
    *out = nullptr;
    id<MTLDevice> device = deviceHandle
        ? (__bridge id<MTLDevice>)deviceHandle : MTLCopyAllDevices().firstObject;
    if (!device) return failure(5, "Metal capture device unavailable");
    AMStatus status = flushDeviceMathBatches(device);
    if (status.code) return status;
    NSString *outputPath = [NSString stringWithUTF8String:path];
    if (!outputPath || !outputPath.isAbsolutePath)
      return failure(5, "Metal GPU trace output path must be absolute");

    MTLCaptureManager *manager = [MTLCaptureManager sharedCaptureManager];
    ensureCaptureLock();
    @synchronized(captureLock) {
      if (activeCapture || manager.isCapturing)
        return failure(5, "another Metal capture is already active");
      if (![manager supportsDestination:MTLCaptureDestinationGPUTraceDocument])
        return failure(5, "Metal GPU trace capture destination is unsupported");

      MTLCaptureDescriptor *descriptor = [MTLCaptureDescriptor new];
      descriptor.captureObject = device;
      descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
      descriptor.outputURL = [NSURL fileURLWithPath:outputPath];
      NSError *error = nil;
      BOOL started = NO;
      @try {
        started = [manager startCaptureWithDescriptor:descriptor error:&error];
      } @catch (NSException *exception) {
        return failure(5, exception.reason.UTF8String ?: "Metal capture start failed");
      }
      if (!started)
        return failure(5, error.localizedDescription.UTF8String ?: "Metal capture start failed");

      AMCapture *capture = [AMCapture new];
      capture.device = device;
      capture.manager = manager;
      capture.path = outputPath;
      capture.externalReferences = 1;
      capture.active = YES;
      activeCapture = capture;
      *out = (__bridge_retained void *)capture;
      return ok();
    }
  }
}

AMStatus am_capture_stop(void *captureHandle) {
  @autoreleasepool {
    if (!captureHandle) return failure(5, "null Metal capture handle");
    id object = (__bridge id)captureHandle;
    if (![object isKindOfClass:AMCapture.class])
      return failure(5, "invalid Metal capture handle");
    ensureCaptureLock();
    @synchronized(captureLock) {
      NSString *message = stopCaptureLocked((AMCapture *)object);
      return message ? failure(5, message.UTF8String) : ok();
    }
  }
}
static AMStatus checkedMetalDevice(void *deviceHandle,
                                   id<MTLDevice> *deviceOut) {
  if (!deviceHandle || !deviceOut)
    return failure(2, "null Metal device handle");
  id object = (__bridge id)deviceHandle;
  if (![object conformsToProtocol:@protocol(MTLDevice)])
    return failure(2, "invalid Metal device handle");
  *deviceOut = (id<MTLDevice>)object;
  return ok();
}

AMStatus am_library(void *deviceHandle, const char *text, int32_t mode,
                    void **out) {
  @autoreleasepool {
    if (!out) return failure(2, "null Metal library output");
    *out = nullptr;
    if (!deviceHandle || !text || (mode != 0 && mode != 1))
      return failure(2, "invalid Metal library device, text, or mode");

    id<MTLDevice> device = nil;
    AMStatus status = checkedMetalDevice(deviceHandle, &device);
    if (status.code) return status;
    NSError *deviceError = nil;
    if (!deviceState(device, &deviceError))
      return failure(1, deviceError.localizedDescription.UTF8String
          ?: "Metal device unavailable");

    NSString *value = [NSString stringWithUTF8String:text];
    if (!value)
      return failure(mode == 0 ? 7 : 9,
                     "Metal library source or path is not valid UTF-8");
    if (mode == 1 && value.length == 0)
      return failure(2, "Metal library path must not be empty");

    NSError *error = nil;
    id<MTLLibrary> library = nil;
    NSString *label = nil;
    if (mode == 0) {
      MTLCompileOptions *options = [MTLCompileOptions new];
      library = [device newLibraryWithSource:value options:options error:&error];
      label = @"runtime source";
      if (!library)
        return failure(7, error.localizedDescription.UTF8String
            ?: "Metal runtime source compilation failed");
    } else {
      NSString *path = [value stringByStandardizingPath];
      NSURL *url = [NSURL fileURLWithPath:path];
      library = [device newLibraryWithURL:url error:&error];
      label = path;
      if (!library)
        return failure(9, error.localizedDescription.UTF8String
            ?: "Metal library file could not be loaded");
    }

    AMLibrary *owner = [AMLibrary new];
    if (!owner) return failure(9, "cannot retain Metal library");
    owner.device = device;
    owner.library = library;
    owner.label = label;
    owner.pipelines = [NSMutableDictionary new];
    *out = (__bridge_retained void *)owner;
    return ok();
  }
}

AMStatus am_kernel(void *libraryHandle, const char *name, void **out) {
  @autoreleasepool {
    if (!out) return failure(2, "null Metal kernel output");
    *out = nullptr;
    if (!libraryHandle || !name)
      return failure(2, "null Metal library or function name");

    id object = (__bridge id)libraryHandle;
    if (![object isKindOfClass:AMLibrary.class])
      return failure(2, "invalid Metal library handle");
    AMLibrary *owner = (AMLibrary *)object;
    NSString *functionName = [NSString stringWithUTF8String:name];
    if (!functionName || functionName.length == 0)
      return failure(2, "Metal function name must not be empty");
    if (!owner.device || !owner.library || !owner.pipelines)
      return failure(9, "Metal library handle is incomplete");

    id<MTLComputePipelineState> pipeline = nil;
    id<MTLFunction> function = nil;
    @synchronized(owner) {
      function = [owner.library newFunctionWithName:functionName];
      if (!function) {
        NSString *message = [NSString stringWithFormat:
            @"Metal function '%@' was not found in library '%@'",
            functionName, owner.label ?: @"<unnamed>"];
        return failure(8, message.UTF8String);
      }
      pipeline = owner.pipelines[functionName];
      if (!pipeline) {
        NSError *error = nil;
        pipeline = [owner.device
            newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline)
          return failure(7, error.localizedDescription.UTF8String
              ?: "Metal compute pipeline compilation failed");
        owner.pipelines[functionName] = pipeline;
      }
    }

    AMKernel *kernel = [AMKernel new];
    if (!kernel) return failure(7, "cannot retain Metal compute pipeline");
    // The library owns cached pipeline states, never AMKernel objects. This
    // keeps the cache acyclic while each retained kernel owns its dependencies.
    kernel.owner = owner;
    kernel.pipeline = pipeline;
    kernel.name = function.name ?: functionName;
    *out = (__bridge_retained void *)kernel;
    return ok();
  }
}

static AMStatus validateCustomBinding(const AMKernelBinding &binding,
                                      id<MTLDevice> device,
                                      AMBuffer **ownerOut,
                                      uint64_t *byteOffsetOut,
                                      uint64_t *byteStartOut,
                                      uint64_t *byteEndOut) {
  if (!binding.buffer || !ownerOut || !byteOffsetOut ||
      !byteStartOut || !byteEndOut)
    return failure(2, "null custom Metal buffer binding");
  if (binding.access < 0 || binding.access > 2 || binding.reserved != 0)
    return failure(2, "invalid custom Metal binding access or reserved field");

  id object = (__bridge id)binding.buffer;
  if (![object isKindOfClass:AMBuffer.class])
    return failure(2, "invalid custom Metal buffer handle");
  AMBuffer *owner = (AMBuffer *)object;
  if (!owner.buffer || !owner.state ||
      owner.buffer.device.registryID != device.registryID ||
      owner.state.device.registryID != device.registryID)
    return failure(2, "custom Metal buffer belongs to a different device");

  uint64_t count = 0;
  AMStatus status = validate(owner, binding.descriptor, count);
  if (status.code) return status;
  const uint64_t width = dtypeWidth(binding.descriptor.dtype);
  if (!width || owner.capacity % width != 0)
    return failure(2, "custom Metal descriptor has an invalid dtype capacity");
  if (count == 0)
    return failure(2, "custom Metal bindings cannot use empty tensors");

  bool cContiguous = true;
  bool fContiguous = true;
  __int128 expected = 1;
  for (int axis = binding.descriptor.rank - 1; axis >= 0; --axis) {
    const int64_t stride = binding.descriptor.strides[axis];
    const int64_t dimension = binding.descriptor.shape[axis];
    if (stride <= 0) cContiguous = false;
    if (dimension > 1 && (__int128)stride != expected)
      cContiguous = false;
    expected *= dimension;
  }
  expected = 1;
  for (int axis = 0; axis < binding.descriptor.rank; ++axis) {
    const int64_t stride = binding.descriptor.strides[axis];
    const int64_t dimension = binding.descriptor.shape[axis];
    if (stride <= 0) fContiguous = false;
    if (dimension > 1 && (__int128)stride != expected)
      fContiguous = false;
    expected *= dimension;
  }
  if (!cContiguous && !fContiguous)
    return failure(2, "custom Metal binding must be positive C- or F-contiguous");

  const __uint128_t byteOffset =
      (__uint128_t)(uint64_t)binding.descriptor.offset * width;
  const __uint128_t byteLength = (__uint128_t)count * width;
  if (byteOffset > UINTPTR_MAX || byteLength > UINT64_MAX ||
      byteOffset > owner.capacity ||
      byteLength > (__uint128_t)owner.capacity - byteOffset ||
      byteOffset + byteLength > owner.buffer.length)
    return failure(2, "custom Metal binding byte range exceeds buffer capacity");

  *ownerOut = owner;
  *byteOffsetOut = (uint64_t)byteOffset;
  *byteStartOut = (uint64_t)byteOffset;
  *byteEndOut = (uint64_t)(byteOffset + byteLength);
  return ok();
}

static AMStatus validateCustomDispatchSize(AMKernel *kernel,
                                           AMGrid grid, AMGrid group,
                                           MTLSize *gridSizeOut,
                                           MTLSize *groupSizeOut) {
  if (!kernel || !kernel.owner.device || !kernel.pipeline ||
      !gridSizeOut || !groupSizeOut)
    return failure(2, "invalid custom Metal dispatch handle");
  if (!grid.x || !grid.y || !grid.z ||
      !group.x || !group.y || !group.z)
    return failure(2, "custom Metal grid and group dimensions must be positive");
  if (grid.x > UINT32_MAX || grid.y > UINT32_MAX ||
      grid.z > UINT32_MAX || group.x > UINT32_MAX ||
      group.y > UINT32_MAX || group.z > UINT32_MAX)
    return failure(2, "custom Metal grid or group exceeds Metal dimension limits");

  const MTLSize deviceLimit = kernel.owner.device.maxThreadsPerThreadgroup;
  if (group.x > deviceLimit.width || group.y > deviceLimit.height ||
      group.z > deviceLimit.depth)
    return failure(2, "custom Metal threadgroup exceeds per-axis device limits");
  const __uint128_t groupProduct =
      (__uint128_t)group.x * group.y * group.z;
  if (groupProduct > kernel.pipeline.maxTotalThreadsPerThreadgroup)
    return failure(2, "custom Metal threadgroup exceeds pipeline thread limit");
  const __uint128_t gridProduct = (__uint128_t)grid.x * grid.y * grid.z;
  if (gridProduct > UINT64_MAX)
    return failure(2, "custom Metal grid product overflows");

  *gridSizeOut = MTLSizeMake((NSUInteger)grid.x, (NSUInteger)grid.y,
                             (NSUInteger)grid.z);
  *groupSizeOut = MTLSizeMake((NSUInteger)group.x, (NSUInteger)group.y,
                              (NSUInteger)group.z);
  return ok();
}

AMStatus am_dispatch(void *kernelHandle, const AMKernelBinding *bindings,
                     uint32_t count, AMGrid grid, AMGrid group,
                     void *streamHandle) {
  @autoreleasepool {
    constexpr uint32_t maximumBindings = 31;
    if (!kernelHandle || (count && !bindings) || count > maximumBindings)
      return failure(2, "invalid custom Metal kernel, bindings, or count");

    id kernelObject = (__bridge id)kernelHandle;
    if (![kernelObject isKindOfClass:AMKernel.class])
      return failure(2, "invalid custom Metal kernel handle");
    AMKernel *kernel = (AMKernel *)kernelObject;
    AMLibrary *library = kernel.owner;
    if (!library || !library.device || !kernel.pipeline || !kernel.name)
      return failure(2, "custom Metal kernel handle is incomplete");

    id<MTLDevice> device = library.device;
    AMDeviceState *state = deviceState(device, nil);
    if (!state) return failure(1, "Metal device accounting state unavailable");
    AMStream *stream = nil;
    if (streamHandle) {
      id streamObject = (__bridge id)streamHandle;
      if (![streamObject isKindOfClass:AMStream.class])
        return failure(2, "invalid explicit Metal stream handle");
      stream = (AMStream *)streamObject;
      if (stream.device.registryID != device.registryID)
        return failure(2, "custom Metal stream belongs to a different device");
    } else {
      AMStatus status = selectedStream(device, state, &stream);
      if (status.code) return status;
    }

    MTLSize gridSize = MTLSizeMake(0, 0, 0);
    MTLSize groupSize = MTLSizeMake(0, 0, 0);
    AMStatus status = validateCustomDispatchSize(kernel, grid, group,
                                                 &gridSize, &groupSize);
    if (status.code) return status;

    AMBuffer *owners[maximumBindings] = {};
    uint64_t offsets[maximumBindings] = {};
    uint64_t starts[maximumBindings] = {};
    uint64_t ends[maximumBindings] = {};
    NSMutableArray<AMCustomLease *> *customLeases =
        [NSMutableArray arrayWithCapacity:count];
    NSMutableArray *retainedHandles =
        [NSMutableArray arrayWithCapacity:(NSUInteger)count + 1];
    NSMutableArray<AMBuffer *> *outputs =
        [NSMutableArray arrayWithCapacity:count];
    [retainedHandles addObject:kernel];

    for (uint32_t index = 0; index < count; ++index) {
      AMBuffer *owner = nil;
      status = validateCustomBinding(bindings[index], device, &owner,
                                     &offsets[index], &starts[index],
                                     &ends[index]);
      if (status.code) return status;
      owners[index] = owner;
      [retainedHandles addObject:owners[index]];

      AMCustomLease *lease = [AMCustomLease new];
      lease.owner = owners[index];
      lease.byteStart = starts[index];
      lease.byteEnd = ends[index];
      lease.streamIdentity = stream.identity;
      lease.access = bindings[index].access;
      [customLeases addObject:lease];

      if (bindings[index].access != 0 &&
          ![outputs containsObject:owners[index]])
        [outputs addObject:owners[index]];
    }

    for (uint32_t right = 0; right < count; ++right) {
      for (uint32_t left = 0; left < right; ++left) {
        if (owners[left].buffer != owners[right].buffer ||
            (bindings[left].access == 0 && bindings[right].access == 0) ||
            starts[left] >= ends[right] || starts[right] >= ends[left])
          continue;
        return failure(10, "custom Metal dispatch has overlapping mutable buffer bindings");
      }
    }

    id<MTLCommandBuffer> command = [stream.queue commandBuffer];
    if (!command) return failure(5, "cannot create custom Metal command");
    command.label = kernel.name;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) return failure(5, "cannot create custom Metal encoder");
    BOOL ended = NO;
    @try {
      [encoder setComputePipelineState:kernel.pipeline];
      for (uint32_t index = 0; index < count; ++index)
        [encoder setBuffer:owners[index].buffer
                    offset:(NSUInteger)offsets[index]
                   atIndex:index];
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
      [encoder endEncoding];
      ended = YES;
    } @catch (NSException *exception) {
      if (!ended) {
        @try { [encoder endEncoding]; }
        @catch (NSException *endException) { (void)endException; }
      }
      return failure(2, exception.reason.UTF8String
          ?: "custom Metal encoder rejected the dispatch");
    }

    const uint64_t invocation = nextInvocation.fetch_add(1);
    AMEvidence evidence = {invocation, device.registryID, 0, 0, 0, 0, 2, 300};
    status = submitCommandDetailed(stream, command, retainedHandles, outputs,
                                   customLeases, invocation, evidence, 0, nullptr);
    return status;
  }
}
