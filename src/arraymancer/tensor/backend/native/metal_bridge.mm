// Copyright 2026 Arraymancer contributors. Apache-2.0.
// Provides native shared storage; producers belong to storage, not metadata aliases.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#include "apple_abi.h"
#include <atomic>
#include <cmath>
#include <stdlib.h>
#include <string.h>

@class AMDeviceState;
@class AMCapture;
@class AMGenerator;

@interface AMBuffer : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, strong) id<MTLCommandBuffer> producer;
@property(nonatomic) uint64_t producerInvocation;
@property(nonatomic, copy) NSString *producerError;
@property(nonatomic) AMEvidence evidence;
@property(nonatomic) uint64_t capacity;
@property(nonatomic, weak) AMDeviceState *state;
@property(nonatomic) uint64_t externalReferences;
@property(nonatomic) uint64_t inFlightLeases;
@property(nonatomic) BOOL cached;
@end
@implementation AMBuffer
@end

@interface AMGenerator : NSObject {
 @public
  AMRngState rng;
}
@property(nonatomic, strong) AMDeviceState *deviceState;
@end
@implementation AMGenerator
@end

@interface AMPendingCommand : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> command;
@property(nonatomic, copy) NSArray *retainedHandles;
@property(nonatomic, copy) NSArray<AMBuffer *> *bufferLeases;
@property(nonatomic, strong) AMBuffer *output;
@property(nonatomic) uint64_t sequence;
@property(nonatomic) uint64_t invocation;
@property(nonatomic) uint64_t materializationBytes;
@property(nonatomic) BOOL finalized;
@end
@implementation AMPendingCommand
@end

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
static AMCapture *activeCapture;

@interface AMStream : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic) uint64_t identity;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, AMPendingCommand *> *pending;
@property(nonatomic) uint64_t acceptedSequence;
@property(nonatomic) uint64_t completedSequence;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSString *> *failures;
@property(nonatomic) uint64_t synchronizedSequence;
@property(nonatomic) BOOL lastAcceptedComplete;
@property(nonatomic) BOOL lastAcceptedFailed;
@property(nonatomic, copy) NSString *lastAcceptedError;
@end
@implementation AMStream
@end

@interface AMEvent : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLSharedEvent> shared;
@property(nonatomic) uint64_t lastScheduledValue;
@property(nonatomic) uint64_t lastRecordedValue;
@end
@implementation AMEvent
@end

@interface AMDeviceState : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) AMGenerator *defaultGenerator;
@property(nonatomic, strong) AMStream *defaultStream;
@property(nonatomic, strong) NSHashTable<AMStream *> *streams;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSMutableArray<AMBuffer *> *> *bufferCache;
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

@interface AMCompute : NSObject
@property(nonatomic, strong) AMDeviceState *state;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@end
@implementation AMCompute
@end
static AMStatus failure(int32_t code, const char *text) {
  return {code, strdup(text ? text : "native Metal operation failed")};
}
static AMStatus ok();
static std::atomic<uint64_t> nextInvocation{1};
static AMStatus ok() { return {0, nullptr}; }
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
static BOOL removeCachedOwnerLocked(AMDeviceState *state, AMBuffer *owner) {
  if (!owner.cached) return NO;
  NSNumber *key = @(owner.buffer.length);
  NSMutableArray<AMBuffer *> *bin = state.bufferCache[key];
  NSUInteger index = [bin indexOfObjectIdenticalTo:owner];
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

static BOOL evictOneCachedOwnerLocked(AMDeviceState *state) {
  for (NSNumber *key in state.bufferCache.allKeys) {
    NSMutableArray<AMBuffer *> *bin = state.bufferCache[key];
    if (!bin.count) continue;
    AMBuffer *owner = bin.lastObject;
    const uint64_t length = owner.buffer.length;
    [bin removeLastObject];
    if (!bin.count) [state.bufferCache removeObjectForKey:key];
    owner.cached = NO;
    owner.buffer = nil;
    state.cachedBytes -= length;
    state.reservedBytes -= length;
    return YES;
  }
  return NO;
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
  return nil;
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
  }
  CFRelease((__bridge CFTypeRef)object);
}
static NSString *const currentStreamKey = @"org.arraymancer.metal.currentStream";

static AMStream *newStream(id<MTLDevice> device, id<MTLCommandQueue> queue) {
  if (!device || !queue) return nil;
  AMStream *stream = [AMStream new];
  stream.device = device;
  stream.queue = queue;
  stream.identity = (uint64_t)(uintptr_t)(__bridge void *)queue;
  stream.pending = [NSMutableDictionary new];
  stream.failures = [NSMutableDictionary new];
  stream.lastAcceptedComplete = YES;
  return stream;
}

static AMDeviceState *deviceState(id<MTLDevice> device, NSError **error) {
  if (!device) {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    device = devices.firstObject;
  }
  if (!device) return nil;

  static NSMutableDictionary<NSNumber *, AMDeviceState *> *states;
  static dispatch_once_t once;
  dispatch_once(&once, ^{ states = [NSMutableDictionary new]; });
  @synchronized(states) {
    NSNumber *key = @(device.registryID);
    AMDeviceState *state = states[key];
    if (state) return state;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      if (error) *error = [NSError errorWithDomain:@"Arraymancer.Metal"
          code:1 userInfo:@{NSLocalizedDescriptionKey: @"cannot create default Metal command queue"}];
      return nil;
    }
    state = [AMDeviceState new];
    state.device = device;
    state.streams = [NSHashTable weakObjectsHashTable];
    state.bufferCache = [NSMutableDictionary new];
    state.memoryFraction = 1.0;
    state.defaultStream = newStream(device, queue);
    [state.streams addObject:state.defaultStream];
    states[key] = state;
    return state;
  }
}

static AMStatus selectedStream(id<MTLDevice> device, AMDeviceState *state,
                               AMStream **result) {
  if (!device || !state || !result) return failure(2, "invalid Metal stream request");
  AMStream *selected = NSThread.currentThread.threadDictionary[currentStreamKey];
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

static void finishStreamCommand(AMStream *stream, AMPendingCommand *record) {
  id<MTLCommandBuffer> command = record.command;
  const uint64_t sequence = record.sequence;
  NSString *message = command.status == MTLCommandBufferStatusError
      ? (command.error.localizedDescription ?: @"Metal command failed") : nil;
  @synchronized(stream) {
    if (sequence > stream.completedSequence) stream.completedSequence = sequence;
    [stream.pending removeObjectForKey:@(sequence)];
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
                                 id<MTLCommandBuffer> command) {
  NSString *message = command.status == MTLCommandBufferStatusError
      ? (command.error.localizedDescription ?: @"Metal command failed") : nil;
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

static void finalizeCommand(AMStream *stream, AMPendingCommand *record) {
  @synchronized(record) {
    id<MTLCommandBuffer> command = record.command;
    if (record.finalized ||
        (command.status != MTLCommandBufferStatusCompleted &&
         command.status != MTLCommandBufferStatusError)) return;
    record.finalized = YES;
    if (record.output)
      finishBufferProducer(record.output, record.invocation, command);
    if (command.status == MTLCommandBufferStatusCompleted &&
        record.output && record.materializationBytes) {
      @synchronized(record.output) {
        if (record.output.evidence.invocation_id == record.invocation) {
          AMEvidence evidence = record.output.evidence;
          evidence.materialization_bytes = record.materializationBytes;
          record.output.evidence = evidence;
        }
      }
    }

    AMDeviceState *state = deviceState(stream.device, nil);
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
  finishStreamCommand(stream, record);
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
  return [owners copy];
}

static AMStatus submitCommand(AMStream *stream, id<MTLCommandBuffer> command,
                              NSArray *leases, AMBuffer *output,
                              uint64_t invocation, AMEvidence evidence,
                              uint64_t materializationBytes,
                              uint64_t *submittedSequence) {
  if (!stream || !command) return failure(5, "cannot submit Metal command");
  NSArray *retainedLeases = [leases copy] ?: @[];
  NSArray<AMBuffer *> *bufferLeases = uniqueBufferOwners(retainedLeases);
  AMDeviceState *state = deviceState(stream.device, nil);
  if (!state) return failure(1, "Metal device accounting state unavailable");
  for (AMBuffer *owner in bufferLeases)
    if (owner.state != state)
      return failure(2, "Metal command buffers belong to different devices");

  AMPendingCommand *record = [AMPendingCommand new];
  record.command = command;
  record.retainedHandles = retainedLeases;
  record.bufferLeases = bufferLeases;
  record.output = output;
  record.invocation = invocation;
  record.materializationBytes = materializationBytes;

  @synchronized(stream) {
    if (stream.acceptedSequence == UINT64_MAX)
      return failure(5, "Metal stream command sequence exhausted");
    @synchronized(state) {
      for (AMBuffer *owner in bufferLeases)
        if (!owner.externalReferences || owner.cached ||
            owner.inFlightLeases == UINT64_MAX)
          return failure(2, "Metal command cannot lease an unowned buffer");
      for (AMBuffer *owner in bufferLeases) {
        if (!owner.inFlightLeases)
          state.inFlightBytes = saturatedAdd(state.inFlightBytes, owner.capacity);
        ++owner.inFlightLeases;
      }
    }
    const uint64_t sequence = ++stream.acceptedSequence;
    record.sequence = sequence;
    if (output) prepareBufferProducer(output, command, invocation, evidence);
    stream.lastAcceptedComplete = NO;
    stream.lastAcceptedFailed = NO;
    stream.lastAcceptedError = nil;
    stream.pending[@(sequence)] = record;
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      @autoreleasepool {
        (void)completed;
        finalizeCommand(stream, record);
      }
    }];
    if (submittedSequence) *submittedSequence = sequence;
    [command commit];
  }
  return ok();
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
    for (AMPendingCommand *record in stream.pending.allValues)
      if (record.sequence <= target) [snapshot addObject:record];
  }
  [snapshot sortUsingComparator:^NSComparisonResult(AMPendingCommand *left,
                                                      AMPendingCommand *right) {
    if (left.sequence < right.sequence) return NSOrderedAscending;
    if (left.sequence > right.sequence) return NSOrderedDescending;
    return NSOrderedSame;
  }];
  return snapshot;
}

static void waitAndFinalize(AMStream *stream,
                            NSArray<AMPendingCommand *> *snapshot) {
  for (AMPendingCommand *record in snapshot) {
    [record.command waitUntilCompleted];
    finalizeCommand(stream, record);
  }
}

static AMStatus waitForProducer(AMBuffer *owner) {
  if (!owner) return failure(2, "null Metal buffer owner");
  for (;;) {
    id<MTLCommandBuffer> producer = nil;
    uint64_t invocation = 0;
    NSString *savedError = nil;
    @synchronized(owner) {
      producer = owner.producer;
      invocation = owner.producerInvocation;
      savedError = owner.producerError;
    }
    if (!producer)
      return savedError ? failure(5, savedError.UTF8String) : ok();

    [producer waitUntilCompleted];
    finishBufferProducer(owner, invocation, producer);
    NSString *message = nil;
    BOOL stillCurrent = NO;
    @synchronized(owner) {
      stillCurrent = owner.producerInvocation == invocation;
      if (stillCurrent) message = owner.producerError;
    }
    if (!stillCurrent) continue;
    return message ? failure(5, message.UTF8String) : ok();
  }
}

AMStatus am_buffer_evidence(void *buffer, AMEvidence *evidence) {
  if (!buffer || !evidence) return failure(2, "null evidence handle");
  AMBuffer *owner = (__bridge AMBuffer *)buffer;
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
      owner.producer = nil;
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
      stream = newStream(device, queue);
      @synchronized(state) { [state.streams addObject:stream]; }
    }
    *identity = stream.identity;
    *out = (__bridge_retained void *)stream;
    return ok();
  }
}

AMStatus am_stream_set(void *streamHandle) {
  @autoreleasepool {
    NSMutableDictionary *dictionary = NSThread.currentThread.threadDictionary;
    if (streamHandle)
      dictionary[currentStreamKey] = (__bridge AMStream *)streamHandle;
    else
      [dictionary removeObjectForKey:currentStreamKey];
  }
  return ok();
}

AMStatus am_stream_sync(void *streamHandle) {
  @autoreleasepool {
    if (!streamHandle) return failure(2, "null Metal stream");
    AMStream *stream = (__bridge AMStream *)streamHandle;
    uint64_t sequence = 0;
    AMStatus status = enqueueMarker(stream, nil, &sequence);
    if (status.code) return status;
    // Waiting only the marker's status is insufficient: publish every prior
    // command's native lease/accounting completion before returning.
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
      AMStatus status = enqueueMarker(stream, nil, &sequence);
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
  AMPendingCommand *latest = nil;
  @synchronized(stream) {
    if (!stream.lastAcceptedComplete && stream.acceptedSequence)
      latest = stream.pending[@(stream.acceptedSequence)];
  }
  if (latest &&
      (latest.command.status == MTLCommandBufferStatusCompleted ||
       latest.command.status == MTLCommandBufferStatusError))
    finalizeCommand(stream, latest);
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
  for (AMBuffer *owner in [bin copy])
    if (owner.capacity == bytes && removeCachedOwnerLocked(state, owner))
      return owner;
  return nil;
}

AMStatus am_buffer(void *deviceHandle, uint64_t bytes, void **buffer) {
  @autoreleasepool {
    if (!deviceHandle || !buffer) return failure(2, "null allocation handle");
    *buffer = nullptr;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)deviceHandle;
    const uint64_t requestedLength = MAX(bytes, 1);
    if (bytes > dev.maxBufferLength || requestedLength > dev.maxBufferLength)
      return failure(4, "Metal allocation exceeds maxBufferLength");

    NSError *error = nil;
    AMDeviceState *state = deviceState(dev, &error);
    if (!state)
      return failure(1, error.localizedDescription.UTF8String ?: "Metal device unavailable");
    AMBuffer *owner = nil;
    @synchronized(state) {
      const uint64_t limit = memoryLimitBytes(state);
      const uint64_t liveReserved = state.reservedBytes - state.cachedBytes;
      // A cache hit is still subject to the requested limit. In particular,
      // a fraction smaller than this allocation cannot be bypassed by reuse.
      if (requestedLength > limit ||
          wouldExceed(liveReserved, requestedLength, limit))
        return failure(4, "Metal allocation exceeds the configured memory fraction");

      owner = cachedOwnerForRequestLocked(state, bytes, requestedLength);
      if (owner) {
        while (state.reservedBytes > limit)
          if (!evictOneCachedOwnerLocked(state))
            return failure(4, "Metal reserved memory exceeds the configured limit");
      } else {
        while (wouldExceed(state.reservedBytes, requestedLength, limit))
          if (!evictOneCachedOwnerLocked(state))
            return failure(4, "Metal allocation exceeds the configured memory fraction");
        id<MTLBuffer> native =
            [dev newBufferWithLength:requestedLength options:MTLResourceStorageModeShared];
        if (!native) return failure(4, "Metal allocation failed");
        const uint64_t actualLength = native.length;
        while (wouldExceed(state.reservedBytes, actualLength, limit))
          if (!evictOneCachedOwnerLocked(state)) {
            native = nil;
            return failure(4, "Metal allocation exceeds the configured memory fraction");
          }
        owner = [AMBuffer new];
        owner.buffer = native;
        state.reservedBytes = saturatedAdd(state.reservedBytes, actualLength);
      }

      owner.state = state;
      owner.capacity = bytes;
      owner.externalReferences = 1;
      owner.inFlightLeases = 0;
      owner.cached = NO;
      owner.producer = nil;
      owner.producerInvocation = 0;
      owner.producerError = nil;
      owner.evidence = {nextInvocation.fetch_add(1), dev.registryID, 0, 0, 0, 1, 1};
      state.liveBytes = saturatedAdd(state.liveBytes, bytes);
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

static AMCompute *compute(id<MTLDevice> device, const char *directory, NSError **error) {
  static NSMutableDictionary<NSNumber *, AMCompute *> *contexts = [NSMutableDictionary new];
  if (!device || !directory) return nil;
  @synchronized(contexts) {
    AMCompute *context = contexts[@(device.registryID)];
    if (!context) {
      context = [AMCompute new];
      context.state = deviceState(device, error);
      NSURL *url = [NSURL fileURLWithPath:
        [[NSString stringWithUTF8String:directory] stringByAppendingPathComponent:@"arraymancer.metallib"]];
      if (!context.state) return nil;
      context.library = [device newLibraryWithURL:url error:error];
      if (!context.library) return nil;
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
      pipeline = [context.state.device newComputePipelineStateWithFunction:function error:error];
      if (pipeline) context.pipelines[name] = pipeline;
    }
    return pipeline;
  }
}

static AMStatus runMathCommand(AMCompute *context,
                               id<MTLComputePipelineState> pipeline,
                               NSString *name, AMBuffer *a, AMBuffer *b,
                               AMBuffer *output, AMTensorDescriptor ad,
                               AMTensorDescriptor bd, AMTensorDescriptor od,
                               AMMathParameters parameters,
                               uint64_t outputCount, MTLSize groups,
                               MTLSize threads, int32_t operation) {
  AMStream *stream = nil;
  AMStatus status = selectedStream(output.buffer.device, context.state, &stream);
  if (status.code) return status;
  id<MTLCommandBuffer> command = [stream.queue commandBuffer];
  if (!command) return failure(5, "cannot create Metal math command");
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (!encoder) return failure(5, "cannot create Metal math encoder");
  command.label = name;
  [encoder setComputePipelineState:pipeline];
  switch (operation) {
    case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: case 9: case 23:
      [encoder setBuffer:(a ? a.buffer : output.buffer) offset:0 atIndex:0];
      [encoder setBuffer:(b ? b.buffer : output.buffer) offset:0 atIndex:1];
      [encoder setBuffer:output.buffer offset:0 atIndex:2];
      [encoder setBytes:&ad length:sizeof(ad) atIndex:3];
      [encoder setBytes:&bd length:sizeof(bd) atIndex:4];
      [encoder setBytes:&od length:sizeof(od) atIndex:5];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:6];
      break;
    case 20:
      [encoder setBuffer:a.buffer offset:0 atIndex:0];
      [encoder setBuffer:b.buffer offset:0 atIndex:1];
      [encoder setBuffer:output.buffer offset:0 atIndex:2];
      [encoder setBytes:&ad length:sizeof(ad) atIndex:3];
      [encoder setBytes:&bd length:sizeof(bd) atIndex:4];
      [encoder setBytes:&od length:sizeof(od) atIndex:5];
      break;
    case 21:
      [encoder setBuffer:a.buffer offset:0 atIndex:0];
      [encoder setBuffer:output.buffer offset:0 atIndex:1];
      [encoder setBytes:&ad length:sizeof(ad) atIndex:2];
      [encoder setBytes:&od length:sizeof(od) atIndex:3];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
      break;
    case 22:
      [encoder setBuffer:a.buffer offset:0 atIndex:0];
      [encoder setBuffer:output.buffer offset:0 atIndex:1];
      [encoder setBytes:&ad length:sizeof(ad) atIndex:2];
      [encoder setBytes:&od length:sizeof(od) atIndex:3];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
      break;
    default:
      [encoder endEncoding];
      return failure(2, "unsupported Metal math operation");
  }
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  [encoder endEncoding];

  const uint64_t invocation = nextInvocation.fetch_add(1);
  AMEvidence evidence = {invocation, output.buffer.device.registryID,
                         0, 0, 0, 0, 2, 100 + operation};
  NSMutableArray *leases = [NSMutableArray new];
  if (a) [leases addObject:a];
  if (b && b != a) [leases addObject:b];
  if (output != a && output != b) [leases addObject:output];
  [leases addObject:pipeline];
  status = submitCommand(stream, command, leases, output, invocation, evidence, 0, nullptr);
  if (status.code) abandonBufferProducer(output, invocation);
  (void)outputCount;
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
    const bool elementwise = (operation >= 1 && operation <= 9) || operation == 23;
    if (!elementwise && operation != 20 && operation != 21 && operation != 22)
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

    if (elementwise) {
      if (operation == 23) {
        if (!b || parameters.scalar_input || !sameShape(ad, bd) || !sameShape(ad, od))
          return failure(2, "squared difference requires matching tensor shapes");
      } else if (operation == 5) {
        if (b || parameters.scalar_input || !sameShape(ad, od))
          return failure(2, "negation requires matching input and output shapes");
      } else if (operation == 6) {
        if (a || b) return failure(2, "fill does not accept input buffers");
      } else if (operation == 8 || operation == 9) {
        if (b || !parameters.scalar_input || !sameShape(ad, od))
          return failure(2, "scalar-left operation requires matching tensor shape");
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
      threads = MTLSizeMake(MIN((NSUInteger)256, (NSUInteger)outputCount), 1, 1);
      groups = MTLSizeMake((outputCount + threads.width - 1) / threads.width, 1, 1);
    } else if (operation == 20) {
      if (parameters.scalar_input || !b || ad.rank != 2 || bd.rank != 2 || od.rank != 2)
        return failure(2, "matmul requires rank-two tensors");
      if (ad.shape[1] != bd.shape[0] || od.shape[0] != ad.shape[0] ||
          od.shape[1] != bd.shape[1])
        return failure(2, "matmul dimensions are incompatible");
      if (outputCount > UINT32_MAX)
        return failure(2, "matmul output exceeds Metal grid limits");
      if (!outputCount) return ok();
      kernelName = @"math_matmul";
      threads = MTLSizeMake(16, 16, 1);
      groups = MTLSizeMake((uint64_t(bd.shape[1]) + 15) / 16,
                           (uint64_t(ad.shape[0]) + 15) / 16, 1);
      if (groups.width > UINT32_MAX || groups.height > UINT32_MAX)
        return failure(2, "matmul grid exceeds Metal limits");
    } else if (operation == 21) {
      if (b)
        return failure(2, "full sum accepts one tensor input");
      const uint64_t partialCount = MAX((uint64_t)1, (aCount + 255) / 256);
      if (outputCount != partialCount || partialCount > UINT32_MAX ||
          (parameters.scalar_input && partialCount != 1))
        return failure(2, "full-sum output must contain one value per reduction group");
      kernelName = @"math_reduce";
      threads = MTLSizeMake(256, 1, 1);
      groups = MTLSizeMake(partialCount, 1, 1);
      dispatchCount = partialCount;
    } else {
      if (b || parameters.scalar_input || parameters.axis < 0 ||
          parameters.axis >= ad.rank || od.rank != ad.rank)
        return failure(2, "invalid axis-sum arguments");
      for (int axis = 0; axis < ad.rank; ++axis) {
        const int64_t expected = axis == parameters.axis ? 1 : ad.shape[axis];
        if (od.shape[axis] != expected)
          return failure(2, "axis-sum output must keep the reduced dimension");
      }
      if (outputCount > UINT32_MAX)
        return failure(2, "axis-sum output exceeds Metal grid limits");
      if (!outputCount) return ok();
      kernelName = @"math_axis_sum";
      threads = MTLSizeMake(256, 1, 1);
      groups = MTLSizeMake(outputCount, 1, 1);
    }

    NSError *error = nil;
    AMCompute *context = compute(output.buffer.device, directory, &error);
    if (!context)
      return failure(9, error ? error.localizedDescription.UTF8String : "Metal queue or library unavailable");
    id<MTLComputePipelineState> pipeline = pipelineFor(context, kernelName, &error);
    if (!pipeline)
      return failure(error ? 7 : 8,
                     error ? error.localizedDescription.UTF8String : "missing Metal math kernel");
    if (operation >= 20 &&
        pipeline.maxTotalThreadsPerThreadgroup < threads.width * threads.height * threads.depth)
      return failure(7, "Metal math pipeline cannot run its required threadgroup");
    if (elementwise) {
      const NSUInteger supported = pipeline.maxTotalThreadsPerThreadgroup;
      const NSUInteger width = MIN((NSUInteger)outputCount, MIN((NSUInteger)256, supported));
      if (!width) return failure(7, "Metal elementwise pipeline has no supported threads");
      threads = MTLSizeMake(width, 1, 1);
      groups = MTLSizeMake((outputCount + width - 1) / width, 1, 1);
      if (groups.width > UINT32_MAX || groups.width * width > UINT32_MAX)
        return failure(2, "elementwise grid exceeds Metal index limits");
    }

    return runMathCommand(context, pipeline, kernelName, a, b, output,
                          ad, bd, od, parameters, dispatchCount,
                          groups, threads, operation);
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
    if (!sameShape(pd, td) || !sameShape(pd, od) ||
        predictionCount != targetCount || predictionCount != outputCount)
      return failure(2, "Metal MSE prediction, target, and output shapes must match");
    if (!sameDevice(upstream, output) || !sameDevice(prediction, output) ||
        !sameDevice(target, output))
      return failure(2, "Metal MSE buffers use different devices");
    if (!predictionCount) return ok();
    if (predictionCount > UINT32_MAX)
      return failure(2, "Metal MSE backward output exceeds grid limits");

    NSError *error = nil;
    AMCompute *context = compute(output.buffer.device, directory, &error);
    if (!context)
      return failure(9, error ? error.localizedDescription.UTF8String : "Metal queue or library unavailable");
    NSString *name = @"math_mse_backward";
    id<MTLComputePipelineState> pipeline = pipelineFor(context, name, &error);
    if (!pipeline)
      return failure(error ? 7 : 8,
                     error ? error.localizedDescription.UTF8String : "missing Metal MSE backward kernel");
    const NSUInteger width = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    if (!width) return failure(7, "Metal MSE backward pipeline has no supported threads");
    const NSUInteger groupCount = (predictionCount + width - 1) / width;
    if (groupCount > UINT32_MAX || groupCount * width > UINT32_MAX)
      return failure(2, "Metal MSE backward grid exceeds index limits");

    AMStream *stream = nil;
    status = selectedStream(output.buffer.device, context.state, &stream);
    if (status.code) return status;
    id<MTLCommandBuffer> command = [stream.queue commandBuffer];
    if (!command) return failure(5, "cannot create Metal MSE backward command");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) return failure(5, "cannot create Metal MSE backward encoder");
    command.label = name;
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
    [encoder endEncoding];

    const uint64_t invocation = nextInvocation.fetch_add(1);
    AMEvidence evidence = {invocation, output.buffer.device.registryID,
                           0, 0, 0, 0, 2, 124};
    NSMutableArray *leases = [NSMutableArray new];
    [leases addObject:upstream];
    [leases addObject:prediction];
    if (target != prediction) [leases addObject:target];
    if (output != upstream && output != prediction && output != target)
      [leases addObject:output];
    [leases addObject:pipeline];
    status = submitCommand(stream, command, leases, output, invocation, evidence, 0, nullptr);
    if (status.code) abandonBufferProducer(output, invocation);
    return status;
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
    auto *native = (char *)owner.buffer.contents;
    auto *cpu = (char *)host;
    for (;;) {
      status = waitForProducer(owner);
      if (status.code) return status;
      bool retry = false;
      @synchronized(owner) {
        if (owner.producer) {
          retry = true;
        } else {
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
      }
      if (!retry) {
        AMDeviceState *state = owner.state;
        @synchronized(state) {
          state.hostCopyBytes = saturatedAdd(state.hostCopyBytes, transferBytes);
        }
        return ok();
      }
    }
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
    id<MTLCommandBuffer> command = [stream.queue commandBuffer];
    if (!command) return failure(5, "cannot create Metal copy command");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) return failure(5, "cannot create Metal copy encoder");
    command.label = name;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src.buffer offset:0 atIndex:0];
    [encoder setBuffer:dst.buffer offset:0 atIndex:1];
    [encoder setBytes:&input length:sizeof(input) atIndex:2];
    [encoder setBytes:&output length:sizeof(output) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(count, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding];
    const uint64_t invocation = nextInvocation.fetch_add(1);
    AMEvidence evidence = {invocation, dst.buffer.device.registryID, 0,
      0, 0, 0, 2, (int32_t)(dtypeWidth(input.dtype) * 8)};
    status = submitCommand(stream, command, @[src, dst, pipeline], dst,
                           invocation, evidence, count * dtypeWidth(input.dtype), nullptr);
    if (status.code) abandonBufferProducer(dst, invocation);
    return status;
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
