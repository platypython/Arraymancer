/* Copyright 2026 Arraymancer contributors. Apache-2.0.
 * Provides the retained native Apple storage boundary. */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { int32_t code; char *message; } AMStatus;
typedef struct {
  int32_t rank;
  int32_t dtype;
  int64_t shape[8];
  int64_t strides[8];
  int64_t offset;
  uint64_t capacity;
} AMTensorDescriptor;
typedef struct {
  uint32_t version, reserved;
  uint64_t seed;
  uint32_t counter[4];
} AMRngState;
#ifdef __cplusplus
static_assert(sizeof(AMRngState) == 32, "AMRngState ABI must be 32 bytes");
#endif
typedef struct {
  uint64_t invocation_id, registry_id, transfer_bytes, materialization_bytes;
  uint64_t cpu_fallback_count;
  int32_t completed, kind, kernel_bits;
} AMEvidence;
AMStatus am_buffer_evidence(void *buffer, AMEvidence *evidence);
AMStatus am_copy(void *source, AMTensorDescriptor input, void *destination,
                 AMTensorDescriptor output, const char *library_dir);
AMStatus am_set_element(void *destination, AMTensorDescriptor output,
                        const void *value, const char *library_dir);
typedef struct {
  int32_t operation, scalar_input, axis, reserved;
  float scalar;
} AMMathParameters;
typedef struct {
  uint64_t live_bytes;
  uint64_t cached_bytes;
  uint64_t in_flight_bytes;
  uint64_t driver_bytes;
  uint64_t recommended_bytes;
  uint64_t reserved_bytes;
  uint64_t host_copy_bytes;
  uint64_t materialization_bytes;
} AMMemoryStats;
AMStatus am_memory_stats(void *device, AMMemoryStats *out);
AMStatus am_memory_empty_cache(void *device);
AMStatus am_memory_fraction(void *device, double *value, int32_t set);
AMStatus am_capture_start(void *device, const char *path, void **out);
AMStatus am_capture_stop(void *capture);
AMStatus am_math(void *a, AMTensorDescriptor ad, void *b, AMTensorDescriptor bd,
                 void *output, AMTensorDescriptor od, AMMathParameters parameters,
                 const char *library_dir);
AMStatus am_mse_backward(void *upstream, AMTensorDescriptor upstream_descriptor,
                         void *prediction, AMTensorDescriptor prediction_descriptor,
                         void *target, AMTensorDescriptor target_descriptor,
                         void *output, AMTensorDescriptor output_descriptor,
                         const char *library_dir);
void am_message_free(char *message);
void am_retain(void *handle);
void am_release(void *handle);
AMStatus am_device_count(int32_t *count);
AMStatus am_device(int32_t index, void **device, uint64_t *identity);
/* A returned stream/event is a retained native handle; use am_release. */
AMStatus am_stream(void *device, int32_t mode, void **out, uint64_t *identity);
AMStatus am_stream_set(void *stream);
AMStatus am_stream_sync(void *stream);
AMStatus am_device_sync(void *device);
AMStatus am_stream_complete(void *stream, int32_t *complete);
AMStatus am_event(void *device, void **out);
AMStatus am_event_record(void *event, void *stream, uint64_t *recorded_value);
AMStatus am_event_wait(void *stream, void *event, uint64_t value,
                       int32_t explicit_value);
AMStatus am_event_signal(void *event, uint64_t value);
AMStatus am_buffer(void *device, uint64_t bytes, void **buffer);
AMStatus am_generator(void *device, uint64_t seed, int32_t mode, void **out);
AMStatus am_rng_state(void *generator, AMRngState *state, int32_t mode);
AMStatus am_rng_seed(void *generator, uint64_t *seed, int32_t entropy);
AMStatus am_rand(void *generator, void *output, AMTensorDescriptor descriptor,
                 const char *library_dir);
AMStatus am_transfer(void *buffer, AMTensorDescriptor descriptor, void *host, int32_t upload);
#ifdef __cplusplus
}
#endif
