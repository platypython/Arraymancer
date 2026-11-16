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
AMStatus am_buffer(void *device, uint64_t bytes, void **buffer);
AMStatus am_transfer(void *buffer, AMTensorDescriptor descriptor, void *host, int32_t upload);
#ifdef __cplusplus
}
#endif
