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
