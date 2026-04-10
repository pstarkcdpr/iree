// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/debugging.h"

#include "iree/io/stdio_stream.h"
#include "iree/tooling/numpy_io.h"

//===----------------------------------------------------------------------===//
// Debug Sink
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_null(void) {
  iree_hal_module_debug_sink_t sink = {0};
  return sink;
}

#if IREE_FILE_IO_ENABLE

#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
static iree_status_t iree_hal_module_buffer_view_trace_stdio(
    void* user_data, iree_string_view_t key, iree_host_size_t buffer_view_count,
    iree_hal_buffer_view_t** buffer_views, iree_allocator_t host_allocator) {
  FILE* file = (FILE*)user_data;

  fprintf(file, "=== %.*s ===\n", (int)key.size, key.data);
  for (iree_host_size_t i = 0; i < buffer_view_count; ++i) {
    iree_hal_buffer_view_t* buffer_view = buffer_views[i];

    // NOTE: this export is for debugging only and a no-op in min-size builds.
    // We heap-alloc here because at the point this export is used performance
    // is not a concern.

    // Query total length (excluding NUL terminator).
    iree_host_size_t result_length = 0;
    iree_status_t status = iree_hal_buffer_view_format(
        buffer_view, IREE_HOST_SIZE_MAX, 0, NULL, &result_length);
    if (!iree_status_is_out_of_range(status)) {
      return status;
    }
    ++result_length;  // include NUL

    // Allocate scratch heap memory to contain the result and format into it.
    char* result_str = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, result_length,
                                               (void**)&result_str));
    status =
        iree_hal_buffer_view_format(buffer_view, IREE_HOST_SIZE_MAX,
                                    result_length, result_str, &result_length);
    if (iree_status_is_ok(status)) {
      fprintf(file, "%.*s\n", (int)result_length, result_str);
    }
    iree_allocator_free(host_allocator, result_str);
    IREE_RETURN_IF_ERROR(status);
  }
  fprintf(file, "\n");
  return iree_ok_status();
}
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE

//===----------------------------------------------------------------------===//
// NumPy .npy file sink
//===----------------------------------------------------------------------===//

typedef struct iree_hal_module_debug_sink_npy_state_t {
  iree_allocator_t host_allocator;
  char* trace_dir;
  iree_host_size_t call_index;
} iree_hal_module_debug_sink_npy_state_t;

static void iree_hal_module_debug_sink_npy_release(void* user_data) {
  iree_hal_module_debug_sink_npy_state_t* state =
      (iree_hal_module_debug_sink_npy_state_t*)user_data;
  iree_allocator_free(state->host_allocator, state->trace_dir);
  iree_allocator_free(state->host_allocator, state);
}

static iree_status_t iree_hal_module_buffer_view_trace_npy(
    void* user_data, iree_string_view_t key, iree_host_size_t buffer_view_count,
    iree_hal_buffer_view_t** buffer_views, iree_allocator_t host_allocator) {
  iree_hal_module_debug_sink_npy_state_t* state =
      (iree_hal_module_debug_sink_npy_state_t*)user_data;

  // Sanitize the key for use in a filename: replace characters that are not
  // alphanumeric, '_', or '-' with '_'.
  char sanitized_key[128];
  iree_host_size_t sanitized_len = 0;
  iree_host_size_t key_copy_len =
      key.size < sizeof(sanitized_key) - 1 ? key.size : sizeof(sanitized_key) - 1;
  for (iree_host_size_t j = 0; j < key_copy_len; ++j) {
    char c = key.data[j];
    sanitized_key[sanitized_len++] =
        ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '-')
            ? c
            : '_';
  }
  sanitized_key[sanitized_len] = '\0';

  for (iree_host_size_t i = 0; i < buffer_view_count; ++i) {
    // Build the output path: <trace_dir>/<call_index>_<key>_<i>.npy
    char file_path[IREE_MAX_PATH + 1];
    int path_len =
        snprintf(file_path, sizeof(file_path), "%s/%" PRIhsz "_%s_%" PRIhsz ".npy",
                 state->trace_dir, state->call_index, sanitized_key, i);
    if (path_len <= 0 || path_len > IREE_MAX_PATH) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "trace npy file path too long");
    }

    iree_io_stream_t* stream = NULL;
    IREE_RETURN_IF_ERROR(iree_io_stdio_stream_open(
        IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
        iree_make_cstring_view(file_path), host_allocator, &stream));

    iree_status_t status = iree_numpy_npy_save_ndarray(
        stream, IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT, buffer_views[i],
        host_allocator);
    iree_io_stream_release(stream);
    IREE_RETURN_IF_ERROR(status);
  }

  ++state->call_index;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_module_debug_sink_npy(iree_string_view_t trace_dir,
                                iree_allocator_t host_allocator,
                                iree_hal_module_debug_sink_t* out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = iree_hal_module_debug_sink_null();

  iree_hal_module_debug_sink_npy_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*state),
                                             (void**)&state));
  state->host_allocator = host_allocator;
  state->call_index = 0;
  state->trace_dir = NULL;

  iree_status_t status = iree_allocator_malloc(
      host_allocator, trace_dir.size + 1, (void**)&state->trace_dir);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, state);
    return status;
  }
  memcpy(state->trace_dir, trace_dir.data, trace_dir.size);
  state->trace_dir[trace_dir.size] = '\0';

  out_sink->release.fn = iree_hal_module_debug_sink_npy_release;
  out_sink->release.user_data = state;
  out_sink->buffer_view_trace.fn = iree_hal_module_buffer_view_trace_npy;
  out_sink->buffer_view_trace.user_data = state;
  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_stdio(FILE* file) {
  iree_hal_module_debug_sink_t sink = {0};

#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
  sink.buffer_view_trace.fn = iree_hal_module_buffer_view_trace_stdio;
  sink.buffer_view_trace.user_data = file;
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE

  return sink;
}

#endif  // IREE_FILE_IO_ENABLE
