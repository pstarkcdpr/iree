// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/profile_recorder.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

//===----------------------------------------------------------------------===//
// Growable record buffers
//===----------------------------------------------------------------------===//

// Packed records of a single content type awaiting a flush.
typedef struct iree_hal_local_profile_buffer_t {
  uint8_t* data;
  iree_host_size_t length;
  iree_host_size_t capacity;
  // Number of records that could not be appended due to allocation failure.
  uint64_t dropped_record_count;
} iree_hal_local_profile_buffer_t;

static void iree_hal_local_profile_buffer_reset(
    iree_hal_local_profile_buffer_t* buffer, iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, buffer->data);
  memset(buffer, 0, sizeof(*buffer));
}

// Appends the concatenation of |chunk_count| |chunks| as one record. Records
// are all-or-nothing so that a partially appended record never reaches the
// sink.
static iree_status_t iree_hal_local_profile_buffer_append(
    iree_hal_local_profile_buffer_t* buffer, iree_host_size_t chunk_count,
    const iree_const_byte_span_t* chunks, iree_allocator_t host_allocator) {
  iree_host_size_t record_length = 0;
  for (iree_host_size_t i = 0; i < chunk_count; ++i) {
    record_length += chunks[i].data_length;
  }
  iree_host_size_t required_capacity = buffer->length + record_length;
  if (required_capacity > buffer->capacity) {
    iree_host_size_t new_capacity = iree_max(buffer->capacity * 2, 64 * 1024);
    new_capacity = iree_max(new_capacity, required_capacity);
    iree_status_t status = iree_allocator_realloc(host_allocator, new_capacity,
                                                  (void**)&buffer->data);
    if (!iree_status_is_ok(status)) {
      ++buffer->dropped_record_count;
      return status;
    }
    buffer->capacity = new_capacity;
  }
  for (iree_host_size_t i = 0; i < chunk_count; ++i) {
    memcpy(buffer->data + buffer->length, chunks[i].data,
           chunks[i].data_length);
    buffer->length += chunks[i].data_length;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_local_profile_recorder_t
//===----------------------------------------------------------------------===//

// The local drivers expose a single physical device per HAL device.
#define IREE_HAL_LOCAL_PROFILE_PHYSICAL_DEVICE_ORDINAL 0

// Data families local devices are able to produce.
#define IREE_HAL_LOCAL_PROFILE_SUPPORTED_DATA_FAMILIES    \
  (IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS | \
   IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA)

// An executable that has had its metadata emitted in the session.
typedef struct iree_hal_local_profile_executable_t {
  // Retained executable, or NULL if the slot is empty.
  iree_hal_executable_t* executable;
  // Session-local identifier referenced by events.
  uint64_t executable_id;
  // Number of entries in |export_enabled|.
  iree_host_size_t export_count;
  // Per-export flag indicating the capture filter export pattern matches.
  uint8_t* export_enabled;
} iree_hal_local_profile_executable_t;

// Order in which pending metadata is written so that consumers see
// descriptions before the events that reference them.
typedef enum iree_hal_local_profile_pending_e {
  IREE_HAL_LOCAL_PROFILE_PENDING_EXECUTABLES = 0,
  IREE_HAL_LOCAL_PROFILE_PENDING_EXECUTABLE_EXPORTS,
  IREE_HAL_LOCAL_PROFILE_PENDING_COMMAND_BUFFERS,
  IREE_HAL_LOCAL_PROFILE_PENDING_HOST_EXECUTION_EVENTS,
  IREE_HAL_LOCAL_PROFILE_PENDING_COUNT,
} iree_hal_local_profile_pending_t;

struct iree_hal_local_profile_recorder_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Retained sink receiving all chunks. NULL once the session has ended.
  iree_hal_profile_sink_t* sink;
  // Process-unique session identifier.
  uint64_t session_id;
  // Session name; references trailing storage.
  iree_string_view_t name;
  // Number of queues on the device.
  iree_host_size_t queue_count;

  // Requested data families after expanding lightweight statistics.
  iree_hal_device_profiling_data_families_t data_families;
  // Capture filter; the export pattern references trailing storage.
  iree_hal_profile_capture_filter_t capture_filter;

  // Monotonic identifier generators.
  iree_atomic_int64_t next_event_id;
  iree_atomic_int64_t next_command_buffer_id;

  // Serializes sink writes so that flushes are emitted in order.
  iree_slim_mutex_t flush_mutex;

  // Guards all fields below.
  iree_slim_mutex_t mutex;
  // True once the session has ended and new records are dropped.
  bool ended;
  // Open-addressed table of executables keyed by pointer.
  iree_hal_local_profile_executable_t* executables;
  iree_host_size_t executable_capacity;  // power of two or 0
  iree_host_size_t executable_count;
  uint64_t next_executable_id;
  // Records awaiting a flush.
  iree_hal_local_profile_buffer_t pending[IREE_HAL_LOCAL_PROFILE_PENDING_COUNT];
};

static uint64_t iree_hal_local_profile_next_session_id(void) {
  static iree_atomic_int64_t next_session_id = IREE_ATOMIC_VAR_INIT(1);
  return (uint64_t)iree_atomic_fetch_add(&next_session_id, 1,
                                         iree_memory_order_relaxed);
}

static iree_hal_profile_chunk_metadata_t
iree_hal_local_profile_recorder_chunk_metadata(
    iree_hal_local_profile_recorder_t* recorder,
    iree_string_view_t content_type, iree_string_view_t name) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = content_type;
  metadata.name = name;
  metadata.session_id = recorder->session_id;
  metadata.physical_device_ordinal =
      IREE_HAL_LOCAL_PROFILE_PHYSICAL_DEVICE_ORDINAL;
  return metadata;
}

static iree_status_t iree_hal_local_profile_recorder_write_session_metadata(
    iree_hal_local_profile_recorder_t* recorder) {
  iree_hal_profile_device_record_t device_record =
      iree_hal_profile_device_record_default();
  device_record.physical_device_ordinal =
      IREE_HAL_LOCAL_PROFILE_PHYSICAL_DEVICE_ORDINAL;
  device_record.queue_count = (uint32_t)recorder->queue_count;
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_local_profile_recorder_chunk_metadata(
          recorder, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES, IREE_SV("devices"));
  iree_const_byte_span_t device_span =
      iree_make_const_byte_span(&device_record, sizeof(device_record));
  IREE_RETURN_IF_ERROR(
      iree_hal_profile_sink_write(recorder->sink, &metadata, 1, &device_span));

  // Queue stream identifiers are the queue ordinals.
  for (iree_host_size_t i = 0; i < recorder->queue_count; ++i) {
    iree_hal_profile_queue_record_t queue_record =
        iree_hal_profile_queue_record_default();
    queue_record.physical_device_ordinal =
        IREE_HAL_LOCAL_PROFILE_PHYSICAL_DEVICE_ORDINAL;
    queue_record.queue_ordinal = (uint32_t)i;
    queue_record.stream_id = i;
    metadata = iree_hal_local_profile_recorder_chunk_metadata(
        recorder, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES, IREE_SV("queues"));
    metadata.queue_ordinal = (uint32_t)i;
    metadata.stream_id = i;
    iree_const_byte_span_t queue_span =
        iree_make_const_byte_span(&queue_record, sizeof(queue_record));
    IREE_RETURN_IF_ERROR(
        iree_hal_profile_sink_write(recorder->sink, &metadata, 1, &queue_span));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_resolve_data_families(
    const iree_hal_device_profiling_options_t* options,
    iree_hal_device_profiling_data_families_t* out_data_families) {
  iree_hal_device_profiling_data_families_t data_families =
      options->data_families;
  if (iree_hal_device_profiling_options_requests_lightweight_statistics(
          options)) {
    data_families |= IREE_HAL_LOCAL_PROFILE_SUPPORTED_DATA_FAMILIES;
  }
  iree_hal_device_profiling_data_families_t unsupported_families =
      data_families & ~IREE_HAL_LOCAL_PROFILE_SUPPORTED_DATA_FAMILIES;
  if (unsupported_families) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "local HAL devices execute on the host and only support the "
        "'host-execution' and 'executable-metadata' profiling data families "
        "(unsupported family bits 0x%llx)",
        (unsigned long long)unsupported_families);
  }
  if (options->counter_set_count != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "local HAL devices do not expose counters");
  }
  *out_data_families = data_families;
  return iree_ok_status();
}

iree_status_t iree_hal_local_profile_recorder_create(
    const iree_hal_device_profiling_options_t* options, iree_string_view_t name,
    iree_host_size_t queue_count, iree_allocator_t host_allocator,
    iree_hal_local_profile_recorder_t** out_recorder) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_recorder);
  *out_recorder = NULL;
  if (!options->sink) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profiling requires a sink");
  }
  iree_hal_device_profiling_data_families_t data_families = 0;
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_resolve_data_families(
      options, &data_families));
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_string_view_t export_pattern =
      options->capture_filter.executable_export_pattern;
  iree_hal_local_profile_recorder_t* recorder = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t name_offset = 0;
  iree_host_size_t pattern_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(*recorder), &total_size,
              IREE_STRUCT_FIELD(name.size, char, &name_offset),
              IREE_STRUCT_FIELD(export_pattern.size, char, &pattern_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&recorder));
  memset(recorder, 0, total_size);
  iree_atomic_ref_count_init(&recorder->ref_count);
  recorder->host_allocator = host_allocator;
  recorder->session_id = iree_hal_local_profile_next_session_id();
  iree_string_view_append_to_buffer(name, &recorder->name,
                                    (char*)recorder + name_offset);
  recorder->queue_count = queue_count;
  recorder->data_families = data_families;
  recorder->capture_filter = options->capture_filter;
  iree_string_view_append_to_buffer(
      export_pattern, &recorder->capture_filter.executable_export_pattern,
      (char*)recorder + pattern_offset);
  iree_atomic_store(&recorder->next_event_id, 1, iree_memory_order_relaxed);
  iree_atomic_store(&recorder->next_command_buffer_id, 1,
                    iree_memory_order_relaxed);
  recorder->next_executable_id = 1;
  iree_slim_mutex_initialize(&recorder->flush_mutex);
  iree_slim_mutex_initialize(&recorder->mutex);

  recorder->sink = options->sink;
  iree_hal_profile_sink_retain(recorder->sink);
  iree_hal_profile_chunk_metadata_t session_metadata =
      iree_hal_local_profile_recorder_chunk_metadata(
          recorder, IREE_HAL_PROFILE_CONTENT_TYPE_SESSION, recorder->name);
  iree_status_t status =
      iree_hal_profile_sink_begin_session(recorder->sink, &session_metadata);
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_write_session_metadata(recorder);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(iree_hal_profile_sink_end_session(
          recorder->sink, &session_metadata, iree_status_code(status)));
    }
  }
  if (!iree_status_is_ok(status)) {
    // The session was never (successfully) begun so there is nothing to end.
    iree_hal_profile_sink_release(recorder->sink);
    recorder->sink = NULL;
    recorder->ended = true;
    iree_hal_local_profile_recorder_release(recorder);
    recorder = NULL;
  }

  *out_recorder = recorder;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_local_profile_recorder_destroy(
    iree_hal_local_profile_recorder_t* recorder) {
  iree_allocator_t host_allocator = recorder->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_ignore(iree_hal_local_profile_recorder_end(recorder));

  for (iree_host_size_t i = 0; i < recorder->executable_capacity; ++i) {
    iree_hal_local_profile_executable_t* entry = &recorder->executables[i];
    if (!entry->executable) continue;
    iree_hal_executable_release(entry->executable);
    iree_allocator_free(host_allocator, entry->export_enabled);
  }
  iree_allocator_free(host_allocator, recorder->executables);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(recorder->pending); ++i) {
    iree_hal_local_profile_buffer_reset(&recorder->pending[i], host_allocator);
  }
  iree_slim_mutex_deinitialize(&recorder->mutex);
  iree_slim_mutex_deinitialize(&recorder->flush_mutex);
  iree_allocator_free(host_allocator, recorder);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_local_profile_recorder_retain(
    iree_hal_local_profile_recorder_t* recorder) {
  if (IREE_LIKELY(recorder)) {
    iree_atomic_ref_count_inc(&recorder->ref_count);
  }
}

void iree_hal_local_profile_recorder_release(
    iree_hal_local_profile_recorder_t* recorder) {
  if (IREE_LIKELY(recorder) &&
      iree_atomic_ref_count_dec(&recorder->ref_count) == 1) {
    iree_hal_local_profile_recorder_destroy(recorder);
  }
}

//===----------------------------------------------------------------------===//
// Flushing
//===----------------------------------------------------------------------===//

static void iree_hal_local_profile_pending_chunk_type(
    iree_hal_local_profile_pending_t pending, iree_string_view_t* out_type,
    iree_string_view_t* out_name) {
  switch (pending) {
    case IREE_HAL_LOCAL_PROFILE_PENDING_EXECUTABLES:
      *out_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES;
      *out_name = IREE_SV("executables");
      break;
    case IREE_HAL_LOCAL_PROFILE_PENDING_EXECUTABLE_EXPORTS:
      *out_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS;
      *out_name = IREE_SV("executable-exports");
      break;
    case IREE_HAL_LOCAL_PROFILE_PENDING_COMMAND_BUFFERS:
      *out_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS;
      *out_name = IREE_SV("command-buffers");
      break;
    default:
      *out_type = IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS;
      *out_name = IREE_SV("host-execution-events");
      break;
  }
}

// Writes all pending records. Caller must hold |flush_mutex| so that records
// taken from the pending buffers reach the sink in the order they were taken.
static iree_status_t iree_hal_local_profile_recorder_flush_locked(
    iree_hal_local_profile_recorder_t* recorder) {
  // Take ownership of the pending records so that recording can continue
  // while the (potentially slow) sink writes happen.
  iree_hal_local_profile_buffer_t pending[IREE_HAL_LOCAL_PROFILE_PENDING_COUNT];
  iree_slim_mutex_lock(&recorder->mutex);
  memcpy(pending, recorder->pending, sizeof(pending));
  memset(recorder->pending, 0, sizeof(recorder->pending));
  iree_slim_mutex_unlock(&recorder->mutex);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(pending); ++i) {
    iree_hal_local_profile_buffer_t* buffer = &pending[i];
    if (iree_status_is_ok(status) && recorder->sink &&
        (buffer->length > 0 || buffer->dropped_record_count > 0)) {
      iree_string_view_t content_type = iree_string_view_empty();
      iree_string_view_t name = iree_string_view_empty();
      iree_hal_local_profile_pending_chunk_type(
          (iree_hal_local_profile_pending_t)i, &content_type, &name);
      iree_hal_profile_chunk_metadata_t metadata =
          iree_hal_local_profile_recorder_chunk_metadata(recorder, content_type,
                                                         name);
      if (buffer->dropped_record_count > 0) {
        metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
        metadata.dropped_record_count = buffer->dropped_record_count;
      }
      iree_const_byte_span_t span =
          iree_make_const_byte_span(buffer->data, buffer->length);
      status = iree_hal_profile_sink_write(recorder->sink, &metadata,
                                           buffer->length ? 1 : 0, &span);
    }
    iree_hal_local_profile_buffer_reset(buffer, recorder->host_allocator);
  }
  return status;
}

iree_status_t iree_hal_local_profile_recorder_flush(
    iree_hal_local_profile_recorder_t* recorder) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&recorder->flush_mutex);
  iree_status_t status = iree_hal_local_profile_recorder_flush_locked(recorder);
  iree_slim_mutex_unlock(&recorder->flush_mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_local_profile_recorder_end(
    iree_hal_local_profile_recorder_t* recorder) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&recorder->flush_mutex);

  iree_slim_mutex_lock(&recorder->mutex);
  const bool already_ended = recorder->ended;
  recorder->ended = true;
  iree_slim_mutex_unlock(&recorder->mutex);

  iree_status_t status = iree_ok_status();
  if (!already_ended && recorder->sink) {
    status = iree_hal_local_profile_recorder_flush_locked(recorder);
    iree_hal_profile_chunk_metadata_t session_metadata =
        iree_hal_local_profile_recorder_chunk_metadata(
            recorder, IREE_HAL_PROFILE_CONTENT_TYPE_SESSION, recorder->name);
    status = iree_status_join(status, iree_hal_profile_sink_end_session(
                                          recorder->sink, &session_metadata,
                                          iree_status_code(status)));
    iree_hal_profile_sink_release(recorder->sink);
    recorder->sink = NULL;
  }

  iree_slim_mutex_unlock(&recorder->flush_mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Command buffers
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_local_profile_recorder_register_command_buffer(
    iree_hal_local_profile_recorder_t* recorder,
    iree_hal_command_buffer_t* command_buffer,
    uint64_t* out_command_buffer_id) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(out_command_buffer_id);
  const uint64_t command_buffer_id = (uint64_t)iree_atomic_fetch_add(
      &recorder->next_command_buffer_id, 1, iree_memory_order_relaxed);
  *out_command_buffer_id = command_buffer_id;

  iree_hal_profile_command_buffer_record_t record =
      iree_hal_profile_command_buffer_record_default();
  record.command_buffer_id = command_buffer_id;
  record.mode = iree_hal_command_buffer_mode(command_buffer);
  record.command_categories =
      iree_hal_command_buffer_allowed_categories(command_buffer);
  record.queue_affinity = command_buffer->queue_affinity;
  record.physical_device_ordinal =
      IREE_HAL_LOCAL_PROFILE_PHYSICAL_DEVICE_ORDINAL;
  iree_const_byte_span_t span =
      iree_make_const_byte_span(&record, sizeof(record));

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&recorder->mutex);
  if (!recorder->ended) {
    status = iree_hal_local_profile_buffer_append(
        &recorder->pending[IREE_HAL_LOCAL_PROFILE_PENDING_COMMAND_BUFFERS], 1,
        &span, recorder->host_allocator);
  }
  iree_slim_mutex_unlock(&recorder->mutex);
  return status;
}

//===----------------------------------------------------------------------===//
// Executables
//===----------------------------------------------------------------------===//

static iree_host_size_t iree_hal_local_profile_executable_hash(
    const iree_hal_executable_t* executable) {
  // Fibonacci hashing of the pointer; low bits are mostly alignment.
  uint64_t value = (uint64_t)(uintptr_t)executable;
  return (iree_host_size_t)((value * 0x9E3779B97F4A7C15ull) >> 32);
}

// Returns the slot for |executable|: either the slot holding it or the empty
// slot it would be inserted into. Requires a non-full table.
static iree_hal_local_profile_executable_t*
iree_hal_local_profile_recorder_find_slot(
    iree_hal_local_profile_executable_t* table, iree_host_size_t capacity,
    const iree_hal_executable_t* executable) {
  iree_host_size_t mask = capacity - 1;
  iree_host_size_t index =
      iree_hal_local_profile_executable_hash(executable) & mask;
  while (table[index].executable && table[index].executable != executable) {
    index = (index + 1) & mask;
  }
  return &table[index];
}

// Grows the executable table so that one more entry keeps it at most half
// full. Caller must hold |mutex|.
static iree_status_t iree_hal_local_profile_recorder_reserve_executable(
    iree_hal_local_profile_recorder_t* recorder) {
  if ((recorder->executable_count + 1) * 2 <= recorder->executable_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      recorder->executable_capacity ? recorder->executable_capacity * 2 : 64;
  iree_hal_local_profile_executable_t* new_table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(recorder->host_allocator, new_capacity,
                                  sizeof(*new_table), (void**)&new_table));
  memset(new_table, 0, new_capacity * sizeof(*new_table));
  for (iree_host_size_t i = 0; i < recorder->executable_capacity; ++i) {
    iree_hal_local_profile_executable_t* entry = &recorder->executables[i];
    if (!entry->executable) continue;
    *iree_hal_local_profile_recorder_find_slot(new_table, new_capacity,
                                               entry->executable) = *entry;
  }
  iree_allocator_free(recorder->host_allocator, recorder->executables);
  recorder->executables = new_table;
  recorder->executable_capacity = new_capacity;
  return iree_ok_status();
}

// Appends the executable record and all export records for |entry|. Caller
// must hold |mutex|.
static iree_status_t iree_hal_local_profile_recorder_append_executable_metadata(
    iree_hal_local_profile_recorder_t* recorder,
    iree_hal_local_profile_executable_t* entry) {
  iree_hal_profile_executable_record_t executable_record =
      iree_hal_profile_executable_record_default();
  executable_record.executable_id = entry->executable_id;
  executable_record.export_count = (uint32_t)entry->export_count;
  iree_const_byte_span_t executable_span =
      iree_make_const_byte_span(&executable_record, sizeof(executable_record));
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_buffer_append(
      &recorder->pending[IREE_HAL_LOCAL_PROFILE_PENDING_EXECUTABLES], 1,
      &executable_span, recorder->host_allocator));

  for (iree_host_size_t i = 0; i < entry->export_count; ++i) {
    iree_hal_executable_export_info_t info;
    memset(&info, 0, sizeof(info));
    IREE_RETURN_IF_ERROR(iree_hal_executable_export_info(
        entry->executable, (iree_hal_executable_export_ordinal_t)i, &info));
    iree_hal_profile_executable_export_record_t export_record =
        iree_hal_profile_executable_export_record_default();
    export_record.executable_id = entry->executable_id;
    export_record.export_ordinal = (uint32_t)i;
    export_record.constant_count = info.constant_count;
    export_record.binding_count = info.binding_count;
    export_record.parameter_count = info.parameter_count;
    memcpy(export_record.workgroup_size, info.workgroup_size,
           sizeof(export_record.workgroup_size));
    export_record.name_length = (uint32_t)info.name.size;
    export_record.record_length =
        (uint32_t)(sizeof(export_record) + export_record.name_length);
    iree_const_byte_span_t export_spans[2] = {
        iree_make_const_byte_span(&export_record, sizeof(export_record)),
        iree_make_const_byte_span(info.name.data, info.name.size),
    };
    IREE_RETURN_IF_ERROR(iree_hal_local_profile_buffer_append(
        &recorder->pending[IREE_HAL_LOCAL_PROFILE_PENDING_EXECUTABLE_EXPORTS],
        IREE_ARRAYSIZE(export_spans), export_spans, recorder->host_allocator));

    bool enabled = true;
    if (iree_any_bit_set(
            recorder->capture_filter.flags,
            IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN)) {
      enabled = iree_string_view_match_pattern(
          info.name, recorder->capture_filter.executable_export_pattern);
    }
    entry->export_enabled[i] = enabled ? 1 : 0;
  }
  return iree_ok_status();
}

// Returns the session entry for |executable|, registering it and emitting its
// metadata on first use. Caller must hold |mutex|.
static iree_status_t iree_hal_local_profile_recorder_lookup_executable(
    iree_hal_local_profile_recorder_t* recorder,
    iree_hal_executable_t* executable,
    iree_hal_local_profile_executable_t** out_entry) {
  *out_entry = NULL;
  if (recorder->executable_capacity > 0) {
    iree_hal_local_profile_executable_t* entry =
        iree_hal_local_profile_recorder_find_slot(
            recorder->executables, recorder->executable_capacity, executable);
    if (entry->executable) {
      *out_entry = entry;
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_local_profile_recorder_reserve_executable(recorder));
  const iree_host_size_t export_count =
      iree_hal_executable_export_count(executable);
  uint8_t* export_enabled = NULL;
  if (export_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        recorder->host_allocator, export_count, (void**)&export_enabled));
  }
  iree_hal_local_profile_executable_t* entry =
      iree_hal_local_profile_recorder_find_slot(
          recorder->executables, recorder->executable_capacity, executable);
  entry->executable = executable;
  iree_hal_executable_retain(executable);
  entry->executable_id = recorder->next_executable_id++;
  entry->export_count = export_count;
  entry->export_enabled = export_enabled;
  ++recorder->executable_count;

  // The entry stays registered even if metadata emission fails: its id must
  // not be reused and the failure is surfaced to the caller.
  IREE_RETURN_IF_ERROR(
      iree_hal_local_profile_recorder_append_executable_metadata(recorder,
                                                                 entry));
  *out_entry = entry;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Host execution events
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_local_profile_recorder_prepare_dispatch(
    iree_hal_local_profile_recorder_t* recorder, uint32_t queue_ordinal,
    uint64_t command_buffer_id, uint32_t command_index,
    iree_hal_executable_t* executable, uint32_t export_ordinal,
    bool* out_enabled, iree_hal_profile_host_execution_event_t* out_event) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_enabled);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_enabled = false;

  if (!iree_hal_profile_capture_filter_matches_location(
          &recorder->capture_filter, command_buffer_id, command_index,
          IREE_HAL_LOCAL_PROFILE_PHYSICAL_DEVICE_ORDINAL, queue_ordinal)) {
    return iree_ok_status();
  }

  // Executable metadata is registered even when host execution events are not
  // requested so that 'executable-metadata' alone describes what ran.
  iree_slim_mutex_lock(&recorder->mutex);
  iree_hal_local_profile_executable_t* entry = NULL;
  iree_status_t status = iree_ok_status();
  if (!recorder->ended) {
    status = iree_hal_local_profile_recorder_lookup_executable(
        recorder, executable, &entry);
  }
  uint64_t executable_id = 0;
  bool export_enabled = false;
  if (entry) {
    executable_id = entry->executable_id;
    export_enabled = export_ordinal < entry->export_count &&
                     entry->export_enabled[export_ordinal];
  }
  iree_slim_mutex_unlock(&recorder->mutex);
  IREE_RETURN_IF_ERROR(status);

  if (!export_enabled ||
      !iree_any_bit_set(recorder->data_families,
                        IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    return iree_ok_status();
  }

  iree_hal_profile_host_execution_event_t event =
      iree_hal_profile_host_execution_event_default();
  event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  if (command_buffer_id != 0) {
    event.flags |= IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_COMMAND_BUFFER;
  }
  // Assigned at record time so that event ids follow program order even when
  // dispatches complete out of order.
  event.event_id = (uint64_t)iree_atomic_fetch_add(&recorder->next_event_id, 1,
                                                   iree_memory_order_relaxed);
  event.command_buffer_id = command_buffer_id;
  event.executable_id = executable_id;
  event.stream_id = queue_ordinal;
  event.physical_device_ordinal =
      IREE_HAL_LOCAL_PROFILE_PHYSICAL_DEVICE_ORDINAL;
  event.queue_ordinal = queue_ordinal;
  event.command_index = command_buffer_id != 0 ? command_index : UINT32_MAX;
  event.export_ordinal = export_ordinal;
  event.operation_count = 1;
  *out_event = event;
  *out_enabled = true;
  return iree_ok_status();
}

void iree_hal_local_profile_recorder_append_host_execution_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_profile_host_execution_event_t* event) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(event);
  iree_const_byte_span_t span =
      iree_make_const_byte_span(event, sizeof(*event));
  iree_slim_mutex_lock(&recorder->mutex);
  if (!recorder->ended) {
    // Allocation failures are counted in the buffer and reported as dropped
    // records on flush.
    iree_status_ignore(iree_hal_local_profile_buffer_append(
        &recorder
             ->pending[IREE_HAL_LOCAL_PROFILE_PENDING_HOST_EXECUTION_EVENTS],
        1, &span, recorder->host_allocator));
  }
  iree_slim_mutex_unlock(&recorder->mutex);
}
