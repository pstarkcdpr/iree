// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_PROFILE_RECORDER_H_
#define IREE_HAL_LOCAL_PROFILE_RECORDER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_local_profile_recorder_t
//===----------------------------------------------------------------------===//

// HAL-native profiling session shared by the local (CPU) HAL drivers.
//
// Local dispatches execute on host threads, so their spans are reported as
// IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS records in the
// iree_time_now() clock domain rather than as device-timestamped dispatch
// events. Supported data families are
// IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS and
// IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA; the
// LIGHTWEIGHT_STATISTICS flag requests both.
//
// Executables are usually loaded before a profiling session begins, so
// executable and export metadata is emitted lazily the first time an
// executable is dispatched within the session. The recorder retains each
// executable it describes so that its identity stays unique for the session.
//
// Recording is thread-safe. Records are buffered in memory and only written to
// the sink from iree_hal_local_profile_recorder_flush and
// iree_hal_local_profile_recorder_end so that sink I/O never happens on the
// execution path.
//
// The recorder is reference counted: command buffers recorded while profiling
// is active retain it so that work which completes after profiling ends can
// still safely report (and have its records dropped).
typedef struct iree_hal_local_profile_recorder_t
    iree_hal_local_profile_recorder_t;

// Creates a recorder for a device exposing |queue_count| queues and begins a
// profiling session on |options->sink| named |name|. Device and queue metadata
// is written to the sink immediately. Fails if |options| requests data that
// local devices cannot produce.
iree_status_t iree_hal_local_profile_recorder_create(
    const iree_hal_device_profiling_options_t* options, iree_string_view_t name,
    iree_host_size_t queue_count, iree_allocator_t host_allocator,
    iree_hal_local_profile_recorder_t** out_recorder);

// Retains the given |recorder| for the caller.
void iree_hal_local_profile_recorder_retain(
    iree_hal_local_profile_recorder_t* recorder);

// Releases the given |recorder| from the caller. The session is ended if it
// has not been already.
void iree_hal_local_profile_recorder_release(
    iree_hal_local_profile_recorder_t* recorder);

// Writes all buffered records to the sink.
iree_status_t iree_hal_local_profile_recorder_flush(
    iree_hal_local_profile_recorder_t* recorder);

// Flushes all buffered records and ends the sink session. Records produced
// after this (by work that was still in flight) are dropped.
iree_status_t iree_hal_local_profile_recorder_end(
    iree_hal_local_profile_recorder_t* recorder);

// Assigns a session-unique identifier to |command_buffer| and records its
// metadata. Must be called once per recording of a one-shot command buffer.
iree_status_t iree_hal_local_profile_recorder_register_command_buffer(
    iree_hal_local_profile_recorder_t* recorder,
    iree_hal_command_buffer_t* command_buffer, uint64_t* out_command_buffer_id);

// Prepares a host-execution event for a dispatch of |export_ordinal| in
// |executable| recorded as the |command_index|-th dispatch of
// |command_buffer_id| (0 for direct queue dispatches) on |queue_ordinal|.
//
// |out_enabled| is set to false when the session is not capturing host
// execution events or the capture filter excludes the dispatch. Otherwise
// |out_event| has its identity fields populated and the caller fills in the
// timing, tile, workgroup, flag, and status fields once the dispatch completes
// and passes it to iree_hal_local_profile_recorder_append_host_execution_event.
iree_status_t iree_hal_local_profile_recorder_prepare_dispatch(
    iree_hal_local_profile_recorder_t* recorder, uint32_t queue_ordinal,
    uint64_t command_buffer_id, uint32_t command_index,
    iree_hal_executable_t* executable, uint32_t export_ordinal,
    bool* out_enabled, iree_hal_profile_host_execution_event_t* out_event);

// Appends a completed host execution |event|. May be called from any thread.
// Events that cannot be retained are counted and reported as dropped records
// on the next flush.
void iree_hal_local_profile_recorder_append_host_execution_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_profile_host_execution_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_PROFILE_RECORDER_H_
