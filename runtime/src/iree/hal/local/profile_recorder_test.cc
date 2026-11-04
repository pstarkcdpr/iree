// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/profile_recorder.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Capturing sink
//===----------------------------------------------------------------------===//

struct CapturedChunk {
  std::string content_type;
  uint64_t flags = 0;
  uint64_t dropped_record_count = 0;
  std::vector<uint8_t> payload;
};

struct CapturingSink {
  iree_hal_resource_t resource;
  int begin_count = 0;
  int end_count = 0;
  std::vector<CapturedChunk> chunks;
};

extern const iree_hal_profile_sink_vtable_t capturing_sink_vtable;

static CapturingSink* CastSink(iree_hal_profile_sink_t* sink) {
  return reinterpret_cast<CapturingSink*>(sink);
}

static void CapturingSinkDestroy(iree_hal_profile_sink_t* sink) {}

static iree_status_t CapturingSinkBeginSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  ++CastSink(sink)->begin_count;
  return iree_ok_status();
}

static iree_status_t CapturingSinkWrite(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  CapturedChunk chunk;
  chunk.content_type =
      std::string(metadata->content_type.data, metadata->content_type.size);
  chunk.flags = metadata->flags;
  chunk.dropped_record_count = metadata->dropped_record_count;
  for (iree_host_size_t i = 0; i < iovec_count; ++i) {
    chunk.payload.insert(chunk.payload.end(), iovecs[i].data,
                         iovecs[i].data + iovecs[i].data_length);
  }
  CastSink(sink)->chunks.push_back(std::move(chunk));
  return iree_ok_status();
}

static iree_status_t CapturingSinkEndSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  ++CastSink(sink)->end_count;
  return iree_ok_status();
}

const iree_hal_profile_sink_vtable_t capturing_sink_vtable = {
    /*.destroy=*/CapturingSinkDestroy,
    /*.begin_session=*/CapturingSinkBeginSession,
    /*.write=*/CapturingSinkWrite,
    /*.end_session=*/CapturingSinkEndSession,
};

//===----------------------------------------------------------------------===//
// Fake executable
//===----------------------------------------------------------------------===//

struct FakeExecutable {
  iree_hal_resource_t resource;
  std::vector<std::string> export_names;
};

extern const iree_hal_executable_vtable_t fake_executable_vtable;

static FakeExecutable* CastExecutable(iree_hal_executable_t* executable) {
  return reinterpret_cast<FakeExecutable*>(executable);
}

static void FakeExecutableDestroy(iree_hal_executable_t* executable) {}

static iree_host_size_t FakeExecutableExportCount(
    iree_hal_executable_t* executable) {
  return CastExecutable(executable)->export_names.size();
}

static iree_status_t FakeExecutableExportInfo(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  const std::string& name =
      CastExecutable(executable)->export_names[export_ordinal];
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = iree_make_string_view(name.data(), name.size());
  out_info->binding_count = 2;
  out_info->workgroup_size[0] = 1;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

const iree_hal_executable_vtable_t fake_executable_vtable = {
    /*.destroy=*/FakeExecutableDestroy,
    /*.export_count=*/FakeExecutableExportCount,
    /*.export_info=*/FakeExecutableExportInfo,
    /*.export_parameters=*/nullptr,
    /*.lookup_export_by_name=*/nullptr,
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

class ProfileRecorderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_resource_initialize(&capturing_sink_vtable, &sink_.resource);
    iree_hal_resource_initialize(&fake_executable_vtable,
                                 &executable_.resource);
    executable_.export_names = {"main_dispatch_0_matmul",
                                "main_dispatch_1_add"};
    memset(&command_buffer_, 0, sizeof(command_buffer_));
    command_buffer_.mode = IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT;
    command_buffer_.allowed_categories = IREE_HAL_COMMAND_CATEGORY_DISPATCH;
    command_buffer_.queue_affinity = 1;
  }

  iree_hal_device_profiling_options_t HostExecutionOptions() {
    iree_hal_device_profiling_options_t options = {};
    options.data_families =
        IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS;
    options.sink = sink();
    return options;
  }

  iree_hal_profile_sink_t* sink() {
    return reinterpret_cast<iree_hal_profile_sink_t*>(&sink_);
  }
  iree_hal_executable_t* executable() {
    return reinterpret_cast<iree_hal_executable_t*>(&executable_);
  }

  // Records and completes one dispatch of |export_ordinal|.
  void RecordDispatch(iree_hal_local_profile_recorder_t* recorder,
                      uint64_t command_buffer_id, uint32_t command_index,
                      uint32_t export_ordinal, bool* out_enabled) {
    iree_hal_profile_host_execution_event_t event;
    IREE_ASSERT_OK(iree_hal_local_profile_recorder_prepare_dispatch(
        recorder, /*queue_ordinal=*/0, command_buffer_id, command_index,
        executable(), export_ordinal, out_enabled, &event));
    if (!*out_enabled) return;
    event.start_host_time_ns = 100;
    event.end_host_time_ns = 150;
    event.tile_count = 4;
    event.tile_duration_sum_ns = 120;
    event.status_code = IREE_STATUS_OK;
    iree_hal_local_profile_recorder_append_host_execution_event(recorder,
                                                                &event);
  }

  std::vector<std::string> ContentTypes() {
    std::vector<std::string> types;
    for (const auto& chunk : sink_.chunks) types.push_back(chunk.content_type);
    return types;
  }

  template <typename T>
  std::vector<T> Records(const char* content_type) {
    std::vector<T> records;
    for (const auto& chunk : sink_.chunks) {
      if (chunk.content_type != content_type) continue;
      for (size_t offset = 0; offset + sizeof(T) <= chunk.payload.size();
           offset += sizeof(T)) {
        T record;
        memcpy(&record, chunk.payload.data() + offset, sizeof(T));
        records.push_back(record);
      }
    }
    return records;
  }

  CapturingSink sink_;
  FakeExecutable executable_;
  iree_hal_command_buffer_t command_buffer_;
};

static const char kHostExecutionEvents[] =
    "application/vnd.iree.hal.profile.host-execution-events";

TEST_F(ProfileRecorderTest, RejectsDeviceTimestampedFamilies) {
  iree_hal_device_profiling_options_t options = HostExecutionOptions();
  options.data_families |= IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  iree_hal_local_profile_recorder_t* recorder = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_local_profile_recorder_create(
                            &options, IREE_SV("local-test"), 1,
                            iree_allocator_system(), &recorder));
  EXPECT_EQ(recorder, nullptr);
  EXPECT_EQ(sink_.begin_count, 0);
}

TEST_F(ProfileRecorderTest, WritesSessionAndQueueMetadataOnCreate) {
  iree_hal_device_profiling_options_t options = HostExecutionOptions();
  iree_hal_local_profile_recorder_t* recorder = nullptr;
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_create(
      &options, IREE_SV("local-test"), /*queue_count=*/2,
      iree_allocator_system(), &recorder));
  EXPECT_EQ(sink_.begin_count, 1);
  auto queues = Records<iree_hal_profile_queue_record_t>(
      "application/vnd.iree.hal.profile.queues");
  ASSERT_EQ(queues.size(), 2u);
  EXPECT_EQ(queues[1].queue_ordinal, 1u);
  EXPECT_EQ(queues[1].stream_id, 1u);

  IREE_EXPECT_OK(iree_hal_local_profile_recorder_end(recorder));
  EXPECT_EQ(sink_.end_count, 1);
  iree_hal_local_profile_recorder_release(recorder);
  EXPECT_EQ(sink_.end_count, 1);
}

TEST_F(ProfileRecorderTest, FlushWritesMetadataBeforeEvents) {
  iree_hal_device_profiling_options_t options = HostExecutionOptions();
  iree_hal_local_profile_recorder_t* recorder = nullptr;
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_create(
      &options, IREE_SV("local-test"), 1, iree_allocator_system(), &recorder));
  sink_.chunks.clear();

  uint64_t command_buffer_id = 0;
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_register_command_buffer(
      recorder, &command_buffer_, &command_buffer_id));
  EXPECT_NE(command_buffer_id, 0u);
  bool enabled = false;
  RecordDispatch(recorder, command_buffer_id, 0, /*export_ordinal=*/1,
                 &enabled);
  EXPECT_TRUE(enabled);
  RecordDispatch(recorder, command_buffer_id, 1, /*export_ordinal=*/1,
                 &enabled);
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_flush(recorder));

  EXPECT_EQ(ContentTypes(),
            (std::vector<std::string>{
                "application/vnd.iree.hal.profile.executables",
                "application/vnd.iree.hal.profile.executable-exports",
                "application/vnd.iree.hal.profile.command-buffers",
                kHostExecutionEvents,
            }));
  auto events =
      Records<iree_hal_profile_host_execution_event_t>(kHostExecutionEvents);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].type, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH);
  EXPECT_EQ(events[0].export_ordinal, 1u);
  EXPECT_EQ(events[0].command_buffer_id, command_buffer_id);
  EXPECT_EQ(events[1].command_index, 1u);
  EXPECT_LT(events[0].event_id, events[1].event_id);
  EXPECT_EQ(events[0].executable_id, events[1].executable_id);
  EXPECT_TRUE(iree_all_bits_set(
      events[0].flags,
      IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_COMMAND_BUFFER));

  // Executable metadata is only emitted the first time it is used.
  sink_.chunks.clear();
  RecordDispatch(recorder, command_buffer_id, 2, /*export_ordinal=*/0,
                 &enabled);
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_flush(recorder));
  EXPECT_EQ(ContentTypes(), (std::vector<std::string>{kHostExecutionEvents}));

  IREE_EXPECT_OK(iree_hal_local_profile_recorder_end(recorder));
  iree_hal_local_profile_recorder_release(recorder);
}

TEST_F(ProfileRecorderTest, ExportPatternFiltersDispatches) {
  iree_hal_device_profiling_options_t options = HostExecutionOptions();
  options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN;
  options.capture_filter.executable_export_pattern = IREE_SV("*matmul*");
  iree_hal_local_profile_recorder_t* recorder = nullptr;
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_create(
      &options, IREE_SV("local-test"), 1, iree_allocator_system(), &recorder));

  bool matmul_enabled = false;
  RecordDispatch(recorder, /*command_buffer_id=*/0, 0, /*export_ordinal=*/0,
                 &matmul_enabled);
  bool add_enabled = true;
  RecordDispatch(recorder, /*command_buffer_id=*/0, 0, /*export_ordinal=*/1,
                 &add_enabled);
  EXPECT_TRUE(matmul_enabled);
  EXPECT_FALSE(add_enabled);

  IREE_EXPECT_OK(iree_hal_local_profile_recorder_end(recorder));
  auto events =
      Records<iree_hal_profile_host_execution_event_t>(kHostExecutionEvents);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].export_ordinal, 0u);
  // Direct dispatches have no command buffer index.
  EXPECT_EQ(events[0].command_index, UINT32_MAX);
  iree_hal_local_profile_recorder_release(recorder);
}

TEST_F(ProfileRecorderTest, MetadataOnlySessionRecordsNoEvents) {
  iree_hal_device_profiling_options_t options = HostExecutionOptions();
  options.data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  iree_hal_local_profile_recorder_t* recorder = nullptr;
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_create(
      &options, IREE_SV("local-test"), 1, iree_allocator_system(), &recorder));
  bool enabled = true;
  RecordDispatch(recorder, 0, 0, 0, &enabled);
  EXPECT_FALSE(enabled);
  IREE_EXPECT_OK(iree_hal_local_profile_recorder_end(recorder));
  EXPECT_EQ(Records<iree_hal_profile_executable_record_t>(
                "application/vnd.iree.hal.profile.executables")
                .size(),
            1u);
  EXPECT_TRUE(
      Records<iree_hal_profile_host_execution_event_t>(kHostExecutionEvents)
          .empty());
  iree_hal_local_profile_recorder_release(recorder);
}

TEST_F(ProfileRecorderTest, EventsAfterEndAreDropped) {
  iree_hal_device_profiling_options_t options = HostExecutionOptions();
  iree_hal_local_profile_recorder_t* recorder = nullptr;
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_create(
      &options, IREE_SV("local-test"), 1, iree_allocator_system(), &recorder));

  // A dispatch recorded during the session that completes after it ends.
  bool enabled = false;
  iree_hal_profile_host_execution_event_t event;
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_prepare_dispatch(
      recorder, 0, 0, 0, executable(), 0, &enabled, &event));
  ASSERT_TRUE(enabled);
  IREE_ASSERT_OK(iree_hal_local_profile_recorder_end(recorder));
  const size_t chunk_count = sink_.chunks.size();
  iree_hal_local_profile_recorder_append_host_execution_event(recorder, &event);
  IREE_EXPECT_OK(iree_hal_local_profile_recorder_flush(recorder));
  EXPECT_EQ(sink_.chunks.size(), chunk_count);
  EXPECT_EQ(sink_.end_count, 1);
  iree_hal_local_profile_recorder_release(recorder);
}

}  // namespace
