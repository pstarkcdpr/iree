// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A tracing provider that records per-dispatch GPU times to a CSV and nothing
// else. It exists so that dispatch costs can be measured without a Tracy build
// or a Tracy UI: the Vulkan HAL already writes a vkCmdWriteTimestamp on either
// side of every vkCmdDispatch and hands the results to the provider through
// iree_tracing_gpu_zone_*, but the only in-tree consumer of those callbacks is
// Tracy. This is the smallest possible consumer.
//
// All host-side (CPU) instrumentation is compiled to nothing. INSTRUMENTATION
// cannot simply be masked off -- vulkan/native_executable.h declares the
// pipeline's source_location inside IREE_TRACE(), and vulkan/tracing.h's zone
// macros expand TracyConcat and brace-initialize iree_tracing_location_t -- so
// the feature bit stays on and every CPU macro below is a no-op instead.
//
// Configuration is by environment variable, because IREE_TRACE_APP_ENTER runs
// before iree_flags_parse_checked and so cannot see IREE flags:
//   IREE_GPU_CSV           path for the per-dispatch event CSV. Defaults to
//                          iree_gpu_dispatches.csv in the working directory;
//                          set it empty to disable output entirely. A summary
//                          aggregated by export name is written alongside it.
//   IREE_GPU_CSV_TRANSFERS set to 1 to also record fills/copies/updates and
//                          debug groups, which are excluded by default.
//
// Caveat worth repeating from vulkan/tracing.h: a timestamp is, in the average
// case, as expensive as a full global execution barrier. Dispatches that could
// have overlapped are serialised, so the numbers are an upper bound on each
// dispatch's cost. The *shape* of the cost distribution survives, which is what
// makes this usable for choosing model split points and unusable as an answer
// to "how fast does this model run".

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"

#ifndef IREE_BASE_TRACING_CSVGPU_H_
#define IREE_BASE_TRACING_CSVGPU_H_

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

// Filter to only supported features. INSTRUMENTATION is required (see above)
// even though it does nothing here; ALLOCATION_TRACKING and LOG_MESSAGES are
// masked out so tracing.h supplies its own no-op definitions for them.
#if !defined(IREE_TRACING_FEATURES)
#define IREE_TRACING_FEATURES                \
  ((IREE_TRACING_FEATURES_REQUESTED) &       \
   (IREE_TRACING_FEATURE_INSTRUMENTATION |   \
    IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE))
#endif  // !IREE_TRACING_FEATURES

//===----------------------------------------------------------------------===//
// C API used for tracing control
//===----------------------------------------------------------------------===//

// Local zone ID used for the C IREE_TRACE_ZONE_* macros. Always 0 here.
typedef uint32_t iree_zone_id_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if IREE_TRACING_FEATURES

// Must match Tracy's ___tracy_source_location_data field order:
// vulkan/tracing.h brace-initializes this positionally as
// {NULL, __FUNCTION__, __FILE__, __LINE__, 0}.
typedef struct iree_tracing_location_t {
  const char* name;
  const char* function;
  const char* file;
  uint32_t line;
  uint32_t color;
} iree_tracing_location_t;

// Matches Tracy's GpuContextType; the Vulkan HAL passes _VULKAN.
typedef enum iree_tracing_gpu_context_type_e {
  IREE_TRACING_GPU_CONTEXT_TYPE_INVALID = 0,
  IREE_TRACING_GPU_CONTEXT_TYPE_OPENGL,
  IREE_TRACING_GPU_CONTEXT_TYPE_VULKAN,
  IREE_TRACING_GPU_CONTEXT_TYPE_OPENCL,
  IREE_TRACING_GPU_CONTEXT_TYPE_DIRECT3D12,
  IREE_TRACING_GPU_CONTEXT_TYPE_DIRECT3D11,
} iree_tracing_gpu_context_type_t;

// vulkan/tracing.h uses TracyConcat in its zone macros; it leaks out of Tracy
// and every provider has to supply it.
#define TracyConcat(x, y) TracyConcat_(x, y)
#define TracyConcat_(x, y) x##y

#define IREE_TRACE_STRLEN(literal) (sizeof(literal) - 1)

void iree_tracing_csvgpu_initialize(void);
void iree_tracing_csvgpu_deinitialize(void);

int64_t iree_tracing_time(void);
int64_t iree_tracing_frequency(void);

uint8_t iree_tracing_gpu_context_allocate(iree_tracing_gpu_context_type_t type,
                                          const char* name, size_t name_length,
                                          bool is_calibrated,
                                          uint64_t cpu_timestamp,
                                          uint64_t gpu_timestamp,
                                          float timestamp_period);
void iree_tracing_gpu_context_calibrate(uint8_t context_id, int64_t cpu_delta,
                                        int64_t cpu_timestamp,
                                        int64_t gpu_timestamp);
void iree_tracing_gpu_zone_begin(uint8_t context_id, uint16_t query_id,
                                 const iree_tracing_location_t* src_loc);
void iree_tracing_gpu_zone_begin_external(
    uint8_t context_id, uint16_t query_id, const char* file_name,
    size_t file_name_length, uint32_t line, const char* function_name,
    size_t function_name_length, const char* name, size_t name_length);
void iree_tracing_gpu_zone_end(uint8_t context_id, uint16_t query_id);
void iree_tracing_gpu_zone_notify(uint8_t context_id, uint16_t query_id,
                                  int64_t gpu_timestamp);

#endif  // IREE_TRACING_FEATURES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Instrumentation macros (C)
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

// Load-bearing: vulkan/native_executable.h declares
//   IREE_TRACE(iree_hal_vulkan_source_location_t source_location;)
// and direct_command_buffer.cc reads that field. If this expands to nothing the
// Vulkan driver does not compile.
#define IREE_TRACE(expr) expr

#define IREE_TRACE_APP_ENTER() iree_tracing_csvgpu_initialize()
#define IREE_TRACE_APP_EXIT(exit_code) \
  (void)(exit_code), iree_tracing_csvgpu_deinitialize()

// Everything below is host-side instrumentation we deliberately do not record.
// Bodies match tracing.h's disabled path so callers keep compiling and pay
// nothing at runtime.
#define IREE_TRACE_SET_APP_INFO(value, value_length) \
  (void)(value), (void)(value_length)
#define IREE_TRACE_SET_THREAD_NAME(name) (void)(name)

#define IREE_TRACE_FIBER_ENTER(fiber) (void)(fiber)
#define IREE_TRACE_FIBER_LEAVE()

#define IREE_TRACE_PUBLISH_SOURCE_FILE(filename, filename_length, content, \
                                       content_length)                     \
  (void)(filename), (void)(filename_length), (void)(content),              \
      (void)(content_length)

#define IREE_TRACE_ZONE_BEGIN(zone_id) \
  iree_zone_id_t zone_id = 0;          \
  (void)zone_id;
#define IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal) \
  IREE_TRACE_ZONE_BEGIN(zone_id)
#define IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name, name_length) \
  IREE_TRACE_ZONE_BEGIN(zone_id)
#define IREE_TRACE_ZONE_BEGIN_EXTERNAL(                        \
    zone_id, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)                   \
  IREE_TRACE_ZONE_BEGIN(zone_id)

#define IREE_TRACE_ZONE_END(zone_id) (void)(zone_id)

#define IREE_RETURN_AND_END_ZONE_IF_ERROR(zone_id, ...) \
  IREE_RETURN_IF_ERROR(__VA_ARGS__)
#define IREE_RETURN_AND_END_ZONE(zone_id, ...) return (__VA_ARGS__)

#define IREE_TRACE_ZONE_SET_COLOR(zone_id, color_xrgb) (void)(color_xrgb)
#define IREE_TRACE_ZONE_APPEND_VALUE_I64(zone_id, value) (void)(value)

// Same variadic dispatch console.h uses; /Zc:preprocessor makes it work on
// MSVC. Do not hand-roll this.
#define IREE_TRACE_ZONE_APPEND_TEXT(...)                                  \
  IREE_TRACE_IMPL_GET_VARIADIC_((__VA_ARGS__,                             \
                                 IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW, \
                                 IREE_TRACE_ZONE_APPEND_TEXT_CSTRING))    \
  (__VA_ARGS__)
#define IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(zone_id, value) (void)(value)
#define IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(zone_id, value, value_length) \
  (void)(value), (void)(value_length)

#define IREE_TRACE_SET_PLOT_TYPE(name_literal, plot_type, step, fill, color) \
  (void)(name_literal), (void)(plot_type), (void)(step), (void)(fill),       \
      (void)(color)
#define IREE_TRACE_PLOT_VALUE_I64(name_literal, value) \
  (void)(name_literal), (void)(value)
#define IREE_TRACE_PLOT_VALUE_F32(name_literal, value) \
  (void)(name_literal), (void)(value)
#define IREE_TRACE_PLOT_VALUE_F64(name_literal, value) \
  (void)(name_literal), (void)(value)

#define IREE_TRACE_FRAME_MARK()
#define IREE_TRACE_FRAME_MARK_NAMED(name_literal)
#define IREE_TRACE_FRAME_MARK_BEGIN_NAMED(name_literal)
#define IREE_TRACE_FRAME_MARK_END_NAMED(name_literal)

#define IREE_TRACE_MESSAGE(level, value_literal) (void)(value_literal)
#define IREE_TRACE_MESSAGE_COLORED(color, value_literal) \
  (void)(color), (void)(value_literal)
#define IREE_TRACE_MESSAGE_DYNAMIC(level, value, value_length) \
  (void)(value), (void)(value_length)
#define IREE_TRACE_MESSAGE_DYNAMIC_COLORED(color, value, value_length) \
  (void)(color), (void)(value), (void)(value_length)

// Utilities:
#define IREE_TRACE_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) NAME
#define IREE_TRACE_IMPL_GET_VARIADIC_(args) \
  IREE_TRACE_IMPL_GET_VARIADIC_HELPER_ args

#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

//===----------------------------------------------------------------------===//
// Instrumentation C++ RAII types, wrappers, and macros
//===----------------------------------------------------------------------===//

#ifdef __cplusplus

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

// Used from tools/ and tooling/comparison.cc; no-ops like the C zone macros.
#define IREE_TRACE_SCOPE()
#define IREE_TRACE_SCOPE_NAMED(name_literal)
#define IREE_TRACE_SCOPE_ID 0

#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

#endif  // __cplusplus

#endif  // IREE_BASE_TRACING_CSVGPU_H_
