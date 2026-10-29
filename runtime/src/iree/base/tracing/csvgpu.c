// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if defined(_MSC_VER)
#pragma warning(disable : 4996)  // getenv/fopen
#endif  // _MSC_VER

#include <stdio.h>
#include <string.h>

#include "iree/base/internal/time.h"
#include "iree/base/tracing.h"

// Guarded on the same bit the header uses to declare these, so that a build
// with device instrumentation off still links (it simply records nothing).
#if IREE_TRACING_FEATURES

#if defined(_WIN32)
#include <windows.h>
typedef SRWLOCK csvgpu_mutex_t;
#define CSVGPU_MUTEX_INIT SRWLOCK_INIT
#define csvgpu_lock(m) AcquireSRWLockExclusive(m)
#define csvgpu_unlock(m) ReleaseSRWLockExclusive(m)
#else
#include <pthread.h>
typedef pthread_mutex_t csvgpu_mutex_t;
#define CSVGPU_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define csvgpu_lock(m) pthread_mutex_lock(m)
#define csvgpu_unlock(m) pthread_mutex_unlock(m)
#endif  // _WIN32

// The Vulkan HAL's query pool holds exactly 32*1024 timestamps and hands back
// ids modulo that capacity, so a slot table of the same size is a 1:1 map and
// the uint16_t query id can never alias.
#define CSVGPU_SLOT_COUNT (32 * 1024)
#define CSVGPU_MAX_CONTEXTS 64
#define CSVGPU_MAX_DEPTH 64
#define CSVGPU_RECORD_COUNT (16 * 1024)
#define CSVGPU_NAME_SLOTS (16 * 1024)  // power of two; open addressed
#define CSVGPU_NAME_ARENA (4 * 1024 * 1024)

enum {
  CSVGPU_KIND_TRANSFER = 0,   // fill/copy/update: the non-external zone path
  CSVGPU_KIND_CMDBUF = 1,     // the command buffer's own enclosing zone
  CSVGPU_KIND_DEBUG_GROUP = 2,
  CSVGPU_KIND_DISPATCH = 3,
};

static const char* const csvgpu_kind_names[] = {"transfer", "cmdbuf",
                                                "debug_group", "dispatch"};

enum {
  CSVGPU_ROLE_NONE = 0,
  CSVGPU_ROLE_BEGIN = 1,
  CSVGPU_ROLE_END = 2,
};

typedef struct {
  int32_t next_free;  // -1 when live
  uint8_t context_id;
  uint8_t kind;
  uint8_t closed;      // the end zone has been seen
  uint8_t have_begin;  // begin timestamp has arrived
  uint8_t have_end;    // end timestamp has arrived
  uint8_t had_child;   // a zone opened while this was on the stack
  uint32_t name_id;
  uint32_t file_id;
  uint32_t line;
  uint32_t cmdbuf_seq;
  uint32_t ordinal;
  uint64_t global_seq;
  uint64_t begin_ts;
  uint64_t end_ts;
} csvgpu_record_t;

typedef struct {
  char name[64];
  float timestamp_period;
  uint32_t cmdbuf_seq;
  uint32_t ordinal;
  int32_t depth;
  int32_t stack[CSVGPU_MAX_DEPTH];
  int32_t slot_record[CSVGPU_SLOT_COUNT];
  uint8_t slot_role[CSVGPU_SLOT_COUNT];
} csvgpu_context_t;

typedef struct {
  uint64_t global_seq;
  uint64_t begin_ts;
  uint64_t end_ts;
  double duration_ns;
  uint32_t name_id;
  uint32_t cmdbuf_seq;
  uint32_t ordinal;
  uint8_t context_id;
  uint8_t kind;
} csvgpu_event_t;

typedef struct {
  uint64_t count;
  double sum_ns;
  double min_ns;
  double max_ns;
} csvgpu_aggregate_t;

static csvgpu_mutex_t csvgpu_mutex = CSVGPU_MUTEX_INIT;

static bool csvgpu_initialized = false;
static bool csvgpu_written = false;
static const char* csvgpu_events_path = NULL;
static bool csvgpu_include_transfers = false;

static csvgpu_context_t* csvgpu_contexts[CSVGPU_MAX_CONTEXTS];
static int32_t csvgpu_context_count = 0;

static csvgpu_record_t* csvgpu_records = NULL;
static int32_t csvgpu_record_free = -1;
static uint64_t csvgpu_records_live = 0;

static csvgpu_event_t* csvgpu_events = NULL;
static size_t csvgpu_event_count = 0;
static size_t csvgpu_event_capacity = 0;
static uint64_t csvgpu_next_global_seq = 0;

// Interned zone names. Names are export symbols, so there are far fewer of them
// than there are dispatches and copying each event's name would dwarf the rest.
static char* csvgpu_name_arena = NULL;
static size_t csvgpu_name_arena_used = 0;
static uint32_t csvgpu_name_offsets[CSVGPU_NAME_SLOTS];
static int32_t csvgpu_name_table[CSVGPU_NAME_SLOTS];
static uint32_t csvgpu_name_count = 0;
static csvgpu_aggregate_t* csvgpu_aggregates = NULL;

// Loss counters. All of these are silent failures otherwise, and a cost table
// built from a lossy capture is wrong in a way that looks plausible, so they
// are reported in the CSV header.
static uint64_t csvgpu_dropped_overrun = 0;
static uint64_t csvgpu_dropped_backwards = 0;
static uint64_t csvgpu_dropped_no_record = 0;
static uint64_t csvgpu_orphan_notify = 0;
static uint64_t csvgpu_unbalanced_end = 0;

//===----------------------------------------------------------------------===//
// Setup
//===----------------------------------------------------------------------===//

static void csvgpu_write_output(void);
static uint32_t csvgpu_intern(const char* name, size_t length);

static void csvgpu_atexit(void) { iree_tracing_csvgpu_deinitialize(); }

// Caller must hold the lock (or be before any concurrency, as APP_ENTER is).
static void csvgpu_ensure_initialized(void) {
  if (csvgpu_initialized) return;
  csvgpu_initialized = true;

  const char* path = getenv("IREE_GPU_CSV");
  csvgpu_events_path = path ? path : "iree_gpu_dispatches.csv";
  if (csvgpu_events_path[0] == '\0') csvgpu_events_path = NULL;

  const char* transfers = getenv("IREE_GPU_CSV_TRANSFERS");
  csvgpu_include_transfers = transfers && transfers[0] == '1';

  csvgpu_records =
      (csvgpu_record_t*)calloc(CSVGPU_RECORD_COUNT, sizeof(csvgpu_record_t));
  csvgpu_name_arena = (char*)malloc(CSVGPU_NAME_ARENA);
  csvgpu_aggregates =
      (csvgpu_aggregate_t*)calloc(CSVGPU_NAME_SLOTS, sizeof(csvgpu_aggregate_t));
  if (!csvgpu_records || !csvgpu_name_arena || !csvgpu_aggregates) {
    fprintf(stderr, "csvgpu: out of memory during initialization\n");
    csvgpu_events_path = NULL;
    return;
  }

  for (int32_t i = 0; i < CSVGPU_RECORD_COUNT; ++i) {
    csvgpu_records[i].next_free = i + 1;
  }
  csvgpu_records[CSVGPU_RECORD_COUNT - 1].next_free = -1;
  csvgpu_record_free = 0;

  for (int32_t i = 0; i < CSVGPU_NAME_SLOTS; ++i) csvgpu_name_table[i] = -1;
  // Reserve id 0 for the empty name so that an interning failure, which also
  // returns 0, degrades to a blank name rather than aliasing a real one.
  csvgpu_intern("", 0);

  // Library embedders never call IREE_TRACE_APP_EXIT; without this the capture
  // is silently discarded at process exit.
  atexit(csvgpu_atexit);
}

void iree_tracing_csvgpu_initialize(void) {
  csvgpu_lock(&csvgpu_mutex);
  csvgpu_ensure_initialized();
  csvgpu_unlock(&csvgpu_mutex);
}

void iree_tracing_csvgpu_deinitialize(void) {
  csvgpu_lock(&csvgpu_mutex);
  if (!csvgpu_written) {
    csvgpu_written = true;
    csvgpu_write_output();
  }
  csvgpu_unlock(&csvgpu_mutex);
}

//===----------------------------------------------------------------------===//
// Clock
//===----------------------------------------------------------------------===//

// iree_platform_time_now() is in nanoseconds, so reporting a frequency of 1e9
// makes vulkan/tracing.cc's `cpu_time *= 1e9 / frequency` scaling a no-op.
// Returning zero here would divide by zero there.
int64_t iree_tracing_time(void) { return iree_platform_time_now(); }
int64_t iree_tracing_frequency(void) { return 1000000000ll; }

//===----------------------------------------------------------------------===//
// Name interning
//===----------------------------------------------------------------------===//

static uint32_t csvgpu_intern(const char* name, size_t length) {
  if (!name || !length) length = 0;

  uint64_t hash = 1469598103934665603ull;  // FNV-1a
  for (size_t i = 0; i < length; ++i) {
    hash ^= (uint8_t)name[i];
    hash *= 1099511628211ull;
  }

  uint32_t mask = CSVGPU_NAME_SLOTS - 1;
  uint32_t slot = (uint32_t)hash & mask;
  for (uint32_t probe = 0; probe <= mask; ++probe) {
    int32_t id = csvgpu_name_table[slot];
    if (id < 0) {
      if (csvgpu_name_count >= CSVGPU_NAME_SLOTS / 2 ||
          csvgpu_name_arena_used + length + 1 > CSVGPU_NAME_ARENA) {
        return 0;  // full; falls back to the empty name
      }
      uint32_t offset = (uint32_t)csvgpu_name_arena_used;
      memcpy(csvgpu_name_arena + offset, name, length);
      csvgpu_name_arena[offset + length] = '\0';
      csvgpu_name_arena_used += length + 1;

      uint32_t new_id = csvgpu_name_count++;
      csvgpu_name_offsets[new_id] = offset;
      csvgpu_name_table[slot] = (int32_t)new_id;
      return new_id;
    }
    const char* existing = csvgpu_name_arena + csvgpu_name_offsets[id];
    if (strlen(existing) == length && memcmp(existing, name, length) == 0) {
      return (uint32_t)id;
    }
    slot = (slot + 1) & mask;
  }
  return 0;
}

static const char* csvgpu_name(uint32_t name_id) {
  if (name_id >= csvgpu_name_count) return "";
  return csvgpu_name_arena + csvgpu_name_offsets[name_id];
}

//===----------------------------------------------------------------------===//
// Records
//===----------------------------------------------------------------------===//

static int32_t csvgpu_record_acquire(void) {
  int32_t index = csvgpu_record_free;
  if (index < 0) {
    ++csvgpu_dropped_no_record;
    return -1;
  }
  csvgpu_record_free = csvgpu_records[index].next_free;
  csvgpu_record_t* record = &csvgpu_records[index];
  memset(record, 0, sizeof(*record));
  record->next_free = -1;
  ++csvgpu_records_live;
  return index;
}

static void csvgpu_record_release(int32_t index) {
  csvgpu_records[index].next_free = csvgpu_record_free;
  csvgpu_record_free = index;
  --csvgpu_records_live;
}

static void csvgpu_event_append(const csvgpu_record_t* record,
                                double duration_ns) {
  if (csvgpu_event_count == csvgpu_event_capacity) {
    size_t capacity = csvgpu_event_capacity ? csvgpu_event_capacity * 2 : 4096;
    csvgpu_event_t* grown = (csvgpu_event_t*)realloc(
        csvgpu_events, capacity * sizeof(csvgpu_event_t));
    if (!grown) return;  // reported as a shortfall against the dispatch count
    csvgpu_events = grown;
    csvgpu_event_capacity = capacity;
  }
  csvgpu_event_t* event = &csvgpu_events[csvgpu_event_count++];
  event->global_seq = record->global_seq;
  event->begin_ts = record->begin_ts;
  event->end_ts = record->end_ts;
  event->duration_ns = duration_ns;
  event->name_id = record->name_id;
  event->cmdbuf_seq = record->cmdbuf_seq;
  event->ordinal = record->ordinal;
  event->context_id = record->context_id;
  event->kind = record->kind;
}

// Completes a record once its end zone and both timestamps have arrived. The
// three can arrive in any order: collect() reads the query pool in chunks that
// wrap, so a notify may land long after the zone closed, or before it.
static void csvgpu_try_finalize(int32_t index) {
  csvgpu_record_t* record = &csvgpu_records[index];
  if (!record->closed || !record->have_begin || !record->have_end) return;

  if (record->end_ts < record->begin_ts) {
    // A wrapped device counter; some drivers expose fewer than 64 valid bits.
    ++csvgpu_dropped_backwards;
    csvgpu_record_release(index);
    return;
  }

  csvgpu_context_t* context = csvgpu_contexts[record->context_id];
  double period = context ? (double)context->timestamp_period : 1.0;
  double duration_ns = (double)(record->end_ts - record->begin_ts) * period;

  if (record->kind == CSVGPU_KIND_DISPATCH || csvgpu_include_transfers) {
    csvgpu_event_append(record, duration_ns);
    if (record->kind == CSVGPU_KIND_DISPATCH &&
        record->name_id < CSVGPU_NAME_SLOTS) {
      csvgpu_aggregate_t* aggregate = &csvgpu_aggregates[record->name_id];
      if (!aggregate->count || duration_ns < aggregate->min_ns) {
        aggregate->min_ns = duration_ns;
      }
      if (!aggregate->count || duration_ns > aggregate->max_ns) {
        aggregate->max_ns = duration_ns;
      }
      ++aggregate->count;
      aggregate->sum_ns += duration_ns;
    }
  }

  csvgpu_record_release(index);
}

//===----------------------------------------------------------------------===//
// GPU zone callbacks
//===----------------------------------------------------------------------===//

uint8_t iree_tracing_gpu_context_allocate(iree_tracing_gpu_context_type_t type,
                                          const char* name, size_t name_length,
                                          bool is_calibrated,
                                          uint64_t cpu_timestamp,
                                          uint64_t gpu_timestamp,
                                          float timestamp_period) {
  (void)type;
  // Calibration is irrelevant here: durations are differences of two device
  // timestamps, so we never leave the GPU clock domain.
  (void)is_calibrated;
  (void)cpu_timestamp;
  (void)gpu_timestamp;

  csvgpu_lock(&csvgpu_mutex);
  csvgpu_ensure_initialized();

  if (csvgpu_context_count >= CSVGPU_MAX_CONTEXTS) {
    csvgpu_unlock(&csvgpu_mutex);
    return 0;
  }
  csvgpu_context_t* context =
      (csvgpu_context_t*)malloc(sizeof(csvgpu_context_t));
  if (!context) {
    csvgpu_unlock(&csvgpu_mutex);
    return 0;
  }
  memset(context, 0, sizeof(*context));
  // The caller passes a stack buffer; this must be copied.
  size_t copy_length = name_length < sizeof(context->name) - 1
                           ? name_length
                           : sizeof(context->name) - 1;
  if (name && copy_length) memcpy(context->name, name, copy_length);
  context->timestamp_period = timestamp_period > 0.0f ? timestamp_period : 1.0f;
  for (int32_t i = 0; i < CSVGPU_SLOT_COUNT; ++i) context->slot_record[i] = -1;

  uint8_t context_id = (uint8_t)csvgpu_context_count++;
  csvgpu_contexts[context_id] = context;
  csvgpu_unlock(&csvgpu_mutex);
  return context_id;
}

void iree_tracing_gpu_context_calibrate(uint8_t context_id, int64_t cpu_delta,
                                        int64_t cpu_timestamp,
                                        int64_t gpu_timestamp) {
  (void)context_id;
  (void)cpu_delta;
  (void)cpu_timestamp;
  (void)gpu_timestamp;
}

// Shared by both begin flavours; caller holds the lock.
static void csvgpu_zone_begin(uint8_t context_id, uint16_t query_id,
                              uint8_t kind, uint32_t name_id, uint32_t file_id,
                              uint32_t line) {
  csvgpu_context_t* context = csvgpu_contexts[context_id];
  if (!context) return;

  // Context bookkeeping happens whether or not a record is available, so the
  // ordinals stay aligned with the real dispatch sequence.
  if (kind == CSVGPU_KIND_CMDBUF) {
    ++context->cmdbuf_seq;
    context->ordinal = 0;
  }
  uint32_t ordinal = 0;
  uint64_t global_seq = 0;
  if (kind == CSVGPU_KIND_DISPATCH) {
    ordinal = context->ordinal++;
    // Assigned at record time, which is program order -- the order the split
    // point search walks @main's top-level dispatches in.
    global_seq = csvgpu_next_global_seq++;
  }

  if (query_id < CSVGPU_SLOT_COUNT && context->slot_record[query_id] >= 0) {
    // IREE lapped its own query ring, so this timestamp is about to overwrite a
    // pending zone's (its assert for that is compiled out in release builds).
    // The stale record is orphaned rather than released: it may still be on the
    // stack, and reusing its slot would corrupt a live zone. It shows up in
    // unresolved_at_exit.
    ++csvgpu_dropped_overrun;
    context->slot_record[query_id] = -1;
  }

  int32_t index = csvgpu_record_acquire();
  if (index >= 0) {
    csvgpu_record_t* record = &csvgpu_records[index];
    record->context_id = context_id;
    record->kind = kind;
    record->name_id = name_id;
    record->file_id = file_id;
    record->line = line;
    record->cmdbuf_seq = context->cmdbuf_seq;
    record->ordinal = ordinal;
    record->global_seq = global_seq;

    if (query_id < CSVGPU_SLOT_COUNT) {
      context->slot_record[query_id] = index;
      context->slot_role[query_id] = CSVGPU_ROLE_BEGIN;
    }
    if (context->depth > 0 && context->depth <= CSVGPU_MAX_DEPTH) {
      int32_t parent = context->stack[context->depth - 1];
      if (parent >= 0) csvgpu_records[parent].had_child = 1;
    }
  }

  // Push unconditionally -- a placeholder for a zone we could not record still
  // has to balance its end, or every later zone pops the wrong entry.
  if (context->depth < CSVGPU_MAX_DEPTH) context->stack[context->depth] = index;
  ++context->depth;
}

void iree_tracing_gpu_zone_begin(uint8_t context_id, uint16_t query_id,
                                 const iree_tracing_location_t* src_loc) {
  // Only fills, copies and updates take this path; dispatches always come
  // through the external variant. Classification is therefore structural and
  // does not depend on matching any string.
  (void)src_loc;
  csvgpu_lock(&csvgpu_mutex);
  csvgpu_zone_begin(context_id, query_id, CSVGPU_KIND_TRANSFER, 0, 0, 0);
  csvgpu_unlock(&csvgpu_mutex);
}

void iree_tracing_gpu_zone_begin_external(
    uint8_t context_id, uint16_t query_id, const char* file_name,
    size_t file_name_length, uint32_t line, const char* function_name,
    size_t function_name_length, const char* name, size_t name_length) {
  // Three callers reach here, distinguishable without string matching except
  // for the command buffer scope:
  //   command_buffer_begin  func = "iree_hal_vulkan_direct_command_buffer",
  //                         file empty, name empty
  //   begin_debug_group     func empty, name = the label
  //   dispatch              func = the export name, name empty
  static const char kCommandBuffer[] = "iree_hal_vulkan_direct_command_buffer";
  uint8_t kind;
  if (function_name_length == sizeof(kCommandBuffer) - 1 && function_name &&
      memcmp(function_name, kCommandBuffer, sizeof(kCommandBuffer) - 1) == 0) {
    kind = CSVGPU_KIND_CMDBUF;
  } else if (!function_name_length || name_length) {
    kind = CSVGPU_KIND_DEBUG_GROUP;
  } else {
    kind = CSVGPU_KIND_DISPATCH;
  }

  csvgpu_lock(&csvgpu_mutex);
  csvgpu_ensure_initialized();
  uint32_t name_id = 0;
  uint32_t file_id = 0;
  if (kind == CSVGPU_KIND_DISPATCH) {
    name_id = csvgpu_intern(function_name, function_name_length);
    file_id = csvgpu_intern(file_name, file_name_length);
  } else if (kind == CSVGPU_KIND_DEBUG_GROUP) {
    name_id = csvgpu_intern(name, name_length);
  }
  csvgpu_zone_begin(context_id, query_id, kind, name_id, file_id, line);
  csvgpu_unlock(&csvgpu_mutex);
}

void iree_tracing_gpu_zone_end(uint8_t context_id, uint16_t query_id) {
  csvgpu_lock(&csvgpu_mutex);
  csvgpu_context_t* context = csvgpu_contexts[context_id];
  if (!context) {
    csvgpu_unlock(&csvgpu_mutex);
    return;
  }
  if (context->depth <= 0) {
    ++csvgpu_unbalanced_end;
    csvgpu_unlock(&csvgpu_mutex);
    return;
  }
  --context->depth;
  int32_t index =
      context->depth < CSVGPU_MAX_DEPTH ? context->stack[context->depth] : -1;
  if (index < 0 || query_id >= CSVGPU_SLOT_COUNT) {
    csvgpu_unlock(&csvgpu_mutex);
    return;
  }

  csvgpu_records[index].closed = 1;
  context->slot_record[query_id] = index;
  context->slot_role[query_id] = CSVGPU_ROLE_END;
  csvgpu_try_finalize(index);
  csvgpu_unlock(&csvgpu_mutex);
}

void iree_tracing_gpu_zone_notify(uint8_t context_id, uint16_t query_id,
                                  int64_t gpu_timestamp) {
  csvgpu_lock(&csvgpu_mutex);
  csvgpu_context_t* context = csvgpu_contexts[context_id];
  if (!context || query_id >= CSVGPU_SLOT_COUNT) {
    csvgpu_unlock(&csvgpu_mutex);
    return;
  }
  int32_t index = context->slot_record[query_id];
  if (index < 0) {
    // Expected exactly once per context: the initial calibration writes query 0
    // without ever opening a zone around it.
    ++csvgpu_orphan_notify;
    csvgpu_unlock(&csvgpu_mutex);
    return;
  }

  csvgpu_record_t* record = &csvgpu_records[index];
  if (context->slot_role[query_id] == CSVGPU_ROLE_BEGIN) {
    record->begin_ts = (uint64_t)gpu_timestamp;
    record->have_begin = 1;
  } else {
    record->end_ts = (uint64_t)gpu_timestamp;
    record->have_end = 1;
  }
  context->slot_record[query_id] = -1;
  context->slot_role[query_id] = CSVGPU_ROLE_NONE;

  csvgpu_try_finalize(index);
  csvgpu_unlock(&csvgpu_mutex);
}

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

static void csvgpu_summary_path(char* buffer, size_t capacity) {
  size_t length = strlen(csvgpu_events_path);
  const char* suffix = ".summary.csv";
  // Replace a trailing .csv rather than appending to it.
  if (length > 4 && strcmp(csvgpu_events_path + length - 4, ".csv") == 0) {
    length -= 4;
  }
  if (length + strlen(suffix) + 1 > capacity) length = 0;
  memcpy(buffer, csvgpu_events_path, length);
  memcpy(buffer + length, suffix, strlen(suffix) + 1);
}

// Zones still waiting on an end or a timestamp when the process exits. Vulkan's
// tracing context always collects once before tearing down its query pool, so
// a non-zero count here means the capture is genuinely incomplete.
static uint64_t csvgpu_unresolved(void) { return csvgpu_records_live; }

static void csvgpu_write_header(FILE* file) {
  fprintf(file, "# iree-gpu-csv v1\n");
  fprintf(file,
          "# durations are device timestamp deltas; a timestamp costs about as "
          "much as a\n"
          "# full execution barrier, so these are an upper bound and "
          "overlapping dispatches\n"
          "# are serialised. Relative cost is trustworthy; absolute cost is "
          "not.\n");
  for (int32_t i = 0; i < csvgpu_context_count; ++i) {
    csvgpu_context_t* context = csvgpu_contexts[i];
    if (!context) continue;
    fprintf(file, "# context=%d queue=\"%s\" timestamp_period_ns=%f\n", i,
            context->name, (double)context->timestamp_period);
  }
  fprintf(file,
          "# dispatch_events=%llu dropped_overrun=%llu dropped_backwards=%llu "
          "dropped_no_record=%llu orphan_notify=%llu unbalanced_end=%llu "
          "unresolved_at_exit=%llu\n",
          (unsigned long long)csvgpu_event_count,
          (unsigned long long)csvgpu_dropped_overrun,
          (unsigned long long)csvgpu_dropped_backwards,
          (unsigned long long)csvgpu_dropped_no_record,
          (unsigned long long)csvgpu_orphan_notify,
          (unsigned long long)csvgpu_unbalanced_end,
          (unsigned long long)csvgpu_unresolved());
}

static void csvgpu_write_output(void) {
  if (!csvgpu_events_path) return;

  FILE* file = fopen(csvgpu_events_path, "w");
  if (!file) {
    fprintf(stderr, "csvgpu: could not open '%s' for writing\n",
            csvgpu_events_path);
    return;
  }
  csvgpu_write_header(file);
  fprintf(file,
          "global_seq,context,cmdbuf_seq,ordinal,kind,export_name,begin_ticks,"
          "end_ticks,duration_ns\n");
  for (size_t i = 0; i < csvgpu_event_count; ++i) {
    const csvgpu_event_t* event = &csvgpu_events[i];
    fprintf(file, "%llu,%u,%u,%u,%s,%s,%llu,%llu,%.1f\n",
            (unsigned long long)event->global_seq, (unsigned)event->context_id,
            (unsigned)event->cmdbuf_seq, (unsigned)event->ordinal,
            csvgpu_kind_names[event->kind], csvgpu_name(event->name_id),
            (unsigned long long)event->begin_ts,
            (unsigned long long)event->end_ts, event->duration_ns);
  }
  fclose(file);

  char summary_path[1024];
  csvgpu_summary_path(summary_path, sizeof(summary_path));
  file = fopen(summary_path, "w");
  if (!file) return;
  csvgpu_write_header(file);
  fprintf(file, "export_name,count,sum_ns,mean_ns,min_ns,max_ns\n");
  for (uint32_t i = 0; i < csvgpu_name_count; ++i) {
    const csvgpu_aggregate_t* aggregate = &csvgpu_aggregates[i];
    if (!aggregate->count) continue;
    fprintf(file, "%s,%llu,%.1f,%.1f,%.1f,%.1f\n", csvgpu_name(i),
            (unsigned long long)aggregate->count, aggregate->sum_ns,
            aggregate->sum_ns / (double)aggregate->count, aggregate->min_ns,
            aggregate->max_ns);
  }
  fclose(file);

  fprintf(stderr, "csvgpu: wrote %llu dispatch events to %s\n",
          (unsigned long long)csvgpu_event_count, csvgpu_events_path);
}

#endif  // IREE_TRACING_FEATURES
