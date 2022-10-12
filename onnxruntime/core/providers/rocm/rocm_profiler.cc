// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_ROCM) && defined(ENABLE_ROCM_PROFILING)

#include <chrono>
#include <time.h>

#include "core/common/profiler_common.h"
#include "core/providers/rocm/rocm_profiler.h"
#include "core/providers/rocm/RoctracerLogger.h"

#define BSIZE 4096

typedef uint64_t timestamp_t;
static timestamp_t timespec_to_ns(const timespec& time) {
  return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
}

namespace onnxruntime {

namespace profiling {

RocmProfiler::RocmProfiler()
: d(&RoctracerLogger::singleton())
{
}

RocmProfiler::~RocmProfiler()
{
}

bool RocmProfiler::StartProfiling()
{
    d->clearLogs();
    d->startLogging();
    return true;
}

void RocmProfiler::EndProfiling(TimePoint start_time, Events& events)
{
  d->stopLogging();

  std::map<uint64_t, std::vector<EventRecord>> event_map;
  std::map<uint64_t, kernelRow*> kernelLaunches;   // correlation id -> kernel info
  std::map<uint64_t, copyRow*> copyLaunches;     // correlation id -> copy info

  // Generate EventRecords
  int64_t profiling_start = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();

  // Wrong clock again - all the cool kids are doing it
  timespec t0, t1, t00;
  clock_gettime(CLOCK_REALTIME, &t0);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  clock_gettime(CLOCK_REALTIME, &t00);
  const uint64_t toffset = (timespec_to_ns(t0) >> 1) + (timespec_to_ns(t00) >> 1) - timespec_to_ns(t1);
  profiling_start = profiling_start - toffset;

  char buff[BSIZE];

  for (auto &item : d->rows_) {
    std::unordered_map<std::string, std::string> args {
      {"op_name", ""}
    };
    addEventRecord(item, profiling_start, std::move(args), event_map);
  }

  for (auto &item : d->mallocRows_) {
    snprintf(buff, BSIZE, "%p", item.ptr);
    const std::string arg_ptr{buff};

    std::unordered_map<std::string, std::string> args {
      {"op_name", ""},
      {"ptr", arg_ptr},
      {"size", std::to_string(item.size)}
    };
    addEventRecord(item, profiling_start, std::move(args), event_map);
  }

  for (auto &item : d->copyRows_) {
    snprintf(buff, BSIZE, "%p", item.stream);
    const std::string arg_stream{buff};
    snprintf(buff, BSIZE, "%p", item.src);
    const std::string arg_src{buff};
    snprintf(buff, BSIZE, "%p", item.dst);
    const std::string arg_dst{buff};

    std::unordered_map<std::string, std::string> args = {
      {"op_name", ""},
      {"stream", arg_stream},
      {"src", arg_src},
      {"dst", arg_dst},
      {"size", std::to_string(item.size)},
      {"kind", std::to_string(item.kind)},
    };
    addEventRecord(item, profiling_start, std::move(args), event_map);
    copyLaunches[item.id] = &item;
  }

  for (auto &item : d->kernelRows_) {
    snprintf(buff, BSIZE, "%p", item.stream);
    const std::string arg_stream{buff};
    if (item.functionAddr)
        snprintf(buff, BSIZE, "%s", demangle(hipKernelNameRefByPtr(item.functionAddr, item.stream)).c_str());
    else if (item.function)
        snprintf(buff, BSIZE, "%s", demangle(hipKernelNameRef(item.function)).c_str());
    else
        snprintf(buff, BSIZE, " ");
    const std::string arg_kernel{buff};

    std::unordered_map<std::string, std::string> args {
      {"op_name", ""},
      {"stream", arg_stream},
      {"kernel", arg_kernel},
      {"grid_x", std::to_string(item.gridX)},
      {"grid_y", std::to_string(item.gridY)},
      {"grid_z", std::to_string(item.gridZ)},
      {"block_x", std::to_string(item.workgroupX)},
      {"block_y", std::to_string(item.workgroupY)},
      {"block_z", std::to_string(item.workgroupZ)},
    };

    addEventRecord(item, profiling_start, std::move(args), event_map);
    kernelLaunches[item.id] = &item;
  }

  // Async Ops - e.g. "Kernel"

  for (auto& buffer : *d->gpuTraceBuffers_) {
    const roctracer_record_t* record = (const roctracer_record_t*)(buffer.data_);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(buffer.data_ + buffer.validSize_);
    while (record < end_record) {
      std::unordered_map<std::string, std::string> args;
      std::string name = roctracer_op_string(record->domain, record->op, record->kind);

      // Add kernel args if we have them
      auto kit = kernelLaunches.find(record->correlation_id);
      if (kit != kernelLaunches.end()) {
        auto &item = *(*kit).second;
        snprintf(buff, BSIZE, "%p", item.stream);
        args["stream"] = std::string(buff);
        args["grid_x"] = std::to_string(item.gridX);
        args["grid_y"] = std::to_string(item.gridY);
        args["grid_z"] = std::to_string(item.gridZ);
        args["block_x"] = std::to_string(item.workgroupX);
        args["block_y"] = std::to_string(item.workgroupY);
        args["block_z"] = std::to_string(item.workgroupZ);
        if (item.functionAddr != nullptr) {
          name = demangle(hipKernelNameRefByPtr(item.functionAddr, item.stream)).c_str();
        }
        else if (item.function != nullptr) {
          name = demangle(hipKernelNameRef(item.function)).c_str();
        }
      }

      // Add copy args if we have them
      auto cit = copyLaunches.find(record->correlation_id);
      if (cit != copyLaunches.end()) {
        auto &item = *(*cit).second;
        snprintf(buff, BSIZE, "%p", item.stream);
        args["stream"] = std::string(buff);
        snprintf(buff, BSIZE, "%p", item.src);
        args["dst"] = std::string(buff);
        snprintf(buff, BSIZE, "%p", item.dst);
        args["src"] = std::string(buff);
        args["size"] = std::to_string(item.size);
        args["kind"] = std::to_string(item.kind);
      }

      EventRecord event {
        onnxruntime::profiling::KERNEL_EVENT,
        static_cast<int>(record->device_id),
        static_cast<int>(record->queue_id),
        name,
        static_cast<int64_t>((record->begin_ns - profiling_start) / 1000),
        static_cast<int64_t>((record->end_ns - record->begin_ns) / 1000),
        std::move(args)
      };

      // FIXME: deal with missing ext correlation
      auto extId = d->externalCorrelations_[RoctracerLogger::CorrelationDomain::Default][record->correlation_id];
      event_map[extId].push_back(std::move(event));
      roctracer_next_record(record, &record);
    }
  }

  // General
  auto existing_events = std::move(events);
  events.clear();

  auto insert_iter = existing_events.begin();
  for (auto& [device_event_ts_u64, device_events] : event_map) {
    auto device_event_ts = static_cast<long long>(device_event_ts_u64);
    while (insert_iter != existing_events.end() && insert_iter->ts < device_event_ts) {
      events.push_back(std::move(*insert_iter));
      ++insert_iter;
    }
    if (insert_iter != existing_events.end() && insert_iter->ts == device_event_ts) {
      for (auto& device_event : device_events) {
        device_event.args["op_name"] = insert_iter->args["op_name"];
      }
    }
    events.insert(events.end(), device_events.begin(), device_events.end());
    while (insert_iter != existing_events.end() && insert_iter->cat == EventCategory::KERNEL_EVENT) {
      events.push_back(std::move(*insert_iter));
      ++insert_iter;
    }
  }

  // insert any remaining events into the merged event list
  while (insert_iter != existing_events.end()) {
    events.push_back(std::move(*insert_iter));
  }
}

void RocmProfiler::Start(uint64_t id)
{
  d->pushCorrelationID(id, RoctracerLogger::CorrelationDomain::Default);
}

void RocmProfiler::Stop(uint64_t)
{
  d->popCorrelationID(RoctracerLogger::CorrelationDomain::Default);
}


void RocmProfiler::addEventRecord(const roctracerRow &item,
                                  int64_t pstart,
                                  std::unordered_map<std::string, std::string>&& event_args,
                                  std::map<uint64_t, std::vector<EventRecord>>& event_map)
{
  EventRecord event {
    onnxruntime::profiling::API_EVENT,
    static_cast<int>(item.pid),
    static_cast<int>(item.tid),
    std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0)),
    static_cast<int64_t>((item.begin - pstart) / 1000),
    static_cast<int64_t>((item.end - item.begin) / 1000),
    std::move(event_args)
  };

  // FIXME: deal with missing ext correlation
  auto extId = d->externalCorrelations_[RoctracerLogger::CorrelationDomain::Default][item.id];
  event_map[extId].push_back(std::move(event));
}

}  // namespace profiling
}  // namespace onnxruntime
#endif
