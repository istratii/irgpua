#pragma once

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include "cuda_tools/cuda_error_checking.cuh"

#define ASSOC_VAL(idx)                                                         \
  (((idx) % 4 == 0) ? 1 : ((idx) % 4 == 1) ? -5 : ((idx) % 4 == 2) ? 3 : -8)

void map_fix(rmm::device_uvector<int>& buffer);