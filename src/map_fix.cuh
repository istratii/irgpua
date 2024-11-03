#pragma once

#include <thrust/transform.h>
#include <raft/core/device_span.hpp>

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/nvtx.cuh"

#define ASSOC_VAL(idx)                                                         \
  (((idx) % 4 == 0) ? 1 : ((idx) % 4 == 1) ? -5 : ((idx) % 4 == 2) ? 3 : -8)

void map_fix(raft::device_span<int> buffer_span, cudaStream_t stream);