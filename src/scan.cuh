#pragma once

#include <cuda/atomic>
#include <raft/core/device_span.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include "cuda_tools/constants.hh"
#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/nvtx.cuh"
#include "image.hh"

enum ScanMode
{
  SCAN_INCLUSIVE,
  SCAN_EXCLUSIVE,
};

void scan(rmm::device_buffer& memchunk,
          raft::device_span<int> buffer_dspan,
          ScanMode mode);