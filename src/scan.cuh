#pragma once

#include <cuda/atomic>
#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>

#include "cuda_tools/cuda_error_checking.cuh"
#include "image.hh"

enum ScanMode
{
  SCAN_INCLUSIVE,
  SCAN_EXCLUSIVE,
};

void scan(rmm::device_uvector<int>& buffer, ScanMode mode);