#pragma once

#include <cuda/atomic>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include "cuda_tools/cuda_error_checking.cuh"

rmm::device_buffer histogram(rmm::device_uvector<int>& buffer);