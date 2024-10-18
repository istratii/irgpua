#pragma once

#include <cuda/atomic>
#include <raft/core/device_span.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include "cuda_tools/constants.hh"
#include "cuda_tools/cuda_error_checking.cuh"

// void reduce(rmm::device_uvector<int>& buffer, rmm::device_scalar<int>& total);
void reduce(raft::device_span<int> buffer_dspan,
            raft::device_span<int> total_dspan,
            cudaStream_t stream);