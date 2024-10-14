#pragma once

#include <cuda/atomic>
#include <raft/core/device_span.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include "cuda_tools/cuda_error_checking.cuh"

void reduce(rmm::device_uvector<int>& buffer, rmm::device_scalar<int>& total);