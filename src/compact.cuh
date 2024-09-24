#pragma once

#include <cuda_runtime.h>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include "cuda_tools/cuda_error_checking.cuh"
#include "scan.cuh"

void compact(rmm::device_uvector<int>& buffer);