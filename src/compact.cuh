#pragma once

#include <cuda_runtime.h>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include "cuda_tools/constants.hh"
#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/nvtx.cuh"
#include "scan.cuh"

void compact(rmm::device_buffer& memchunk, raft::device_span<int> buffer_dspan);