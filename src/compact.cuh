#pragma once

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <raft/core/device_span.hpp>
#include <rmm/device_buffer.hpp>

#include "cuda_tools/constants.hh"
#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/nvtx.cuh"
#include "scan.cuh"

void compact(rmm::device_buffer& memchunk, raft::device_span<int> buffer_dspan);