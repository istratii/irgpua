#pragma once

#include <cuda/atomic>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <cub/cub.cuh>
#include <raft/core/device_span.hpp>
#include <rmm/device_scalar.hpp>

#include "cuda_tools/constants.hh"
#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/nvtx.cuh"
#include "scan.cuh"

void equalize_histogram(rmm::device_buffer& memchunk,
                        raft::device_span<int> buffer_dspan);