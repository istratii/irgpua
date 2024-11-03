#pragma once

#include <raft/core/device_span.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>

#include "compact.cuh"
#include "cuda_tools/constants.hh"
#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/nvtx.cuh"
#include "histogram.cuh"
#include "image.hh"
#include "map_fix.cuh"
#include "reduce.cuh"
#include "scan.cuh"

void fix_image_gpu(Image& to_fix, cudaStream_t stream);