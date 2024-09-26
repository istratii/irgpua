#pragma once

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include "map_fix.cuh"
#include "compact.cuh"
#include "cuda_tools/cuda_error_checking.cuh"
#include "histogram.cuh"
#include "image.hh"
#include "scan.cuh"

void fix_image_gpu(Image& to_fix);