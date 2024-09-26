#pragma once

#include <cuda/atomic>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include "cuda_tools/cuda_error_checking.cuh"
#include "scan.cuh"

rmm::device_uvector<int> histogram(rmm::device_uvector<int>& buffer);
void equalize_histogram(rmm::device_uvector<int>& buffer,
                        rmm::device_uvector<int>& hist);