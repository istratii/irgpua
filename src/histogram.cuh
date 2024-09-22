#pragma once

#include <cuda/atomic>
#include <rmm/device_uvector.hpp>

void histogram(rmm::device_uvector<int>& buffer);