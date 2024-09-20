#pragma once

#include <cuda/atomic>
#include <rmm/device_uvector.hpp>

#include "image.hh"

void scan(rmm::device_uvector<int>& buffer, bool inclusive);