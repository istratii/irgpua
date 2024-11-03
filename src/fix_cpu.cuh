#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include "cuda_tools/nvtx.cuh"
#include "image.hh"

void fix_image_cpu(Image& to_fix);