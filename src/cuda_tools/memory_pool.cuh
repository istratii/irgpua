#pragma once

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#define ALIGN32(N) ((N + 31) & ~31)

void init_memory_pool(size_t bytes);
void free_memory_pool();