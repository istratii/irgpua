#pragma once

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#define ALIGN32(N) ((N + 31) & ~31)

void init_device_memory_pool(size_t bytes);
void free_device_memory_pool();
void init_host_pinned_memory_pool(size_t bytes);
void* allocate_host_pinned_memory(size_t bytes);
void free_host_pinned_memory(void* ptr, size_t bytes);
void free_host_pinned_memory_pool();