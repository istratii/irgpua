
#include "memory_pool.cuh"

static std::shared_ptr<rmm::mr::cuda_memory_resource> device_memory_resource;
static std::shared_ptr<
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>
  device_pool_resource;
static std::shared_ptr<rmm::mr::pinned_memory_resource>
  host_pinned_memory_resource;
static std::shared_ptr<
  rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>>
  host_pinned_pool_resource;

void init_device_memory_pool(size_t bytes)
{
  bytes = ALIGN32(bytes);

  if (!device_memory_resource)
    device_memory_resource = std::make_shared<rmm::mr::cuda_memory_resource>();

  if (!device_pool_resource)
    device_pool_resource = std::make_shared<
      rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
      device_memory_resource.get(), bytes);

  rmm::mr::set_current_device_resource(device_pool_resource.get());
}

void free_device_memory_pool()
{
  rmm::mr::set_current_device_resource(nullptr);
  device_pool_resource.reset();
  device_memory_resource.reset();
}

void init_host_pinned_memory_pool(size_t bytes)
{
  bytes = ALIGN32(bytes);

  if (!host_pinned_memory_resource)
    host_pinned_memory_resource =
      std::make_shared<rmm::mr::pinned_memory_resource>();

  if (!host_pinned_pool_resource)
    host_pinned_pool_resource = std::make_shared<
      rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>>(
      host_pinned_memory_resource.get(), bytes, bytes);
}

void* allocate_host_pinned_memory(size_t bytes)
{
  return host_pinned_pool_resource->allocate(bytes);
}

void free_host_pinned_memory(void* ptr, size_t bytes)
{
  host_pinned_pool_resource->deallocate(ptr, bytes);
}

void free_host_pinned_memory_pool()
{
  host_pinned_pool_resource.reset();
  host_pinned_memory_resource.reset();
}