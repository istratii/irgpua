
#include "memory_pool.cuh"

static std::shared_ptr<rmm::mr::cuda_memory_resource> cuda_memory_resource;
static std::shared_ptr<
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>
  pool_resource;

void init_memory_pool(size_t bytes)
{
  bytes = ALIGN32(bytes);

  if (!cuda_memory_resource)
    cuda_memory_resource = std::make_shared<rmm::mr::cuda_memory_resource>();

  if (!pool_resource)
    pool_resource = std::make_shared<
      rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
      cuda_memory_resource.get(), bytes);

  rmm::mr::set_current_device_resource(pool_resource.get());
}