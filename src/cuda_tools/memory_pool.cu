
#include "memory_pool.cuh"

void init_memory_pool(size_t bytes)
{
  bytes = ALIGN32(bytes);
  auto cuda_memory_resource = std::make_shared<rmm::mr::cuda_memory_resource>();
  auto pool_ressource = std::make_shared<
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
    cuda_memory_resource.get(), bytes);
  rmm::mr::set_current_device_resource(pool_ressource.get());
}