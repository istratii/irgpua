

#include "map_fix.cuh"

static __global__ void _map_fix(raft::device_span<int> d_buffer)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < d_buffer.size())
    d_buffer[idx] += ASSOC_VAL(idx);
}

void map_fix(rmm::device_uvector<int>& buffer)
{
  constexpr unsigned int block_size = 1024;
  const unsigned int grid_size = (buffer.size() + block_size - 1) / block_size;

  _map_fix<<<grid_size, block_size, 0, buffer.stream()>>>(
    raft::device_span<int>(buffer.data(), buffer.size()));
}
