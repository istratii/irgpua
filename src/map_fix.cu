

#include "map_fix.cuh"

static __global__ void _map_fix(raft::device_span<int> d_buffer)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < d_buffer.size())
    d_buffer[idx] += ASSOC_VAL(idx);
}

void map_fix(rmm::device_uvector<int>& buffer)
{
#define THREADS_PER_BLOCK 1024

  _map_fix<<<(buffer.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
             THREADS_PER_BLOCK, 0, buffer.stream()>>>(
    raft::device_span<int>(buffer.data(), buffer.size()));

#undef THREADS_PER_BLOCK
}
