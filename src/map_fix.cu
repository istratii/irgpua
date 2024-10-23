

#include "map_fix.cuh"

static __global__ void _map_fix(raft::device_span<int> buffer_dspan)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < buffer_dspan.size())
    buffer_dspan[idx] += ASSOC_VAL(idx);
}

void map_fix(raft::device_span<int> buffer_dspan, cudaStream_t stream)
{
  raft::common::nvtx::range fscope("map fix");

  constexpr unsigned int block_size = 1024;
  const unsigned int grid_size =
    (buffer_dspan.size() + block_size - 1) / block_size;

  _map_fix<<<grid_size, block_size, 0, stream>>>(buffer_dspan);
}
