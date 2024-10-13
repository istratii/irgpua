#include "reduce.cuh"

static __global__ void _reduce(raft::device_span<int> buffer,
                               raft::device_span<int> total)
{
  extern __shared__ int sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = ii < buffer.size() ? buffer[ii] : 0;
  __syncthreads();

  for (int ss = blockDim.x / 2; ss > 0; ss >>= 1)
    {
      if (tid < ss)
        sdata[tid] += sdata[tid + ss];
      __syncthreads();
    }

  if (tid == 0)
    atomicAdd(total.data(), sdata[0]);
}

void reduce(rmm::device_uvector<int>& buffer, rmm::device_scalar<int>& total)
{
  constexpr unsigned int block_size = 64;
  const unsigned int grid_size = (buffer.size() + block_size - 1) / block_size;

  _reduce<<<grid_size, block_size, block_size * sizeof(int), buffer.stream()>>>(
    raft::device_span<int>(buffer.data(), buffer.size()),
    raft::device_span<int>(total.data(), total.size()));
}