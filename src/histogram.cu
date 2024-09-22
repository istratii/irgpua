
#include "histogram.cuh"

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

static __global__ void _histogram(raft::device_span<int> buffer,
                                  raft::device_span<int> hist)
{
  constexpr unsigned int hist_size = 256;
  extern __shared__ int s_histo[];
  const unsigned int tid = threadIdx.x;
  const unsigned int id = blockIdx.x * blockDim.x + tid;

  // block-loop pattern
  for (int ii = tid; ii < hist_size; ii += blockDim.x)
    s_histo[ii] = 0;
  __syncthreads();

  if (id < buffer.size())
    atomicAdd(&s_histo[buffer[id]], 1);
  __syncthreads();

  for (int ii = tid; ii < hist_size; ii += blockDim.x)
    atomicAdd(&hist[ii], s_histo[ii]);
}

#define THREADS_PER_BLOCK 1024

void histogram(rmm::device_uvector<int>& buffer)
{
  const raft::handle_t handle{};
  rmm::device_uvector<int> hist(256, handle.get_stream());
  thrust::uninitialized_fill(handle.get_thrust_policy(), buffer.begin(),
                             buffer.end(), 0);

  _histogram<<<(buffer.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
               THREADS_PER_BLOCK, sizeof(int) * 256, buffer.stream()>>>(
    raft::device_span<int>(buffer.data(), buffer.size()),
    raft::device_span<int>(hist.data(), buffer.size()));

  // CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}