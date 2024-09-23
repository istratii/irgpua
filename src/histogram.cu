
#include "histogram.cuh"

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

rmm::device_buffer histogram(rmm::device_uvector<int>& buffer)
{
  constexpr unsigned int hist_bytes_size = 256 * sizeof(int);
  rmm::device_buffer hist(hist_bytes_size, buffer.stream());
  CUDA_CHECK_ERROR(cudaMemset(hist.data(), 0, hist_bytes_size));

  _histogram<<<(buffer.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
               THREADS_PER_BLOCK, hist_bytes_size, buffer.stream()>>>(
    raft::device_span<int>(buffer.data(), buffer.size()),
    raft::device_span<int>(static_cast<int*>(hist.data()), buffer.size()));

  CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));

  return hist;
}