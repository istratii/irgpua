#include "compact.cuh"

#define GARBAGE_VAL -27

static __global__ void _compact_kernel(raft::device_span<int> buffer,
                                       raft::device_span<int> d_predicate)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < buffer.size())
    d_predicate[idx] = buffer[idx] != GARBAGE_VAL;
}

static __global__ void _scatter_kernel(raft::device_span<int> buffer,
                                       raft::device_span<int> d_predicate)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < buffer.size() && buffer[idx] != GARBAGE_VAL)
    buffer[d_predicate[idx]] = buffer[idx];
}

#define THREADS_PER_BLOCK 1024

void compact(rmm::device_uvector<int>& buffer)
{
  unsigned int size = buffer.size();
  cudaStream_t stream = buffer.stream();

  rmm::device_uvector<int> d_predicate(size, stream);

  _compact_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                    THREADS_PER_BLOCK, 0, stream>>>(
    raft::device_span<int>(buffer.data(), size),
    raft::device_span<int>(d_predicate.data(), size));
  CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

  scan(d_predicate, SCAN_EXCLUSIVE);

  _scatter_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                    THREADS_PER_BLOCK, 0, stream>>>(
    raft::device_span<int>(buffer.data(), size),
    raft::device_span<int>(d_predicate.data(), size));
  CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
}