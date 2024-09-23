#include <cuda_runtime.h>
#include <raft/handle.hpp>
#include <raft/linalg/scan.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>

#define GARBAGE_VAL -27

__global__ void compact_kernel(int* d_in, int* d_predicate, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_predicate[idx] = (d_in[idx] != GARBAGE_VAL)
}

__global__ void
scatter_kernel(int* d_in, int* d_out, int* d_predicate, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size && d_in[idx] != GARBAGE_VAL)
    {
      int out_idx = d_predicate[idx];
      d_out[out_idx] = d_in[idx];
    }
}

void compact(rmm::device_uvector<int>& d_in,
             rmm::device_uvector<int>& d_out,
             raft::handle_t const& handle)
{
  cudaStream_t stream = handle.get_stream();
  std::size_t size = d_in.size();

  rmm::device_uvector<int> d_predicate(size, stream);

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  scatter_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
    raft::device_span<int>(d_in.data(), d_in.size()), 
    raft::device_span<int>(d_out.data(), d_out.size()), 
    raft::device_span<int>(d_predicate.data(), d_predicate.size()), 
    size);
  cudaStreamSynchronize(stream);

  raft::linalg::inclusive_scan(handle, d_predicate.data(), d_predicate.data(),
                               size, stream);

  scatter_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
    d_in.data(), d_out.data(), d_predicate.data(), size);
  cudaStreamSynchronize(stream);
}