#include "compact.cuh"

#define GARBAGE_VAL -27

static __global__ void _compact(raft::device_span<int> buffer_dspan,
                                raft::device_span<int> pred_dspan)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < buffer_dspan.size())
    pred_dspan[idx] = buffer_dspan[idx] != GARBAGE_VAL;
}

static __global__ void _scatter(raft::device_span<int> buffer_dspan,
                                raft::device_span<int> pred_dspan)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < buffer_dspan.size() && buffer_dspan[idx] != GARBAGE_VAL)
    buffer_dspan[pred_dspan[idx]] = buffer_dspan[idx];
}
#ifdef _IRGPUA_GPU
#else // _IRGPUA_GPU_INDUS
struct IsNotGarbage
{
  __device__ bool operator()(const int x) const { return x != GARBAGE_VAL; }
};
#endif

void compact(rmm::device_buffer& memchunk, raft::device_span<int> buffer_dspan)
{
  raft::common::nvtx::range fscope("compact");

  cudaStream_t stream = memchunk.stream();
  char* chunk = static_cast<char*>(memchunk.data());
  raft::device_span<int> pred_dspan(
    reinterpret_cast<int*>(chunk + predicate_offset),
    bytes_per_predicate / sizeof(int));

  const unsigned int size = buffer_dspan.size();
  constexpr unsigned int block_size = 1024;
  const unsigned int grid_size = (size + block_size - 1) / block_size;
#ifdef _IRGPUA_GPU
  _compact<<<grid_size, block_size, 0, stream>>>(buffer_dspan, pred_dspan);
  scan(memchunk, pred_dspan, SCAN_EXCLUSIVE);
#else // _IRGPUA_GPU_INDUS
  thrust::transform(thrust::cuda::par.on(stream), buffer_dspan.begin(),
                    buffer_dspan.end(), pred_dspan.begin(), IsNotGarbage());
  thrust::exclusive_scan(thrust::cuda::par.on(stream), pred_dspan.begin(),
                         pred_dspan.end(), pred_dspan.begin());
#endif
  _scatter<<<grid_size, block_size, 0, stream>>>(buffer_dspan, pred_dspan);
}