#include "reduce.cuh"

#ifdef _IRGPUA_GPU
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
#else // _IGRPUA_GPU_INDUS
#endif

void reduce(raft::device_span<int> buffer_dspan,
            raft::device_span<int> total_dspan,
            cudaStream_t stream)
{
  raft::common::nvtx::range fscope("reduce");

#ifdef _IRGPUA_GPU
  constexpr unsigned int block_size = 64;
  const unsigned int grid_size =
    (buffer_dspan.size() + block_size - 1) / block_size;

  _reduce<<<grid_size, block_size, block_size * sizeof(int), stream>>>(
    buffer_dspan, total_dspan);
#else // _IRGPUA_GPU_INDUS
  thrust::device_pointer_cast(total_dspan.data())[0] =
    thrust::reduce(thrust::cuda::par.on(stream), buffer_dspan.begin(),
                   buffer_dspan.end(), 0, thrust::plus<int>());
#endif
}