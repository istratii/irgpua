
#include "scan.cuh"

#define MAX_BLOCKS 4096
#define THREADS_PER_BLOCK 1024

#define X 0
#define A 1
#define P 2

__device__ int counter = 0;
__device__ int global[MAX_BLOCKS];
__device__ int local[MAX_BLOCKS];
__device__ cuda::atomic<int, cuda::thread_scope_device> states[MAX_BLOCKS];

__global__ void _init_descriptors()
{
  const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  global[id] = 0;
  local[id] = 0;
  states[id] = X;
}

__global__ void _scan(raft::device_span<T> buffer)
{
  __shared__ unsigned int bid;
  __shared__ unsigned int state;
  extern __shared__ int s_buffer[];

  const unsigned int tid = threadIdx.x;

  // replace blockIdx.x to avoid dead locks
  if (tid == 0)
    bid = atomicAdd(&counter, 1);
  __syncthreads();

  const unsigned int id = bid * blockDim.x + tid;
  const int N = buffer.size();

  s_buffer[tid] = id < N ? buffer[id] : 0;
  __syncthreads();

  // compute local sum
  for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
      int val{};
      __syncthreads();
      if (tid >= offset)
        val = s_buffer[tid - offset];
      __syncthreads();
      if (tid >= offset)
        s_buffer[tid] += val;
    }

  if (tid == blockDim.x - 1)
    {
      local[bid] = s_buffer[tid];
      if (bid == 0)
        global[bid] = local[bid];
      states[bid].store(bid ? A : P, cuda::memory_order_seq_cst);
    }
  __syncthreads();

  for (int ii = bid - 1; ii >= 0; --ii)
    {
      if (tid == 0)
        while ((state = states[ii].load(cuda::memory_order_seq_cst)) == X)
          ;
      __syncthreads();

      if (state == A)
        {
          s_buffer[tid] += local[ii];
          __syncthreads();
        }
      else // P
        {
          s_buffer[tid] += global[ii];
          if (tid == blockDim.x - 1)
            {
              global[bid] = s_buffer[tid];
              states[bid].store(P, cuda::memory_order_seq_cst);
            }
          __syncthreads();
          break;
        }
    }

  if (id < N)
    buffer[id] = s_buffer[tid];
}

void scan(rmm::device_uvector<int>& buffer, bool inclusive)
{
  const int tmp = 0;
  cudaMemcpyToSymbol(counter, &tmp, sizeof(int), 0, cudaMemcpyHostToDevice);

  _init_descriptors<<<(MAX_BLOCKS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                      THREADS_PER_BLOCK, 0, buffer.stream()>>>();

  _scan<<<(buffer.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
          THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int),
          buffer.stream()>>>(
    raft::device_span<int>(buffer.data(), buffer.size()));

  CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}