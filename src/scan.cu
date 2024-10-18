
#include "scan.cuh"

#define MAX_BLOCKS 4096

#define X 0
#define A 1
#define P 2

struct _Setup
{
  int next_block_id;
  int global_sums[4096];
  int local_sums[4096];
  cuda::atomic<int, cuda::thread_scope_device> states[4096];
};

static __global__ void _scan(raft::device_span<int> buffer,
                             raft::device_span<_Setup> setup)
{
  __shared__ unsigned int bid;
  __shared__ unsigned int state;
  extern __shared__ int s_buffer[];

  const unsigned int tid = threadIdx.x;
  _Setup* sptr = setup.data();

  // replace blockIdx.x to avoid dead locks
  // first come first served
  if (tid == 0)
    bid = atomicAdd(&sptr->next_block_id, 1);
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
      sptr->local_sums[bid] = s_buffer[tid];
      if (bid == 0)
        sptr->global_sums[bid] = sptr->local_sums[bid];
      sptr->states[bid].store(bid ? A : P, cuda::memory_order_seq_cst);
    }
  __syncthreads();

  for (int ii = bid - 1; ii >= 0; --ii)
    {
      if (tid == 0)
        while ((state = sptr->states[ii].load(cuda::memory_order_seq_cst)) == X)
          ;
      __syncthreads();

      if (state == A)
        {
          s_buffer[tid] += sptr->local_sums[ii];
          __syncthreads();
        }
      else // P
        {
          s_buffer[tid] += sptr->global_sums[ii];
          if (tid == blockDim.x - 1)
            {
              sptr->global_sums[bid] = s_buffer[tid];
              sptr->states[bid].store(P, cuda::memory_order_seq_cst);
            }
          __syncthreads();
          break;
        }
    }

  if (id < N)
    buffer[id] = s_buffer[tid];
}

static __global__ void
_prepare_buffer_for_exclusive_scan(raft::device_span<int> buffer)
{
  if (threadIdx.x == 0)
    buffer[0] = 0;
}

void scan(rmm::device_uvector<int>& memchunk,
          raft::device_span<int> buffer_dspan,
          ScanMode mode)
{
  cudaStream_t stream = memchunk.stream();

  // prepare setup
  int* begin_raw_setup = memchunk.data() + scan_offset / sizeof(int);
  constexpr size_t size_raw_setup = bytes_per_scan / sizeof(int);
  raft::device_span<int> raw_setup_span(begin_raw_setup, size_raw_setup);
  CUDA_CHECK_ERROR(
    cudaMemsetAsync(raw_setup_span.data(), 0, bytes_per_scan, stream));
  raft::device_span<_Setup> setup_span((_Setup*)begin_raw_setup, 1);

  constexpr unsigned int block_size = 1024;
  const unsigned int grid_size =
    (buffer_dspan.size() + block_size - 1) / block_size;

  if (mode == SCAN_EXCLUSIVE)
    _prepare_buffer_for_exclusive_scan<<<1, 1, 0, stream>>>(buffer_dspan);

  _scan<<<grid_size, block_size, block_size * sizeof(int), stream>>>(
    buffer_dspan, setup_span);
}