#include "compact.cuh"

#define GARBAGE_VAL -27

static __global__ void _compact(raft::device_span<int> buffer,
                                raft::device_span<int> d_predicate)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < buffer.size())
    d_predicate[idx] = buffer[idx] != GARBAGE_VAL;
}

static __global__ void _scatter(raft::device_span<int> buffer,
                                raft::device_span<int> d_predicate)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < buffer.size() && buffer[idx] != GARBAGE_VAL)
    buffer[d_predicate[idx]] = buffer[idx];
}

void compact(rmm::device_uvector<int>& buffer)
{
  const unsigned int size = buffer.size();
  constexpr unsigned int block_size = 1024;
  const unsigned int grid_size = (size + block_size - 1) / block_size;
  cudaStream_t stream = buffer.stream();
  rmm::device_uvector<int> pred(size, stream);
  raft::device_span<int> buffer_span(buffer.data(), size);
  raft::device_span<int> pred_span(pred.data(), size);

  _compact<<<grid_size, block_size, 0, stream>>>(buffer_span, pred_span);
  scan(pred, SCAN_EXCLUSIVE);
  _scatter<<<grid_size, block_size, 0, stream>>>(buffer_span, pred_span);
}