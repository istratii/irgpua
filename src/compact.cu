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

void compact(char* chunk,
             cudaStream_t stream,
             raft::device_span<int> buffer_dspan)
{
  raft::common::nvtx::range fscope("compact");

  const unsigned int size = buffer_dspan.size();
  constexpr unsigned int block_size = 1024;
  const unsigned int grid_size = (size + block_size - 1) / block_size;

  raft::device_span<int> pred_dspan(
    reinterpret_cast<int*>(chunk + predicate_offset),
    bytes_per_predicate / sizeof(int));

  WRAP_NVTX(
    "compact kernel",
    (_compact<<<grid_size, block_size, 0, stream>>>(buffer_dspan, pred_dspan)));
  scan(chunk, stream, pred_dspan, SCAN_EXCLUSIVE);
  WRAP_NVTX(
    "scatter kernel",
    (_scatter<<<grid_size, block_size, 0, stream>>>(buffer_dspan, pred_dspan)));
}