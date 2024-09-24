
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

rmm::device_uvector<int> histogram(rmm::device_uvector<int>& buffer)
{
  const unsigned int hist_size = 256;
  constexpr unsigned int hist_bytes_size = hist_size * sizeof(int);
  rmm::device_uvector<int> hist(hist_size, buffer.stream());

  CUDA_CHECK_ERROR(cudaMemset(hist.data(), 0, hist_bytes_size));

#define THREADS_PER_BLOCK 1024

  _histogram<<<(buffer.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
               THREADS_PER_BLOCK, hist_bytes_size, buffer.stream()>>>(
    raft::device_span<int>(buffer.data(), buffer.size()),
    raft::device_span<int>(static_cast<int*>(hist.data()), hist.size()));

#undef THREADS_PER_BLOCK

  CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));

  return hist;
}

__device__ int cdf_min = 0;

static __global__ void _compute_first_non_zero(raft::device_span<int> hist)
{
  const unsigned int N = hist.size();
  int ii = 0;
  while (ii < N && !hist[ii])
    ++ii;
  cdf_min = ii < N ? hist[ii] : -1;
}

static __global__ void _equalize_histogram(raft::device_span<int> buffer,
                                           raft::device_span<int> hist)
{
  __shared__ int s_cdf_min;
  const unsigned int tid = threadIdx.x;
  const unsigned int id = tid + blockIdx.x * blockDim.x;
  const unsigned int N = buffer.size();

  if (tid == 0)
    s_cdf_min = cdf_min;
  __syncthreads();

  if (id < N)
    buffer[id] = roundf(
      ((hist[buffer[id]] - s_cdf_min) / static_cast<float>(N - s_cdf_min))
      * 255.0f);
}

void equalize_histogram(rmm::device_uvector<int>& buffer,
                        rmm::device_uvector<int>& hist)
{
  cudaStream_t stream = buffer.stream();

  scan(hist, SCAN_INCLUSIVE);

  _compute_first_non_zero<<<1, 1, 0, buffer.stream()>>>(
    raft::device_span<int>(hist.data(), hist.size()));

  CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

#define THREADS_PER_BLOCK 1024

  _equalize_histogram<<<(buffer.size() + THREADS_PER_BLOCK - 1)
                          / THREADS_PER_BLOCK,
                        THREADS_PER_BLOCK, 0, stream>>>(
    raft::device_span<int>(buffer.data(), buffer.size()),
    raft::device_span<int>(hist.data(), hist.size()));

  CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

#undef THREADS_PER_BLOCK
}