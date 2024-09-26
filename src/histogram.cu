
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
  cudaStream_t stream = buffer.stream();
  const unsigned int hist_size = 256;
  constexpr unsigned int hist_bytes_size = hist_size * sizeof(int);
  rmm::device_uvector<int> hist(hist_size, stream);

  CUDA_CHECK_ERROR(cudaMemsetAsync(hist.data(), 0, hist_bytes_size, stream));

#define THREADS_PER_BLOCK 1024

  _histogram<<<(buffer.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
               THREADS_PER_BLOCK, hist_bytes_size, stream>>>(
    raft::device_span<int>(buffer.data(), buffer.size()),
    raft::device_span<int>(hist.data(), hist.size()));

#undef THREADS_PER_BLOCK

  return hist;
}

static __global__ void _compute_first_non_zero(raft::device_span<int> hist,
                                               raft::device_span<int> cdf_min)
{
  const unsigned int N = hist.size();
  int ii = 0;
  while (ii < N && !hist[ii])
    ++ii;
  cdf_min[0] = ii < N ? hist[ii] : -1;
  assert(cdf_min[0] >= 0);
}

static __global__ void _equalize_histogram(raft::device_span<int> buffer,
                                           raft::device_span<int> hist,
                                           raft::device_span<int> cdf_min)
{
  const unsigned int tid = threadIdx.x;
  const unsigned int id = tid + blockIdx.x * blockDim.x;
  const unsigned int N = buffer.size();

  if (id < N)
    {
      float tmp = (hist[buffer[id]] - cdf_min[0])
        / static_cast<float>(N - cdf_min[0] + 1e-9f);
      tmp = roundf(tmp * 255.0f);
      buffer[id] = tmp;
    }
}

void equalize_histogram(rmm::device_uvector<int>& buffer,
                        rmm::device_uvector<int>& hist)
{
  cudaStream_t stream = buffer.stream();
  rmm::device_scalar<int> cdf_min(0, stream);
  raft::device_span<int> cdf_min_span(cdf_min.data(), cdf_min.size());
  raft::device_span<int> hist_span(hist.data(), hist.size());

  scan(hist, SCAN_INCLUSIVE);
  _compute_first_non_zero<<<1, 1, 0, stream>>>(hist_span, cdf_min_span);

#define THREADS_PER_BLOCK 1024

  _equalize_histogram<<<(buffer.size() + THREADS_PER_BLOCK - 1)
                          / THREADS_PER_BLOCK,
                        THREADS_PER_BLOCK, 0, stream>>>(
    raft::device_span<int>(buffer.data(), buffer.size()), hist_span,
    cdf_min_span);

#undef THREADS_PER_BLOCK
}