
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
      float tmp = hist[buffer[id]] - cdf_min[0];
      tmp /= static_cast<float>(N - cdf_min[0] + 1e-9f);
      tmp *= 255.0f;
      tmp = roundf(tmp);
      buffer[id] = tmp;
    }
}

void equalize_histogram(rmm::device_uvector<int>& buffer)
{
  cudaStream_t stream = buffer.stream();
  raft::device_span<int> buffer_span(buffer.data(), buffer.size());
  constexpr unsigned int block_size = 1024;
  const unsigned int grid_size = (buffer.size() + block_size - 1) / block_size;

  // compute histogram
  rmm::device_uvector<int> histogram(256, stream);
  constexpr unsigned int histogram_size_bytes = 256 * sizeof(int);
  cudaMemsetAsync(histogram.data(), 0, histogram_size_bytes, stream);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
  raft::device_span<int> histogram_span(histogram.data(), histogram.size());
  _histogram<<<grid_size, block_size, histogram_size_bytes, stream>>>(
    buffer_span, histogram_span);

  // equalize histogram
  rmm::device_scalar<int> cdf_min(0, stream);
  raft::device_span<int> cdf_min_span(cdf_min.data(), cdf_min.size());
  // compute cumulative histogram sum
  scan(histogram, SCAN_INCLUSIVE);
  _compute_first_non_zero<<<1, 1, 0, stream>>>(histogram_span, cdf_min_span);
  _equalize_histogram<<<grid_size, block_size, 0, stream>>>(
    buffer_span, histogram_span, cdf_min_span);
}