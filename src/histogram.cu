
#include "histogram.cuh"

#ifdef _IRGPUA_GPU
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
  // assert(cdf_min[0] >= 0);
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
#else // _IRGPUA_GPU_INDUS
struct EqualizeFunctor
{
  int cdf_min;
  unsigned int N;

  __device__ int operator()(int value) const
  {
    float tmp = value - cdf_min;
    tmp /= static_cast<float>(N - cdf_min + 1e-9f);
    tmp *= 255.0f;
    return roundf(tmp);
  }
};
#endif

void equalize_histogram(rmm::device_buffer& memchunk,
                        raft::device_span<int> buffer_dspan)
{
  raft::common::nvtx::range fscope("equalize histrogram");

  cudaStream_t stream = memchunk.stream();
  char* chunk = static_cast<char*>(memchunk.data());
  raft::device_span<int> histogram_dspan(
    reinterpret_cast<int*>(chunk + histogram_offset),
    bytes_per_histogram / sizeof(int));
  // its part from memchunk is already set to 0
  raft::device_span<int> cdf_min_dspan(
    reinterpret_cast<int*>(chunk + cdf_min_offset), 1);

#ifdef _IRGPUA_GPU
  constexpr unsigned int block_size = 1024;
  const unsigned int grid_size =
    (buffer_dspan.size() + block_size - 1) / block_size;
  // compute histogram
  _histogram<<<grid_size, block_size, bytes_per_histogram, stream>>>(
    buffer_dspan, histogram_dspan);
  // equalize histogram
  // compute cumulative histogram sum
  scan(memchunk, histogram_dspan, SCAN_INCLUSIVE);
  _compute_first_non_zero<<<1, 1, 0, stream>>>(histogram_dspan, cdf_min_dspan);
  _equalize_histogram<<<grid_size, block_size, 0, stream>>>(
    buffer_dspan, histogram_dspan, cdf_min_dspan);
#else // _IRGPUA_GPU_INDUS
  constexpr int HIST_SIZE = 256;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven<int*, int, int, std::size_t>(
    d_temp_storage, temp_storage_bytes, buffer_dspan.data(),
    histogram_dspan.data(), HIST_SIZE, 0, HIST_SIZE - 1, buffer_dspan.size(),
    stream);
  rmm::device_buffer temp_storage(temp_storage_bytes, stream);
  d_temp_storage = temp_storage.data();
  cub::DeviceHistogram::HistogramEven<int*, int, int, std::size_t>(
    d_temp_storage, temp_storage_bytes, buffer_dspan.data(),
    histogram_dspan.data(), HIST_SIZE, 0, HIST_SIZE - 1, buffer_dspan.size(),
    stream);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), histogram_dspan.begin(),
                         histogram_dspan.end(), histogram_dspan.begin());
  thrust::device_pointer_cast(cdf_min_dspan.data())[0] =
    thrust::reduce(thrust::cuda::par.on(stream), histogram_dspan.begin(),
                   histogram_dspan.end(), std::numeric_limits<int>::max(),
                   thrust::minimum<int>());
  thrust::transform(
    thrust::cuda::par.on(stream), buffer_dspan.begin(), buffer_dspan.end(),
    buffer_dspan.begin(),
    EqualizeFunctor{thrust::device_pointer_cast(cdf_min_dspan.data())[0],
                    static_cast<unsigned int>(buffer_dspan.size())});
#endif
}