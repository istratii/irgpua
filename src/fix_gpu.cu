#include "fix_gpu.cuh"

void fix_image_gpu(Image& to_fix)
{
  const unsigned int true_size = to_fix.size();
  const unsigned int image_size = to_fix.width * to_fix.height;
  const raft::handle_t handle{};
  cudaStream_t stream = handle.get_stream();

  rmm::device_uvector<int> buffer(true_size, stream);
  CUDA_CHECK_ERROR(cudaMemcpyAsync(buffer.data(), to_fix.buffer, true_size * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));

  // #1 Compact
  compact(buffer);
  buffer.resize(image_size, stream);

  // #2 Apply map to fix pixels
  map_fix(buffer);

  // // #3 Histogram equalization
  rmm::device_uvector<int> hist = histogram(buffer);
  equalize_histogram(buffer, hist);

  CUDA_CHECK_ERROR(cudaMemcpyAsync(to_fix.buffer, buffer.data(), image_size * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
}