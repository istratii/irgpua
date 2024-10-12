#include "fix_gpu.cuh"

void fix_image_gpu(Image& to_fix, cudaStream_t stream)
{
  const unsigned int actual_size = to_fix.size();
  const unsigned int image_size = to_fix.width * to_fix.height;
  rmm::device_uvector<int> buffer(actual_size, stream);

  // copy image to device memory, stream aware
  cudaMemcpyAsync(buffer.data(), to_fix.buffer, actual_size * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

  // #1 Compact
  // Build predicate vector
  // Compute the exclusive sum of the predicate
  // Scatter to the corresponding addresses
  compact(buffer);

  // only `image_size` has to be used after
  // `image_size` <= `actual_size`, thus no overhead
  buffer.resize(image_size, stream);

  // #2 Apply map to fix pixels
  map_fix(buffer);

  // #3 Histogram equalization
  // Histogram
  // Compute the inclusive sum scan of the histogram
  // Find the first non-zero value in the cumulative histogram
  // Apply the map transformation of the histogram equalization
  equalize_histogram(buffer);

  // copy image to host back from device, stream aware
  cudaMemcpyAsync(to_fix.buffer, buffer.data(), image_size * sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
}