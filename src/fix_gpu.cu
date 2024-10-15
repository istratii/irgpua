#include "fix_gpu.cuh"

void fix_image_gpu(Image& to_fix, cudaStream_t stream)
{
  const unsigned int actual_size = to_fix.size();
  const unsigned int image_size = to_fix.width * to_fix.height;
  rmm::device_uvector<int> buffer(actual_size, stream);

  // copy image to device memory, stream aware
  cudaMemcpyAsync(buffer.data(), to_fix.buffer, actual_size * sizeof(int),
                  cudaMemcpyHostToDevice, stream);

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

  // compute the total of each image
  // doing this here because the images are still on device memory
  rmm::device_scalar<int> total(0, stream);
  reduce(buffer, total);

  // copy total pixel sum to host, stream aware
  cudaMemcpyAsync(&to_fix.to_sort.total, total.data(), sizeof(int),
                  cudaMemcpyDeviceToHost, stream);

  // copy image to host back from device, stream aware
  cudaMemcpyAsync(to_fix.buffer, buffer.data(), image_size * sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
}