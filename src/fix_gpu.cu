#include "fix_gpu.cuh"

void fix_image_gpu(Image& to_fix, cudaStream_t stream)
{
  const unsigned int actual_size = to_fix.size();
  const unsigned int image_size = to_fix.width * to_fix.height;
  rmm::device_uvector<int> memchunk(bytes_per_chunk / sizeof(int), stream);

  // create device span for buffer
  int* begin_buffer = memchunk.data() + buffer_offset / sizeof(int);
  raft::device_span<int> buffer_dspan(begin_buffer, actual_size);

  // copy image to device memory, stream aware
  CUDA_CHECK_ERROR(cudaMemcpyAsync(buffer_dspan.data(), to_fix.buffer,
                                   actual_size * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));

  // #1 Compact
  // Build predicate vector
  // Compute the exclusive sum of the predicate
  // Scatter to the corresponding addresses
  compact(memchunk, buffer_dspan);

  // only `image_size` has to be used after
  // `image_size` <= `actual_size`, thus no overhead
  buffer_dspan = raft::device_span<int>(begin_buffer, image_size);

  // #2 Apply map to fix pixels
  map_fix(buffer_dspan, stream);

  // #3 Histogram equalization
  // Histogram
  // Compute the inclusive sum scan of the histogram
  // Find the first non-zero value in the cumulative histogram
  // Apply the map transformation of the histogram equalization
  equalize_histogram(memchunk, buffer_dspan);

  // compute the total of each image
  // doing this here because the images are still on device memory
  int* begin_total = memchunk.data() + total_offset / sizeof(int);
  raft::device_span<int> total_dspan(begin_total, 1);
  CUDA_CHECK_ERROR(cudaMemsetAsync(total_dspan.data(), 0, sizeof(int), stream));
  reduce(buffer_dspan, total_dspan, stream);

  // copy total pixel sum to host, stream aware
  CUDA_CHECK_ERROR(cudaMemcpyAsync(to_fix.to_sort.total, total_dspan.data(),
                                   sizeof(int), cudaMemcpyDeviceToHost,
                                   stream));

  // copy image to host back from device, stream aware
  CUDA_CHECK_ERROR(cudaMemcpyAsync(to_fix.buffer, buffer_dspan.data(),
                                   image_size * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
}
