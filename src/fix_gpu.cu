#include "fix_gpu.cuh"

void fix_image_gpu(Image& to_fix, cudaStream_t stream)
{
  raft::common::nvtx::range fscope("fix image gpu");

  const unsigned int actual_size = to_fix.size();
  const unsigned int image_size = to_fix.width * to_fix.height;
  rmm::device_buffer memchunk(bytes_per_chunk, stream);
  char* chunk = static_cast<char*>(memchunk.data());

  // create device span for buffer
  raft::device_span<int> buffer_dspan(
    reinterpret_cast<int*>(chunk + buffer_offset), actual_size);

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
  buffer_dspan = raft::device_span<int>(
    reinterpret_cast<int*>(chunk + buffer_offset), image_size);

  // #2 Apply map to fix pixels
  map_fix(buffer_dspan, stream);

  // reduce number of cudaMemsetAsync calls
  CUDA_CHECK_ERROR(cudaMemsetAsync(
    chunk + histogram_offset, 0,
    bytes_per_histogram + bytes_per_cdf_min + bytes_per_total, stream));

  // #3 Histogram equalization
  // Histogram
  // Compute the inclusive sum scan of the histogram
  // Find the first non-zero value in the cumulative histogram
  // Apply the map transformation of the histogram equalization
  equalize_histogram(memchunk, buffer_dspan);

  // compute the total of each image
  // doing this here because the images are still on device memory
  raft::device_span<int> total_dspan(
    reinterpret_cast<int*>(chunk + total_offset), 1);
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
