constexpr int garbage_val = -27;

__global__ void
compact_kernel(int* d_in, int* d_out, int* d_predicate, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    d_predicate[idx] = (d_in[idx] != garbage_val) ? 1 : 0;
}

__global__ void
scatter_kernel(int* d_in, int* d_out, int* d_predicate, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size && d_in[idx] != garbage_val)
    {
      int targetIdx = d_predicate[idx];
      d_out[targetIdx] = d_in[idx];
    }
}

void compact(int* d_in, int* d_out, int* d_predicate, int size)
{
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  compactKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_predicate, size);
  cudaDeviceSynchronize();

  // A remplacer par notre scan pour la version handwwritten
  thrust::exclusive_scan(thrust::device, d_predicate, d_predicate + size,
                         d_predicate);

  scatterKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, d_predicate,
                                                    size);
  cudaDeviceSynchronize();
}
