#define ASSOC_VAL(idx) (((idx) % 4 == 0) ? 1 : \
                        ((idx) % 4 == 1) ? -5 : \
                        ((idx) % 4 == 2) ? 3 : -8)

__global__ void map_kernel(int* d_buffer, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
      d_buffer[idx] += ASSOC_VAL(idx);
}
