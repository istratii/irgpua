__global__ void map_kernel(int* d_buffer, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // on evite le plus possible le branching avec trop de if
        const int assoc_val[4] = {1, -5, 3, -8};
        d_buffer[idx] += ops[idx % 4];
    }
}