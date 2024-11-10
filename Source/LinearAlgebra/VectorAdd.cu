#include <VectorAdd.cuh>

__global__ void
vectorAdd(float *A, float *B, float *C)
{
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int id = tid + bid*blockDim.x;

  C[id] = A[id] + B[id];
}