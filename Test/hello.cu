
#include <VectorAdd.cuh>

#include <stdio.h>
#include <thread>
#include <time.h>

void 
setGpu()
{
  //获取GPU设备数量
  int iDeviceCount = 0;
  cudaGetDeviceCount(&iDeviceCount); 
  printf("GPU count: %d\n", iDeviceCount);
  
  //获取运行时设备的index 
  int iDev = 0;
  cudaSetDevice(iDev);
  printf("GPU index: %d\n", iDev);
}

void 
initialData(float *arr, int elementCount)
{
  for(int i = 0; i < elementCount; i++)
  {
    arr[i] = i;
  }
}

void
vectorAddCpu(float *A, float *B, float *C, int N)
{
  for(int i = 0; i < N; i++)
  {
    C[i] = A[i] + B[i];
  }
}

int 
main()
{
  setGpu();

  unsigned long long iElemCount = 100000000;
  size_t stBytesCount = iElemCount*sizeof(float);
  float *fpHostA = (float *)malloc(stBytesCount);
  float *fpHostB = (float *)malloc(stBytesCount);
  float *fpHostC = (float *)malloc(stBytesCount);

  memset(fpHostA, 0, stBytesCount);
  memset(fpHostB, 0, stBytesCount);
  memset(fpHostC, 0, stBytesCount);

  float *fpDeviceA, *fpDeviceB, *fpDeviceC;
  cudaMalloc(&fpDeviceA, stBytesCount);
  cudaMalloc(&fpDeviceB, stBytesCount);
  cudaMalloc(&fpDeviceC, stBytesCount);

  cudaMemset(fpHostA, 0, stBytesCount);
  cudaMemset(fpHostB, 0, stBytesCount);
  cudaMemset(fpHostC, 0, stBytesCount);

  initialData(fpHostA, iElemCount);
  initialData(fpHostB, iElemCount);

  cudaMemcpy(fpDeviceA, fpHostA, stBytesCount, cudaMemcpyHostToDevice);
  cudaMemcpy(fpDeviceB, fpHostB, stBytesCount, cudaMemcpyHostToDevice);

  dim3 block(32);
  dim3 grid(iElemCount/32);

  clock_t startGpu, endGpu, startCpu, endCpu;

  startCpu = clock();
  vectorAddCpu(fpHostA, fpHostB, fpHostC, iElemCount);
  endCpu = clock();

  startGpu = clock();
  vectorAdd<<<grid, block>>>(fpDeviceA, fpDeviceB, fpDeviceC);
  endGpu = clock();
  cudaDeviceSynchronize();

  printf("cpu:%f\ngpu:%f\n", (double)(endCpu - startCpu)/CLOCKS_PER_SEC, (double)(endGpu - startGpu)/CLOCKS_PER_SEC);
  
  cudaMemcpy(fpHostA, fpDeviceA, stBytesCount, cudaMemcpyDeviceToHost);
  cudaMemcpy(fpHostB, fpDeviceB, stBytesCount, cudaMemcpyDeviceToHost);
  cudaMemcpy(fpHostC, fpDeviceC, stBytesCount, cudaMemcpyDeviceToHost);

  // for(int i = 0; i < iElemCount; i++)
  // {
  //   printf("%f + %f = %f\n", fpHostA[i], fpHostB[i], fpHostC[i]);
  // }
} 