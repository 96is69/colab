#include<bits/stdc++.h>
using namespace std;
// Kernel function to add the elements of two arrays
float *x, *y;

__global__
void add(int n, float *x, float *y, vector<int> v)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];

  printf("%d ", v[2]);
}

int main()
{
  int N = 1<<20;
  

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

 

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) 
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  vector<int> nx = {1,2,3};


  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y, nx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  cout<<x[0];

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}