#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 256

__global__ void matmul_elem(int n, float *a, float *b, float *c){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row<n && col<n){
		float dot_prod = 0.f;
		for(int i=0;i<n;i++){
			// iterate over the row of the first matrix
			// and the columns of the second matrix
			dot_prod += a[row*n + i] * b[i*n + col];
		}
		c[row*n + col] = dot_prod;
	}
}


void matmul_cpu(int n, float a[N][N], float b[N][N], float c[N][N]){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			float sum = 0.f;
			for(int k=0;k<n;k++){
				sum += a[i][k] * b[k][j];
			}
			c[i][j] = sum ;
		}
	}
}

int main(){
	//int N=256;
	size_t size = N*N*sizeof(float);
	static float a[N][N], b[N][N], c[N][N], c_cpu[N][N];

	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			a[i][j]=1.0f;
			b[i][j]=1.0f;
		}
	}
	float *a_d, *b_d, *c_d;
	cudaMalloc(&a_d, size);
	cudaMalloc(&b_d, size);
	cudaMalloc(&c_d, size);

	cudaMemcpy(a_d,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(b_d,b,size,cudaMemcpyHostToDevice);

	dim3 block(16,16);
	dim3 grid((N+15)/16,(N+15)/16);
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaEventRecord(start);
	matmul_elem<<<grid,block>>>(N,a_d,b_d,c_d);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float gpu_time = 0;
	cudaEventElapsedTime(&gpu_time,start,stop);

	printf("#### GPU Time : %f ms ####\n",gpu_time);

	cudaMemcpy(c,c_d,size,cudaMemcpyDeviceToHost);
	
	// CPU TIMING
	auto start_cpu = std::chrono::high_resolution_clock::now();
	matmul_cpu(N,a,b,c_cpu);
	auto end_cpu = std::chrono::high_resolution_clock::now();

	double cpu_time = std::chrono::duration<double,std::milli>(end_cpu - start_cpu).count();
	printf("#### CPU Time : %f ms ####\n",cpu_time);



	printf("c_gpu[0][0] = %f\n", c[0][0]);
	printf("c_cpu[0][0] = %f\n", c_cpu[0][0]);
	
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	return 0;
}


