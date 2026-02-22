#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>


__global__ void add(int n, float* a, float* b, float* c){
	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	if (i<n){
		c[i] = a[i] + b [i];
	}
}

// cpu version
void add_cpu(int n, float* a, float* b, float* c){
	for(int i=0;i<n;i++){
		c[i] = a[i] + b [i];
	}
}

int main(){
	// allocate memory on cpu 
	int N = 2e8 ; 
	size_t size = N*sizeof(float);

	float *a = (float*)malloc(size);
	float *b = (float*)malloc(size);
	float *c_cpu = (float*)malloc(size);
	float *c_gpu = (float*)malloc(size);
	
	// initialize data 
	for (int i=0;i<N;i++){
		a[i] = i; 
		b[i] = 2*i;
	}
	
	// allocate memmory on GPU

	float *a_d;
	float *b_d;
	float *c_d; 
	cudaMalloc((void**)&a_d,N*sizeof(float));
	cudaMalloc((void**)&b_d,N*sizeof(float));
	cudaMalloc((void**)&c_d,N*sizeof(float));

	// move data from CPU to GPU

	cudaMemcpy(a_d,a,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(b_d,b,N*sizeof(float),cudaMemcpyHostToDevice);
	
	// run the kernel, GPU TIMING

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int BLOCK_SIZE = 256; // how many blocks we want to run
        int GRID_SIZE = (N + BLOCK_SIZE -1) / BLOCK_SIZE;	
	cudaEventRecord(start);
	add<<<GRID_SIZE,BLOCK_SIZE>>>(N,a_d, b_d, c_d); 
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float gpu_time = 0;
	cudaEventElapsedTime(&gpu_time,start,stop);

	printf("#### GPU Time : %f ms ####\n",gpu_time);
	// copy back data from GPU to CPU
	cudaMemcpy(c_gpu,c_d,N*sizeof(float),cudaMemcpyDeviceToHost);
	
	// CPU TIMING
	auto start_cpu = std::chrono::high_resolution_clock::now();
	add_cpu(N,a,b,c_cpu);
	auto end_cpu = std::chrono::high_resolution_clock::now();

	double cpu_time = std::chrono::duration<double,std::milli>(end_cpu - start_cpu).count();
	printf("#### CPU Time : %f ms ####\n",cpu_time);
	

	printf("c[0]   = %f\n", c_gpu[0]);
	printf("c[100] = %f\n ", c_gpu[100]);
	
	// Free the memory

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	free(a);
	free(b);
	free(c_cpu);
	free(c_gpu);

	return 0 ; 	

}







