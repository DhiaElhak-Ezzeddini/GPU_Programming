#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 256
#define TILE_WIDTH 16 

__global__ void tiled_matmul(int n, float *a, float*b, float *c){
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float dot_prod = 0.f;
    for(int tile_offset=0; tile_offset<n; tile_offset+=TILE_WIDTH){
        int a_chk = tile_offset+tx<n && row<n;
        tile_a[tx][ty] = a_chk? a[row*n+tile_offset+tx] : 0.f;
        int b_chk = tile_offset+ty<n && col<n;
        tile_b[tx][ty] = b_chk? b[(tile_offset+ty)*n+col] : 0.f;

        __syncthreads(); // synchronize all threads since it is not guaranteed that all threads will reach this point at the same time
        for(int i=0; i<TILE_WIDTH;i++){
            dot_prod += tile_a[ty][i] * tile_b[i][tx];
        }
        __syncthreads();
    }
    if(row<n && col<n){
        c[row*n+col] = dot_prod;
    }
}


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
	static float a[N][N], b[N][N], c[N][N], c_tiled[N][N], c_cpu[N][N];

	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			a[i][j]=1.0f;
			b[i][j]=1.0f;
		}
	}
	float *a_d, *b_d, *c_d, *c_d_tiled;
	cudaMalloc(&a_d, size);
	cudaMalloc(&b_d, size);
	cudaMalloc(&c_d, size);
	cudaMalloc(&c_d_tiled, size);

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


	float gpu_tiled_time = 0;

	cudaEventRecord(start);
	tiled_matmul<<<grid,block>>>(N,a_d,b_d,c_d_tiled);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpu_tiled_time,start,stop);

	printf("#### GPU Tiled Time : %f ms ####\n",gpu_tiled_time);

	cudaMemcpy(c_tiled,c_d_tiled,size,cudaMemcpyDeviceToHost);


	// CPU TIMING
	auto start_cpu = std::chrono::high_resolution_clock::now();
	matmul_cpu(N,a,b,c_cpu);
	auto end_cpu = std::chrono::high_resolution_clock::now();

	double cpu_time = std::chrono::duration<double,std::milli>(end_cpu - start_cpu).count();
	printf("#### CPU Time : %f ms ####\n",cpu_time);



	printf("c_gpu[0][0] = %f\n", c[0][0]);
	printf("c_gpu_tiled[0][0] = %f\n", c_tiled[0][0]);
	printf("c_cpu[0][0] = %f\n", c_cpu[0][0]);
	
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	return 0;
}


