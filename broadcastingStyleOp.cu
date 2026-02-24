#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define Nx 256
#define Ny 256
#define Nz 256


__global__ void add_3d_2d_1d(
	int nx, int ny, int nz,
	float *a,float *b,float *c,float *out_arr
){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if(x<nx && y<ny && z<nz){
		int idx3d = x*ny*nz + y*nz + z;
		int idx2d = x*ny +y; 
		int idx1d = x; 
		out_arr[idx3d] = a[idx3d] + b[idx2d] + c[idx1d];
	}
}

void add_cpu(float *a_h,float *b_h,float *c_h,float *ref_h){
	for(int x=0;x<Nx;x++){
		for (int y=0;y<Ny;y++){
			for(int z=0;z<Nz;z++){
				int idx3d = x*Ny*Nz + y*Nz + z;
				int idx2d = x*Ny +y; 
				int idx1d = x; 

				ref_h[idx3d] = a_h[idx3d] + b_h[idx2d] + c_h[idx1d];
			}
		}
	}
}
		

int main(){

	int N3 = Nx*Ny*Nz;
	int N2 = Nx*Ny;
	int N1 = Nx;
	size_t size3 = N3*sizeof(float);
	size_t size2 = N2*sizeof(float);
	size_t size1 = N1*sizeof(float);

	float *a_h = (float*)malloc(size3);
	float *b_h = (float*)malloc(size2);
	float *c_h = (float*)malloc(size1);
	float *out_h = (float*)malloc(size3);
	float *ref_h = (float*)malloc(size3);

	for(int i=0;i<N3;i++) a_h[i]=1.0f;
	for(int i=0;i<N2;i++) b_h[i]=2.0f;
	for(int i=0;i<N1;i++) c_h[i]=3.0f;


	float *a_d,*b_d,*c_d,*out_d;
	cudaMalloc(&a_d,size3);
	cudaMalloc(&b_d,size2);
	cudaMalloc(&c_d,size1);
	cudaMalloc(&out_d,size3);
	
	cudaMemcpy(a_d,a_h,size3,cudaMemcpyHostToDevice);
	cudaMemcpy(b_d,b_h,size2,cudaMemcpyHostToDevice);
	cudaMemcpy(c_d,c_h,size1,cudaMemcpyHostToDevice);
	
	dim3 block(8,8,8);
	dim3 grid(
			(Nx + block.x -1 )/block.x,
			(Ny + block.y -1 )/block.y,
			(Nz + block.z -1 )/block.z);
	
	//Timing 
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	add_3d_2d_1d<<<grid, block>>>(Nx,Ny,Nz,a_d,b_d,c_d,out_d);
	cudaEventRecord(stop);

	cudaMemcpy(out_h,out_d,size3,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	float ms =0.0f;
	cudaEventElapsedTime(&ms,start,stop);
	printf("#### GPU Time : %f ms ####\n", ms);

	// CPU implementation 
	// CPU TIMING
	auto start_cpu = std::chrono::high_resolution_clock::now();
	
	add_cpu(a_h,b_h,c_h,ref_h);

	auto end_cpu = std::chrono::high_resolution_clock::now();

	double cpu_time = std::chrono::duration<double,std::milli>(end_cpu - start_cpu).count();
	printf("#### CPU Time : %f ms ####\n",cpu_time);

	//cleanup

	free(a_h);free(b_h);free(c_h);free(out_h);free(ref_h);
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	cudaFree(out_d);

	return 0;
}

