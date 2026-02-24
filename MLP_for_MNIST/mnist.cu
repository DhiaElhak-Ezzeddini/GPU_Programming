#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>


__global__ void forward(int batch_size, int n, int out_w, float *input, float *weights, float *biases, float *output){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row<batch_size && col<out_w){
		output[row*out_w+col] = biases[col];
		for(int i=0;i<n;i++){
			output[row*out_w+col] += input[i*out_w+col] * weights[row*n+i];
		}
	}
}

__global__ void forward(int batch_size, int n, int out_w, float *input, float *weights, float *biases, float *output){
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row<batch_size && col<out_w){
                float out = biases[col];
                for(int i=0;i<n;i++){
                        out += input[i*out_w+col] * weights[row*n+i];
                }
		output[row*out_w+col] = out > 0.f? out : 0.f; 
        }
}


__global__ void relu(int w, int h, float *input, float *output){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y; //each thread calculate one element in the output matrix

	if(row<h && col<w){
		float activ = input[row*w+col];
		output[row*w+col] = activ > 0.f? activ : 0.f;
	}
}

__global__ void softmax(int w, int h, float *input, float *output){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(row<h && col<w){
		float maxval = input[row*w];
		for(int i=0;i<w;i++){
			maxval = max(maxval, input[row*w+i]);
		}
		float divisor = 0.f;
		for(int i=0; i<w; i++){
			divisor += exp(input[row*w+i]-maxval);
		}
		output[row*w+col] = exp(input[row*w+col]-maxval) / (divisor) ; 
	}
}

//Loss Function

__global__ void loss(int w, int h, float *preds, float *real, float *output){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<h){
		float loss = 0.f;
		for(int i=0;i<w;i++){ // w = 10 MNIST integers, the loss is computed over all the numbers 
			loss -= real[idx*w+i] * log(max(1e-6,preds[idx*w+i])); // addding the 1e-6 for numerical stability
		}
		output[idx] = loss;
	}
}


__global__ void init_rand(int w, int h, float *mat){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row<h && col<w){
		curandState state; 
		curand_init(42,row*w+col,0,&state);
		mat[row*w+col] = curand_normal(&state)*sqrtf(2.f/h);
	}
}
//backward pass

__global__ void cross_entropy_loss(int w, int h, float *preds, float *real, float *output){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row<h && col<w){
		output[row*w+col] = preds[row*w+col] - real[row*w+col];
	}
}

__global__ void backward(int batch_size, int n, int out_w, float *weights, float *biases, float *d_l, float *out_d_l){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(row<batch_size && col<out_w){
		float dl=0.0f;
		for(int i=0;i<n;i++){
			float w = weights[i*out_w+col];
			dl += w * d_l[row*n+i];
		}
		out_d_l[row*out_w+col] = dl; 
	}
}


__global__ void relu_backward(int w, int h, int ns, float *a, float *d_l, float* b){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(row<h && col<w){
		float activation = a[row*w+col];
		b[row*w+col] = activation > 0.f ? d_l[row*w+col]: 0.f;
	}
}


__global__ void update_layer(int w, int h, int batch_size, float lr, float *weights, float *biases,
							float *activations, float *d_l){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<h && col<w){
		float dw = 0.f;
		float db = 0.f;
		for(int i=0; i<batch_size;i++){
			float act = activations[i*w+col];
			float dl = d_l[i*w+col];
			dw += act*dl;
			db += dl;
		}
		weights[row*w+col] -= lr*dw/batch_size;
		biases[row*w+col]  -= lr*db/batch_size;
	}
}