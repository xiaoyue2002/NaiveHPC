// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"

namespace {

#define INF 200000
#define batch_n 8
#define batch_n_2 6

__device__ int read_block_mem(int n, int * graph, int i, int j){
	if (i < n && j < n)
		return graph[i * n + j];
	else
		return INF;
}

__device__ void write_block_mem(int n, int * graph, int i, int j, int value){
	if (i < n && j < n){
		graph[i * n + j] = value;
	}
}

// phase 1 : 计算对角线上的块
__global__ void phase_1(int n, int s, int *graph) {
	__shared__ int shared_block[32][32 + 1];
	int read_i = 32 * s + threadIdx.y;
	int read_j = 32 * s + threadIdx.x;
	shared_block[threadIdx.y][threadIdx.x] = read_block_mem(n, graph, read_i, read_j);
	__syncthreads();
	
	int temp_distance = shared_block[threadIdx.y][threadIdx.x];
	for (int k = 0 ; k < 32 ; k ++){
		temp_distance = min(temp_distance, shared_block[threadIdx.y][k] + shared_block[k][threadIdx.x]);
	}
	write_block_mem(n, graph, s * 32 + threadIdx.y, s * 32 + threadIdx.x, temp_distance);
}

__global__ void phase_1_quick_read(int n, int s, int * graph){
    __shared__ int shared_block[32][32];
    int read_i = 32 * s + threadIdx.y;
    int read_j = 32 * s + threadIdx.x;
    shared_block[threadIdx.y][threadIdx.x] = graph[read_i * n + read_j];
    __syncthreads();
    
    int temp_distance = shared_block[threadIdx.y][threadIdx.x];
    for (int k = 0 ; k < 32 ; k ++){
        temp_distance = min(temp_distance, shared_block[threadIdx.y][k] + shared_block[k][threadIdx.x]);
    }
    graph[read_i * n + read_j] = temp_distance;
}

// phase 2 : 计算十字线上的块
__global__ void phase_2(int n, int s, int * graph){
	__shared__ int shared_block[32][32 + 1];
	__shared__ int cross_block[batch_n][32][32 + 1];

	// read shared block
	shared_block[threadIdx.y][threadIdx.x] = read_block_mem(n, graph, s * 32 + threadIdx.y , s * 32 + threadIdx.x);
	
	int temp_distance;
	if (blockIdx.y == 0){ // vertical line 
		int block_i = blockIdx.x * batch_n * 32;
		#pragma unroll
		for (int p = 0, thread_i = block_i + threadIdx.y ; p < batch_n ; p++, thread_i += 32){
			cross_block[p][threadIdx.y][threadIdx.x] = read_block_mem(n, graph, thread_i, s * 32 + threadIdx.x);
		}
		__syncthreads();
		for (int p = 0, thread_i = block_i + threadIdx.y ; p < batch_n ; p++, thread_i += 32){
			temp_distance = cross_block[p][threadIdx.y][threadIdx.x];
			for (int k = 0 ; k < 32 ; k++){
				temp_distance = min(temp_distance, shared_block[k][threadIdx.x] + cross_block[p][threadIdx.y][k]);
			}
			write_block_mem(n, graph, thread_i, s * 32 + threadIdx.x, temp_distance);
		}
	} else { // horizontal line
		int block_j = blockIdx.x * batch_n * 32;	
		#pragma unroll
		for (int p = 0, thread_j = block_j + threadIdx.x ; p < batch_n ; p++, thread_j += 32){
			cross_block[p][threadIdx.y][threadIdx.x] = read_block_mem(n, graph, s * 32 + threadIdx.y, thread_j);
		}
		__syncthreads();
		for (int p = 0, thread_j = block_j + threadIdx.x ; p < batch_n ; p++, thread_j += 32){
			temp_distance = cross_block[p][threadIdx.y][threadIdx.x];
			for (int k = 0 ; k < 32 ; k++){
				temp_distance = min(temp_distance, shared_block[threadIdx.y][k] + cross_block[p][k][threadIdx.x]);
			}
			write_block_mem(n, graph, s * 32 + threadIdx.y, thread_j, temp_distance);
		}
	}
}

// phase_3 计算剩余的块
__global__ void phase_3(int n, int s, int * graph){
	__shared__ int vertical_blocks[batch_n_2][32][32];
	__shared__ int horizontal_blocks[batch_n_2][32][32];
	int thread_i = blockIdx.y * batch_n_2 * 32 + threadIdx.y;
	int thread_j = blockIdx.x * batch_n_2 * 32 + threadIdx.x;
	bool check_border = (s == (n - 1) / 32) || (blockIdx.y >= ((n - 2) / (batch_n_2 * 32))) || (blockIdx.x >=  ((n - 2) / (batch_n_2 * 32)));
	int temp_distance;
	
	if (check_border){
		# pragma unroll
		for (int k = 0 , read_i = thread_i ; k < batch_n_2 ; k++, read_i += 32){
			vertical_blocks[k][threadIdx.y][threadIdx.x] = read_block_mem(n, graph, read_i, s * 32 + threadIdx.x);
		}
		# pragma unroll
		for (int k = 0 , read_j = thread_j ; k < batch_n_2 ; k++, read_j += 32){
			horizontal_blocks[k][threadIdx.y][threadIdx.x] = read_block_mem(n, graph, s * 32 + threadIdx.y, read_j);
		}
	} else {
		# pragma unroll
		for (int k = 0 , read_i = thread_i ; k < batch_n_2 ; k++, read_i += 32){
			vertical_blocks[k][threadIdx.y][threadIdx.x] = graph[read_i * n + s * 32 + threadIdx.x];
		}
		# pragma unroll
		for (int k = 0 , read_j = thread_j ; k < batch_n_2 ; k++, read_j += 32){
			horizontal_blocks[k][threadIdx.y][threadIdx.x] = graph[(s * 32 + threadIdx.y) * n + read_j];
		}
	}
	
	__syncthreads();

	if (check_border){
		for (int p = 0, read_i = thread_i ; p < batch_n_2 ; p++ , read_i += 32){
			for (int q = 0 , read_j = thread_j ; q < batch_n_2 ; q++, read_j += 32){
				temp_distance = read_block_mem(n, graph, read_i, read_j);
				for (int k = 0 ; k < 32; k++){
					temp_distance = min(temp_distance, vertical_blocks[p][threadIdx.y][k] + horizontal_blocks[q][k][threadIdx.x]);
				}
				write_block_mem(n, graph, read_i, read_j, temp_distance);
			}	
		}	
	} else {
		for (int p = 0, read_i = thread_i ; p < batch_n_2 ; p++ , read_i += 32){
			for (int q = 0 , read_j = thread_j ; q < batch_n_2 ; q++, read_j += 32){
				temp_distance = graph[read_i * n + read_j];
				for (int k = 0 ; k < 32; k++){
					temp_distance = min(temp_distance, vertical_blocks[p][threadIdx.y][k] + horizontal_blocks[q][k][threadIdx.x]);
				}
				graph[read_i * n + read_j] = temp_distance;
			}	
		}	
	}
	
}

}

void apsp(int n, /* device */ int *graph) {
	int steps = (n - 1) / 32; // max_step = ( n - 1 ) / 32 + 1;
	dim3 thr(32, 32);
	int k1 = (n - 1 - 1) / (batch_n * 32) + 1;
	int k2 = (n - 1 - 1) / (batch_n_2 * 32) + 1;
	dim3 blk_phase_2(k1, 2); // 分成 k * 2 个线程组
	dim3 blk_phase_3(k2, k2); // 分成 k * k 个线程组
	int s;
	for (s = 0; s < steps; s++)
	{
		phase_1_quick_read<<<1, thr>>>(n, s, graph);
		phase_2<<<blk_phase_2, thr>>>(n, s, graph);
		phase_3<<<blk_phase_3, thr>>>(n, s, graph);
	}
	phase_1<<<1, thr>>>(n, s, graph);
    phase_2<<<blk_phase_2, thr>>>(n, s, graph);
    phase_3<<<blk_phase_3, thr>>>(n, s, graph);
}

