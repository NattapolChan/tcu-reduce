#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <cub/cub.cuh>


using namespace nvcuda;

// --- CUDA Error Checking ---
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error at %s:%d\n", file, line);
        fprintf(stderr, "└ %s in function: %s\n", cudaGetErrorString(err), func);
	exit(1);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error at %s:%d\n", file, line);
        fprintf(stderr, "└ %s\n", cudaGetErrorString(err));
	exit(1);
    }
}

#define WMMA_SIZE 16
#define THREADS_PER_BLOCK 128
#define WARPS_PER_BLOCK ((THREADS_PER_BLOCK / 32))

__global__ void wmma_reduction_kernel(half *input, float* output, int N) {
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int offset = warp_idx* 256;

    if (offset >= N) return;

    extern __shared__ float smem[];
    float* local_smem = &smem[threadIdx.x/32 * 256];

    wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE,
        half, wmma::row_major> local_a;
    wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE,
        half, wmma::col_major> local_b;
    wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE,
        float> local_out;

    wmma::fill_fragment(local_out, 0.);

    for (int i=0;i<local_a.num_elements;i++) local_a.x[i] = __float2half(1.);
    wmma::load_matrix_sync(local_b, input+offset, 16);

    // C ik = SUMj(A ij . B jk)
    wmma::mma_sync(local_out, local_a, local_b, local_out);

    wmma::store_matrix_sync(local_smem, local_out, 16, wmma::mem_row_major);

    __syncwarp();

    float sum = 0.;
    int lane = threadIdx.x % 32;
    if (lane < 16) sum = local_smem[lane];

    for (int stride = 8; stride>0;stride>>=1) { sum += __shfl_down_sync(0xFFFFFFFF, sum, stride); }

    if (lane==0) atomicAdd(output, sum);
}

__global__ void cub_reduction_kernel(__half* d_input, float* d_results, int N) {
    using BlockReduce = cub::BlockReduce<__half, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage smem_reduce;

    if (threadIdx.x >= THREADS_PER_BLOCK) return;

    __half data = 0.;
    if (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x < N) {
	    data = d_input[blockIdx.x * THREADS_PER_BLOCK + threadIdx.x];
    }
    float aggregate = (float)BlockReduce(smem_reduce).Sum(data);
    if (threadIdx.x == 0) atomicAdd(d_results, aggregate);
}

int main() {
    const int N = 1 << 29;
    int bytes = N * sizeof(half);

    printf("Tensor Core Reduction with N = %d\n", N);
    
    std::vector<half> h_input(N);
    for (int i=0;i<N;i++) h_input[i] = __float2half(1.);
    half *d_input;
    float *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0., sizeof(float)));

    int total_warps = (N + 255) / 256;
    int total_blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    int shared_bytes = WARPS_PER_BLOCK * 256 * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    wmma_reduction_kernel<<<total_blocks, THREADS_PER_BLOCK, shared_bytes>>>(
        d_input, d_output, N);

    CHECK_LAST_CUDA_ERROR();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float d_results_tcu_reduce = 0.;
    float ms_tcu_reduce = 0.;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_tcu_reduce, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_results_tcu_reduce, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, sizeof(float)));

    cudaDeviceSynchronize();

    cudaEvent_t start_cub, stop_cub;
    cudaEventCreate(&start_cub);
    cudaEventCreate(&stop_cub);

    cudaEventRecord(start_cub);
    cub_reduction_kernel<<<((N - 1 + THREADS_PER_BLOCK) / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
	d_input, d_output, N);
    CHECK_LAST_CUDA_ERROR();
    cudaEventRecord(stop_cub);
    cudaEventSynchronize(stop_cub);
    float ms_cub = 0.;
    float d_results_cub = 0.;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_cub, start_cub, stop_cub));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_results_cub, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    printf("\t\tTCU Reduce\t\tCUB\n");
    printf("Time:\t\t%f ms\t\t%f ms\n", ms_tcu_reduce, ms_cub);
    printf("Throughput:\t%f GB/s\t\t%f GB/s\n",
	(float) (bytes*1000. / (ms_tcu_reduce) / 1e9),
	(float) (bytes*1000. / (ms_cub) / 1e9)
	);
    printf("Results:\t%f \t%f (expected %f)\n", d_results_tcu_reduce, d_results_cub, (float)N);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
