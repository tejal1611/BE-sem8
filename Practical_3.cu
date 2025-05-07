
#include <iostream>
#include <vector>
#include <climits>

__global__ void min_reduction_kernel(int* arr, int size, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicMin(result, arr[tid]);

    }
}

__global__ void max_reduction_kernel(int* arr, int size, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicMax(result, arr[tid]);
    }
}

__global__ void sum_reduction_kernel(int* arr, int size, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(result, arr[tid]);
    }
}

__global__ void average_reduction_kernel(int* arr, int size, int* sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(sum, arr[tid]);
    }
}

int main() {
    std::vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};
    int size = arr.size();
    int* d_arr;
    int* d_result;
    int result_min = INT_MAX;
    int result_max = INT_MIN;
    int result_sum = 0;

    // Allocate memory on the device
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, arr.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result_min, sizeof(int), cudaMemcpyHostToDevice);

    // Perform min reduction
    min_reduction_kernel<<<(size + 255) / 256, 256>>>(d_arr, size, d_result);
    cudaMemcpy(&result_min, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Minimum value: " << result_min << std::endl;

    // Perform max reduction
    cudaMemcpy(d_result, &result_max, sizeof(int), cudaMemcpyHostToDevice);
    max_reduction_kernel<<<(size + 255) / 256, 256>>>(d_arr, size, d_result);
    cudaMemcpy(&result_max, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Maximum value: " << result_max << std::endl;

    // Perform sum reduction
    cudaMemcpy(d_result, &result_sum, sizeof(int), cudaMemcpyHostToDevice);
    sum_reduction_kernel<<<(size + 255) / 256, 256>>>(d_arr, size, d_result);
    cudaMemcpy(&result_sum, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Sum: " << result_sum << std::endl;

    // Perform average reduction
    cudaMemcpy(d_result, &result_sum, sizeof(int), cudaMemcpyHostToDevice);
    average_reduction_kernel<<<(size + 255) / 256, 256>>>(d_arr, size, d_result);
    cudaMemcpy(&result_sum, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Average: " << static_cast<double>(result_sum) / size << std::endl;

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_result);

    return 0;
}

