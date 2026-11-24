# Exp5 Bubble Sort and Merge sort in CUDA
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```

%%writefile sorting.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

// GPU Bubble Sort Kernel
__global__ void bubbleSortKernel(int *d_arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < n - 1; i++) {
        if (idx < n - 1 - i) {
            if (d_arr[idx] > d_arr[idx + 1]) {
                int temp = d_arr[idx];
                d_arr[idx] = d_arr[idx + 1];
                d_arr[idx + 1] = temp;
            }
        }
        __syncthreads();
    }
}

// GPU Merge Sort Kernel
__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left = idx * (2 * size);
    int mid = left + size;
    int right = left + 2 * size;

    if (left < n) {
        if (mid > n) mid = n;
        if (right > n) right = n;

        int i = left;
        int j = mid;
        int k = left;

        while (i < mid && j < right) {
            if (d_arr[i] <= d_arr[j])
                d_temp[k++] = d_arr[i++];
            else
                d_temp[k++] = d_arr[j++];
        }

        while (i < mid)
            d_temp[k++] = d_arr[i++];

        while (j < right)
            d_temp[k++] = d_arr[j++];
    }
}

// CPU Merge Function
void mergeHost(int *arr, int *temp, int left, int mid, int right) {
    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < mid)
        temp[k++] = arr[i++];

    while (j < right)
        temp[k++] = arr[j++];
}

// CPU Bubble Sort
void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Bubble Sort (CPU) took %f milliseconds\n", duration.count());
}

// CPU Merge Sort
void mergeSortCPU(int *arr, int n) {
    int *temp = (int*)malloc(n * sizeof(int));

    auto start = std::chrono::high_resolution_clock::now();

    for (int size = 1; size < n; size *= 2) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = left + size;
            int right = left + 2 * size;

            if (mid > n) mid = n;
            if (right > n) right = n;

            if (mid < right)
                mergeHost(arr, temp, left, mid, right);
            else {
                for (int i = left; i < right; i++)
                    temp[i] = arr[i];
            }
        }

        for (int i = 0; i < n; i++)
            arr[i] = temp[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Merge Sort (CPU) took %f milliseconds\n", duration.count());

    free(temp);
}

// GPU Bubble Sort
void bubbleSortGPU(int *arr, int n) {
    int *d_arr;

    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    bubbleSortKernel<<<(n + 255) / 256, 256>>>(d_arr, n);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);

    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// GPU Merge Sort
void mergeSortGPU(int *arr, int n) {
    int *d_arr, *d_temp;

    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int size = 1; size < n; size *= 2) {
        int blocks = (n + 2 * size - 1) / (2 * size);
        int threads = 256;

        mergeSortKernel<<<(blocks + threads - 1) / threads, threads>>>(d_arr, d_temp, n, size);

        cudaDeviceSynchronize();

        int *temp = d_arr;
        d_arr = d_temp;
        d_temp = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);

    cudaFree(d_arr);
    cudaFree(d_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Print Array
void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    printf("\n");
}

// Main Function
int main() {
    int n = 10000;

    int *arr = (int*)malloc(n * sizeof(int));

    srand(42);

    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    printf("Original Array:\n");
    printArray(arr, n);

    bubbleSortCPU(arr, n);

    printf("Sorted Array using Bubble Sort (CPU):\n");
    printArray(arr, n);

    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    bubbleSortGPU(arr, n);

    printf("Sorted Array using Bubble Sort (GPU):\n");
    printArray(arr, n);

    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    printf("Original Array:\n");
    printArray(arr, n);

    mergeSortCPU(arr, n);

    printf("Sorted Array using Merge Sort (CPU):\n");
    printArray(arr, n);

    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    mergeSortGPU(arr, n);

    printf("Sorted Array using Merge Sort (GPU):\n");
    printArray(arr, n);

    free(arr);

    return 0;
}

```

## OUTPUT:
<img width="1407" height="360" alt="image" src="https://github.com/user-attachments/assets/4d75ea80-61a2-4526-89b8-2987e4461030" />


## RESULT:
Thus, the program has been executed using CUDA to ________________.
