#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

// Kernel for Bubble Sort
__global__ void bubbleSortKernel(int *d_arr, int n) {
    int temp;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread handles a single bubble sort pass
    for (int i = 0; i < n - 1; i++) {
        if (idx < n - 1 - i) {
            if (d_arr[idx] > d_arr[idx + 1]) {
                // Swap
                temp = d_arr[idx];
                d_arr[idx] = d_arr[idx + 1];
                d_arr[idx + 1] = temp;
            }
        }
        __syncthreads(); // Synchronize threads after each pass
    }
}

// Device function for merging arrays
__device__ void merge(int *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

// Kernel for Merge Sort
__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n) {
    for (int size = 1; size < n; size *= 2) {
        int left = 0;
        while (left + size < n) {
            int mid = left + size - 1;
            int right = min(left + 2 * size - 1, n - 1);

            merge(d_arr, left, mid, right); // Call device merge
            left += 2 * size;
        }
        // Copy merged result back to d_arr
        for (int i = 0; i < n; i++) {
            d_temp[i] = d_arr[i];
        }
        for (int i = 0; i < n; i++) {
            d_arr[i] = d_temp[i];
        }
    }
}

// Host function for merging arrays
void mergeHost(int *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

// Bubble Sort on GPU
void bubbleSort(int *arr, int n) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Start GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bubbleSortKernel<<<1, n>>>(d_arr, n);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);
}

// Merge Sort on GPU
void mergeSort(int *arr, int n) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Start GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mergeSortKernel<<<1, 1>>>(d_arr, d_temp, n);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);
}

// Bubble Sort on CPU
void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap
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

// Merge Sort on CPU
void mergeSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int size = 1; size < n; size *= 2) {
        int left = 0;
        while (left + size < n) {
            int mid = left + size - 1;
            int right = min(left + 2 * size - 1, n - 1);

            mergeHost(arr, left, mid, right); // Call host merge
            left += 2 * size;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Merge Sort (CPU) took %f milliseconds\n", duration.count());
}

// Print array
void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

// Main function
int main() {
    int n = 10000; // Increase for larger datasets
    int *arr = (int*)malloc(n * sizeof(int));

    // Generating random array
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    printf("Original array: \n");
    printArray(arr, n);

    // Bubble Sort CPU
    bubbleSortCPU(arr, n);
    printf("Sorted array using Bubble Sort (CPU): \n");
    printArray(arr, n);

    // Generating random array again for GPU
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    // Bubble Sort GPU
    bubbleSort(arr, n);
    printf("Sorted array using Bubble Sort (GPU): \n");
    printArray(arr, n);

    // Generating random array again for Merge Sort
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    printf("Original array: \n");
    printArray(arr, n);

    // Merge Sort CPU
    mergeSortCPU(arr, n);
    printf("Sorted array using Merge Sort (CPU): \n");
    printArray(arr, n);

    // Generating random array again for GPU
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    // Merge Sort GPU
    mergeSort(arr, n);
    printf("Sorted array using Merge Sort (GPU): \n");
    printArray(arr, n);

    free(arr);
    return 0;
}
