/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

const int fragment_size = 64;

bool in_array(int key, int *array, int size){
    /*
     Checks if key is in the array. Naive implementation
     */

    for (int i=0; i < size; i++){
        if (key == array[i]){
            return true;
        }
    }

    return false;
}

int bit_at_position(int number, int position){
    /* Given a number, it returns the bit at that position */
    int mask = 1 << position;
    if (number & mask){
        return 1;
    }
    return 0;
}

/**
 * CUDA Kernel Device code
 *
 * Performs application of quantum gate to the quantum state supplied in
 * A. C contains the state after the application.
 */
__global__ void matrix_mul(float *A, int *B, float *C, float *gates, int state_size, int *inactive_bits, int inactive_bit_count){

    //int i = blockDim.x * blockIdx.x + threadIdx.x;
    int i = threadIdx.x;

    //copy into shared memory from global
    __shared__ float S_A[fragment_size];
    int filled = 0;
    for (int j=0; j < state_size; j++){
        if (B[j] == blockIdx.x){
            S_A[filled] = A[j];
            filled++;
        }
    }

    float a, b, c, d;
    //the matrix multiplication code: we find the pair
    //that will work together
    //i=x1, flipped=x2
    int round = 0;
    while (round < 6){
        a = gates[(round * 4) + 0];
        b = gates[(round * 4) + 1];
        c = gates[(round * 4) + 2];
        d = gates[(round * 4) + 3];
        

        int flipped = i ^ (1 << round);
        if (i < fragment_size){
            if (flipped  > i){
                float s_a_i, s_a_flipped;
                s_a_i = S_A[i];
                s_a_flipped = S_A[flipped];
                S_A[i] = (s_a_i * a ) + (s_a_flipped * b);
                S_A[flipped] = (s_a_i * c ) + (s_a_flipped * d);
            }
        }
        ++round;
    }

    //copy data out of shared memory to global memory
    __syncthreads();
    filled = 0;
    for (int j=0; j < state_size; j++){
        if (B[j] == blockIdx.x){
            C[j] = S_A[filled];
            filled++;
        }
    }
}


/**
 * Host main routine
 */

int main(int argc, char **argv){
    /////////////////////////////////////////////////////////////////
    //                    Reading From Input                       //
    /////////////////////////////////////////////////////////////////

    FILE* in_file = fopen(argv[1], "r");                   // read only
    // equivalent to saying if ( in_file == NULL )
     if (!in_file){
        printf("oops, file can't be read\n");
        exit(-1);
     }

    int NUM_QUANTUM_GATES = 6;
    int QUANTUM_GATE_SIZE = 4;
    float gates[NUM_QUANTUM_GATES * QUANTUM_GATE_SIZE];
    int T_BITS[NUM_QUANTUM_GATES];

    // Read and store the quantum gate matrices in a 2-D array called gates
    // Each matrix is reprsented like this:
    // [a,b,c,d] = a  b
    //             c  d
    char input_elem[32];                                   // arbitrary length
    for (int j=0; j < NUM_QUANTUM_GATES * QUANTUM_GATE_SIZE; j++){
        int r = fscanf(in_file, "%s", &input_elem[0]);
        if (r == EOF){
            printf("Incorrect input formatting. Abort\n");
            return 1;
        }
        gates[j] = atof(input_elem);
    }

    // Read the rest of the file
    int max_vector_size=pow(2,30);
    float* state_vector  = (float*)malloc(max_vector_size * sizeof(float));
    int count = 0;
    while (fscanf(in_file, "%s", &input_elem[0]) != EOF){
        state_vector[count] = atof(input_elem);
        count++;
    }

    // Go to the back of the state_vector, and grabs the
    // last NUM_QUANTUM_GATES elements, convert them into ints
    // and store them in an array
    int tmp;
    for (int i=0; i<NUM_QUANTUM_GATES; i++){
        tmp =  (int)state_vector[count-i-1];
        T_BITS[NUM_QUANTUM_GATES - i - 1] = tmp;
    }

    /////////////////////////////////////////////////////////////////
    //                    Kernel Code                              //
    /////////////////////////////////////////////////////////////////

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = count - NUM_QUANTUM_GATES;
    size_t size = numElements * sizeof(float);
    size_t b_size = numElements * sizeof(int);

    // Allocate the host input vector A - contains states
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B - contains blockID assignments
    int *h_B = (int *)malloc(b_size);

    // Allocate the host output vector C - contains results
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_C == NULL || h_B == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i){
        h_A[i] = state_vector[i];
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Memory corruption #1\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Allocate the device input vector B
    int *d_B = NULL;
    err = cudaMalloc((void **)&d_B, b_size);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Actual size of n - the n-qubit state
    int n = (int)log2((float)numElements);

    // Extrapolate the inactive bits from the list of active bits
    int num_active_bits = 6; // God this is hardcoded
    int inactive_bits[n - num_active_bits];
    int inactive_bit_count = 0;
    for (int i=0; i <n; i++){
        if (!in_array(i, T_BITS, num_active_bits)){
            inactive_bits[inactive_bit_count] = i;
            inactive_bit_count++;
        }
    }

    // Allocate memory for gates on devices
    float *gates_device = NULL;
    size_t gates_size = NUM_QUANTUM_GATES * QUANTUM_GATE_SIZE * sizeof(float);
    err = cudaMalloc((void **)&gates_device, gates_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate gates array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(gates_device, gates, gates_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy gates from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the array for inactive bits
    int *inactive_bits_device = NULL;
    err = cudaMalloc((void **)&inactive_bits_device, inactive_bit_count);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate inactive bits array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaMemcpy(inactive_bits_device, inactive_bits, inactive_bit_count, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Calculate blockID assignments using the number of state vectors
    // TODO: make sure you have the size right
    // TODO: make sure the inactive bit array is in sorted order (0..n)
    for (int i=0; i < numElements; i++){
       int block_id = 0;
       for (int ib=0; ib < inactive_bit_count; ib++){
           int cib = inactive_bits[ib];
           int b = bit_at_position(i, cib);

           block_id = block_id << 1;

           if (b==1){    // setting bit according to the original number
               block_id |= 1;
           } else {
               block_id &= ~1;
           }
       }
       h_B[i] = block_id;
    }

    err = cudaMemcpy(d_B, h_B, b_size, cudaMemcpyHostToDevice);
    printf("Memory corruption #2\n");

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = fragment_size / 2; //2^5 (not 2^6)
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    matrix_mul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, gates_device, numElements, inactive_bits_device, inactive_bit_count);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        printf("%.3f\n", h_C[i]);
    }


    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    //Free IO memory
    free(state_vector);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

