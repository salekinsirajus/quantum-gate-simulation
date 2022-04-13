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

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
// A contains input
__global__ void matrix_mul(float *A, float *C, float a, float b, float c, float d, int state_size, int t_bit){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    //i=x1, flipped=x2
    int flipped = ((1 << t_bit) | i);
    if (i < state_size){
        if (flipped  > i){
            C[i] = (A[i] * a ) + (A[flipped] * b);
            C[flipped] = (A[i] * c ) + (A[flipped] * d);
        }
    }
   
}


/**
 * Host main routine
 */
int
main(int argc, char **argv)
{
    /*
        INPUT READING FROM FILE
    */
     float a, b, c, d; // quantum gate matrix 
     int t_bit; 
     int vector_size=pow(2,30), final_size;
     float*  state_vector  = (float*)malloc(vector_size * sizeof(float));

     float number;
      FILE* in_file = fopen(argv[1], "r"); // read only  
     
      if (! in_file ) // equivalent to saying if ( in_file == NULL ) 
         {  
            printf("oops, file can't be read\n"); 
            exit(-1); 
         } 

      // attempt to read the next line and store 
      // the value in the "number" variable 
      int count = 0;
      float last;
      while ( fscanf(in_file, "%f", &number ) != EOF )  
         { 
           if (count==0) a = number;
           else if (count==1) b = number;
           else if (count==2) c = number;
           else if (count==3) d = number;

           else {
               state_vector[count-4] = number; 
            }

           last = number;
           count++;
         } 
           
     t_bit = (int) last;
     final_size = count - 5;
    //printf("Qubit %d size: %d a:%f, b:%f c:%f d:%f\n", qubit, final_size, a, b, c, d);
    /* END READING FROM INPUT */


    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = final_size;
    size_t size = numElements * sizeof(float);
    //printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = state_vector[i]; 
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMallocManaged((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMallocManaged((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    matrix_mul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, a, b, c, d, numElements, t_bit);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
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


    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
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

