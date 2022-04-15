#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main(int argc, char **argv){
     // How many quantum gate matrices we will have? variable or fixed?
     // Six - fixed
     int NUM_QUANTUM_GATES = 6;
     int QUANTUM_GATE_SIZE = 4;
     float gates[NUM_QUANTUM_GATES][QUANTUM_GATE_SIZE];
     int T_BITS[NUM_QUANTUM_GATES];
     
     FILE* in_file = fopen(argv[1], "r");                   // read only  
     char input_elem[32];                                   // arbitrary length

     // Read and store the quantum gate matrices in a 2-D array called gates
     // Each matrix is reprsented like this:
     //
     // [a,b,c,d] = a  b
     //             c  d
     //
     for (int i=0; i < NUM_QUANTUM_GATES; i++){
        for (int j=0; j < QUANTUM_GATE_SIZE; j++){
            int r = fscanf(in_file, "%s", &input_elem[0]);
            if (r == EOF){
                printf("Incorrect input formatting. Abort\n");
                return 1;
            }
            gates[i][j] = atof(input_elem);
        } 
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

    printf("size of the state vector: %d", count);
    return 0;
}
