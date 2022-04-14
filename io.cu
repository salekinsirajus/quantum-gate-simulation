#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main(int argc, char **argv){
     // How many quantum gate matrices we will have? variable or fixed?
     // Six - fixed
     int NUM_QUANTUM_GATES = 6;
     int QUANTUM_GATE_SIZE = 4;
     float gates[NUM_QUANTUM_GATES][QUANTUM_GATE_SIZE];
     
     FILE* in_file = fopen(argv[1], "r");                   // read only  
     char input_elem[32];                                   // arbitrary length

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

    //reading from the entered data
    for (int i=0; i < NUM_QUANTUM_GATES; i++){
        for (int j=0; j < QUANTUM_GATE_SIZE; j++){
            printf("%0.3f \n", gates[i][j]);
        }
        printf("\n");
    }

    return 0;
}
