#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main(int argc, char **argv){

     // How many quantum gate matrices we will have? variable or fixed?
     // Six - fixed
     int NUM_QUANTUM_GATES = 6;
     int QUANTUM_GATE_SIZE = 4;

     float gates[NUM_QUANTUM_GATES][QUANTUM_GATE_SIZE];
                
     int qgate_elems_size = NUM_QUANTUM_GATES * QUANTUM_GATE_SIZE;
     
     FILE* in_file = fopen(argv[1], "r"); // read only  
     float input_elem; 
     for (int i=0; i < qgate_elems_size; i++){
        fscanf(in_file, "%f", &input_elem);
        if (input_elem == EOF){
            printf("Incorrect input formatting. Abort\n");
            return 1;
        }
        
        printf("%0.2f\n", input_elem);
     }
    
    return 0;
}
