#include <stdio.h>
#include <vector>
#include <fstream>
#include <string>
#include <string.h>
#include <fstream>
#include <sstream>
#include <map>



static void load_profiled_addressess(){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen("refcount.txt", "r");
    if (fp == NULL){
        printf("Could not read input file. Turned on profiling\n");
        return;
    }

    char *token;
    int wordcount=0, count=0;
    char *placeholders[5]; //just an estimate

    //std::map<std::pair<int, unsigned long long>, int> sm_ref_holder;

    int which_sm=NULL;
    while ((read = getline(&line, &len, fp)) != -1) {
        //line length =, read);
        token = strtok(line, " \n\t");
        /* While there are tokens in "string" */
        while( token != NULL ) {
            placeholders[count] = token;
            /* Get next token: */
            token = strtok( NULL, " \n\t");
            count++;
        }
        wordcount=count;
        count=0;

        // if c == 3, this contains kernel, address, and count
        // if c == 2, this contains SM
        // if c == 1, this contains the end of the SM
        if (wordcount == 3){
            //continue loading data into here

            printf("%d ", atoi(placeholders[0]));    //kernel
            printf("%llu ", atoll(placeholders[1]));  //address ref 
            printf("%d \n", atoi(placeholders[2]));    //count
            /*
            sm_ref_holder[std::make_pair(
                    atoi(placeholders[0]),
                    atoll(placeholders[1])
                    )] = atoi(placeholders[2];
                    */
        } else if (wordcount == 1){
            //stop loading data
            //find and copy over to the SM
            //
        } else if (wordcount == 2){
           //get the SM number
           which_sm = atoi(placeholders[1]);  //get SM
           printf("SM: %d\n", which_sm);
        }
    }

    fclose(fp);
    if (line){
        free(line);
    }
}

int main() {

    load_profiled_addressess();
    return 0;
}

