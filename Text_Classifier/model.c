#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "model.h"

// 파라미터 설정
#define MAX_LEN 100
#define NUM_CLASSES 2
#define vocal_size 2000
#define embedding_dim = 32
#define dff = 64
int input_dim = MAX_LEN, output_dim = NUM_CLASSES;
transLayer *layer;
float *input, *output;

transLayer *TokenAndPositionEmbedding(){

}

transLayer *TransformerBlock(){

}

transLayer *Pooling(){

}

float *Fully_connected(transLayer *layer, float *input){
    // 계산
    output = (float*)malloc(sizeof(float)*output_dim);
    for(int i=0; i<output_dim; i++){
         output[i] = layer->b[i];
        for(int j=0; j<input_dim; j++){
            output[i] += input[j] * layer->w[i][j];
        }
    }

    return output;
}

transLayer *Dropout(){

}

float *classifier(transLayer *layer, float *data) {
    input = data;
    output = Fully_connected(layer, data);
    return output;
}
