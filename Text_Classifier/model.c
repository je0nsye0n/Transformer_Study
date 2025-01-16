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

fclayer *create_fclayer(int input_dim, int output_dim) {
    fclayer *layer = malloc(sizeof(fclayer));
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->w = malloc(output_dim * sizeof(float *));
    
    srand((unsigned int)time(NULL)); // 랜덤 초기화


    for (int i = 0; i < output_dim; i++) {
        layer->w[i] = malloc(input_dim * sizeof(float));
        for(int j=0; j<input_dim; j++){ // 가중치 랜덤 초기화
            layer->w[i][j] = ((float)rand()/RAND_MAX) * 0.01;
            //printf("weight : %.3f\n", layer->w[i][j]);
        }
    }

    layer->b = malloc(output_dim * sizeof(float));
    for(int i=0; i<output_dim; i++){
        layer->b[i] = 0.0f; // 편향은 0으로 초기화
    }

    return layer;
}

float *forward_fc(fclayer *layer, float *input) {
    float *output = malloc(layer->output_dim * sizeof(float));
    for (int i = 0; i < layer->output_dim; i++) {
        output[i] = layer->b[i];
        for (int j = 0; j < layer->input_dim; j++) {
            output[i] += input[j] * layer->w[i][j];
        }
    }
    return output;
}


fclayer *classifier(Dataset train_data, Dataset test_data) {

    fclayer *model = create_fclayer(MAX_LEN, NUM_CLASSES);
    return model;
}
