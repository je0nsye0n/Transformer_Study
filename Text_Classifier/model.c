#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "model.h"
#define MAX_LEN 100
#define NUM_CLASSES 2

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

float compute_loss(float *output, int label) {
    float loss = 0.0;
    float sum_exp = 0.0;
    
    for(int i=0; i<NUM_CLASSES; i++){
        sum_exp += exp(output[i]);
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        float softmax = exp(output[i]) / sum_exp;
        float target = (i == label) ? 1.0 : 0.0;
        loss += -target * log(softmax); // 교차 엔트로피    
    }

    return loss / NUM_CLASSES;
}

void backward_and_update(fclayer *layer, float *input, int label, float learning_rate) {
    float sum_exp = 0.0;

    for(int i=0; i<layer->output_dim; i++){
        sum_exp += exp(layer->b[i]);
    }

    for (int i = 0; i < layer->output_dim; i++) {
        float softmax = exp(layer->b[i]) / sum_exp;
        float target = (i == label) ? 1.0 : 0.0;
        float error = softmax - target; // 소프트맥스를 기반으로 오차 계산

        for (int j = 0; j < layer->input_dim; j++) {
            layer->w[i][j] -= learning_rate * error * input[j];
        }
        layer->b[i] -= learning_rate * error;
    }
}

int argmax(float *array, int size) {
    int max_idx = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] > array[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

void train_model(fclayer *model, Dataset train_data, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        for (int i = 0; i < train_data.size; i++) {
            float *input = (float *)train_data.data[i];
            int label = train_data.labels[i];

            float *output = forward_fc(model, input);
            float loss = compute_loss(output, label);
            total_loss += loss;

            backward_and_update(model, input, label, learning_rate);
            free(output);
        }
        printf("Epoch %d/%d, Loss: %.4f\n", epoch + 1, epochs, total_loss / train_data.size);
    }
}

float evaluate_model(fclayer *model, Dataset test_data) {
    int correct = 0;
    for (int i = 0; i < test_data.size; i++) {
        float *input = (float *)test_data.data[i];
        int label = test_data.labels[i];

        float *output = forward_fc(model, input);
        int predicted = argmax(output, NUM_CLASSES);
        if (predicted == label) {
            correct++;
        }
        free(output);
    }
    return (float)correct / test_data.size;
}

void model_fc(Dataset train_data, Dataset test_data) {
    fclayer *model = create_fclayer(MAX_LEN, NUM_CLASSES);
    train_model(model, train_data, 10, 0.0001);

    float accuracy = evaluate_model(model, test_data);
    printf("Test Accuracy: %.4f\n", accuracy);
}
