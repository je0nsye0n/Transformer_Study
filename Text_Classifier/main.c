#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "model.h"
#include "header.h"
#define MAX_LEN 100 // 각 리뷰의 최대 길이
#define VOCAB_SIZE 2000 // 허용할 단어의 최대 개수
#define NUM_CLASSES 2

Dataset load_dataset(const char *file_path, int max_len) {
    Dataset dataset;
    dataset.size = 0;

    FILE *file = fopen(file_path, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), file)) {
        dataset.size++;
    }

    dataset.data = malloc(dataset.size * sizeof(int *));
    dataset.labels = malloc(dataset.size * sizeof(int));

    rewind(file);
    int idx = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        dataset.data[idx] = malloc(max_len * sizeof(int));
        char *token = strtok(buffer, ",");
        int word_idx = 0;

        while (token && word_idx < max_len) {
            dataset.data[idx][word_idx++] = atoi(token);
            token = strtok(NULL, ",");
        }

        while (word_idx < max_len) {
            dataset.data[idx][word_idx++] = 0;
        }

        if (token) {
            dataset.labels[idx] = atoi(token);
        }
        idx++;
    }

    fclose(file);
    return dataset;
}

void free_dataset(Dataset dataset) {
    for (int i = 0; i < dataset.size; i++) {
        free(dataset.data[i]);
    }
    free(dataset.data);
    free(dataset.labels);
}

void normalize_dataset(Dataset *dataset, int max_value) {
    for (int i = 0; i < dataset->size; i++) {
        for (int j = 0; j < MAX_LEN; j++) {
            if (dataset->data[i][j] > 0) { // 0이 아닌 값만 정규화
                dataset->data[i][j] = (float)dataset->data[i][j] / max_value;
            } else {
                dataset->data[i][j] = 0.0f; // 0인 경우 유지
            }
        }
    }
}


Dataset train_data, test_data;

void dataload() {
    train_data = load_dataset("data/train_data.csv", MAX_LEN);
    test_data = load_dataset("data/test_data.csv", MAX_LEN);

    normalize_dataset(&train_data, VOCAB_SIZE - 1);
    normalize_dataset(&test_data, VOCAB_SIZE - 1);


    printf("Train Dataset Size: %d\n", train_data.size);
    printf("Test Dataset Size: %d\n", test_data.size);

    printf("First Train Sample (Padded):\n");
    for (int i = 0; i < MAX_LEN; i++) {
        printf("%d ", train_data.data[0][i]);
    }
    printf("\nLabel: %d\n", train_data.labels[0]);
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

void clear() {
    free_dataset(train_data);
    free_dataset(test_data);
}

int main() {

    // DataLoader
    dataload();

    // model
    fclayer *model = classifier(train_data, test_data);
    
    train_model(model, train_data, 10, 0.0001);
    float accuracy = evaluate_model(model, test_data);
    printf("Test Accuracy: %.4f\n", accuracy);
    // clear
    clear();
    return 0;
}
