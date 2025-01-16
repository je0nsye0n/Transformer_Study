#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#define MAX_LEN 100 // 각 리뷰의 최대 길이
#define VOCAB_SIZE 2000 // 허용할 단어의 최대 개수

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

void clear() {
    free_dataset(train_data);
    free_dataset(test_data);
}

int main() {
    dataload();
    model_fc(train_data, test_data);
    clear();
    return 0;
}
