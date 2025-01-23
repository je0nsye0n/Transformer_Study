#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define batch_size 1
#define seq_len 10
#define d_model 6
#define header 1

void data_allocate(float ****data) {
    *data = (float ***)malloc(batch_size * sizeof(float **));
    if (*data == NULL) {
        perror("Failed to allocate memory for data");
        return;
    }

    for (int i = 0; i < batch_size; i++) {
        (*data)[i] = (float **)malloc(seq_len * sizeof(float *));
        if ((*data)[i] == NULL) {
            perror("Failed to allocate memory for data[i]");
            return;
        }

        for (int j = 0; j < seq_len; j++) {
            (*data)[i][j] = (float *)malloc(d_model * sizeof(float));
            if ((*data)[i][j] == NULL) {
                perror("Failed to allocate memory for data[i][j]");
                return;
            }
        }
    }
}

void data_load(const char *filename, float ***data) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < d_model; k++) {
                fscanf(file, "%f", &data[i][j][k]);
            }
        }
    }

    fclose(file);
}

/* 디버깅-검증용 */
void print_tensor(float ***data) {
    for (int i = 0; i < batch_size; i++) {
        printf("Batch %d:\n", i);
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < d_model; k++) {
                printf("%f ", data[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void Softmax(float score[10][10]){
    for(int i=0; i<seq_len; i++){
        float max = score[i][0];
        for(int j=0; j<seq_len; j++){
            if(score[i][j]>max){
                max = score[i][j];
            }
        }

        float sum = 0.0;
        for(int j=0; j<seq_len; j++){
            score[i][j] = exp(score[i][j] - max);
            sum += score[i][j];
        }
        for(int j=0; j<seq_len; j++){
            score[i][j] /= sum;
        }
    }
}

void Attention(float ***data, float ***key, float ***value, float ***output) {
    float score, scale = 1.0f / sqrt((float)d_model);
    float tmp[batch_size][10][10], tmp2[batch_size][10][10];
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < seq_len; k++) {
                score = 0.0f;
                for (int l = 0; l < d_model; l++) {
                    score += data[i][j][l] * key[i][k][l];
                }
                tmp[i][j][k] = score * scale;  // 결과를 output에 저장
            }
        }

        Softmax(tmp[i]);
    }
    printf("%f\n",sqrt((float)d_model));

    for(int i=0; i<batch_size; i++){
        for(int j=0; j<10; j++){
            for(int k=0; k<10; k++){
                printf("%f ",tmp[i][j][k]);
            }
            printf("\n");
        }
    }
}

int main() {
    // Q, K, V load
    float ***query, ***key, ***value, ***output;
    data_allocate(&query);
    data_allocate(&key);
    data_allocate(&value);
    data_allocate(&output);

    data_load("./data/query.txt", query);
    data_load("./data/key.txt", key);
    data_load("./data/value.txt", value);

    // print_tensor(query);
    // print_tensor(key);
    // print_tensor(value);

    Attention(query, key, value, output);

    return 0;
}
