#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define batch_size 2
#define seq_len 10
#define d_model 6
#define header 2
#define d_k d_model/header

typedef struct{
    float weights[d_model][d_model];
    float biases[d_model];
} Linear;

typedef struct{
    float Q[batch_size][header][seq_len][d_k];
    float K[batch_size][header][seq_len][d_k];
    float V[batch_size][header][seq_len][d_k];
} Results;

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

void QKV_load(const char *filename, float ***data) {
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

void WeightBias_load(const char *filename1, const char *filename2, Linear *linear){
    FILE *file1 = fopen(filename1, "r");
    FILE *file2 = fopen(filename2, "r");
    if(!file1|!file2){
        perror("Failed to open file");
        exit(1);
    }

    for(int i=0; i<d_model; i++){
        for(int j=0; j<d_model; j++){
            fscanf(file1,"%f",&linear->weights[i][j]);
        }
        fscanf(file2,"%f",&linear->biases[i]);
    }

    fclose(file1);
    fclose(file2);
}

void LinearMapping(Linear *linear, float ***input, float output[batch_size][seq_len][d_model]){
    float O;

    for(int i=0; i<batch_size; i++){
        for(int j=0; j<seq_len; j++){
            for(int k=0; k<d_model; k++){
                O = linear->biases[k];
                for(int l=0; l<d_model; l++){
                    O += input[i][j][l] * linear->weights[k][l];
                }
                output[i][j][k] = O;
            }
        }
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


void Attention_h(float query[header][seq_len][d_k], float key[header][seq_len][d_k], float value[header][seq_len][d_k], 
                float output[header][seq_len][d_k]) {
    
    float score, scale = 1.0f / sqrt((float)d_k);
    float tmp[header][seq_len][seq_len];

    // attention score 구하기
    for (int i = 0; i < header; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < seq_len; k++) {
                score = 0.0f;
                for (int l = 0; l < d_k; l++) {
                    score += query[i][j][l] * key[i][k][l];
                }
                tmp[i][j][k] = score * scale;  // 결과를 output에 저장
            }
        }
        Softmax(tmp[i]);
    }

    // attention value 구하기
    for(int i=0; i<header; i++){
        for(int j=0; j<seq_len; j++){
            for(int k=0; k<d_k; k++){
                score = 0.0f;
                for(int l=0; l<seq_len; l++){
                    score += tmp[i][j][l] * value[i][k][l];
                }
                output[i][j][k] = score;
            }
        }
    }
}

void MultiHeadAttention(float ***query, float ***key, float ***value, float ***output,
                        Linear *linear_q, Linear *linear_k, Linear *linear_v){
    
    Results results;
    float output_tmp[batch_size][header][seq_len][d_k];

    /*차례로 선형 변환 후 헤드 분리*/
    
    for(int a=0; a<3; a++){
        float tmp [batch_size][seq_len][d_model];
        float head1[batch_size][seq_len][header][d_k], head2[batch_size][header][seq_len][d_k];
        // linear
        if(a==0) LinearMapping(linear_q,query,tmp);
        if(a==1) LinearMapping(linear_k,key,tmp);
        if(a==2) LinearMapping(linear_v,value,tmp);

        // split
        for(int i=0; i<batch_size; i++){
            for(int j=0; j<seq_len; j++){
              for(int k=0; k<header; k++){
                for(int l = 0; l<d_k; l++){
                    head1[i][j][k][l] = tmp[i][j][k*d_k+l];
                    }
                }
            }
        }
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < seq_len; j++) {
                for (int k = 0; k < header; k++) {
                    for (int l = 0; l < d_k; l++) {
                        // seq_len과 header를 전치
                        head2[i][k][j][l] = head1[i][j][k][l];
                    }
                }
            }
        }
        if(a==0) memcpy(results.Q, head2, sizeof(results.Q));
        if(a==1) memcpy(results.K, head2, sizeof(results.K));
        if(a==2) memcpy(results.V, head2, sizeof(results.V));
    }

    /*attention*/
    for(int i=0; i<batch_size; i++){
        Attention_h(results.Q[i],results.K[i],results.V[i],output_tmp[i]);
    }

    /*concat*/
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < d_model; k++) {
                int header_idx = k / d_k;  // header index
                int d_k_idx = k % d_k;    // d_k index
                output[i][j][k] = output_tmp[i][header_idx][j][d_k_idx];
            }
        }
    }

    print_tensor(output);
}

int main() {
    
    /*Q, K, V load*/
    float ***query, ***key, ***value, ***output;
    data_allocate(&query);
    data_allocate(&key);
    data_allocate(&value);
    data_allocate(&output);

    QKV_load("./data/query.txt", query);
    QKV_load("./data/key.txt", key);
    QKV_load("./data/value.txt", value);

    /*Q,K,V's weight and bias load*/
    Linear linear_q, linear_k, linear_v;
    WeightBias_load("./data/linear1_weights.txt","./data/linear1_biases.txt",&linear_q);
    WeightBias_load("./data/linear2_weights.txt","./data/linear2_biases.txt",&linear_k);
    WeightBias_load("./data/linear3_weights.txt","./data/linear4_biases.txt",&linear_v);

    MultiHeadAttention(query,key,value,output,&linear_q,&linear_k,&linear_v);

    return 0;
}
