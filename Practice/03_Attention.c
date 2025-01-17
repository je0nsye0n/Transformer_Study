#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// 파라미터
#define seq_len 4
#define d_model 6
#define head 2

// Q,K,V
float query[seq_len][d_model];
float key[seq_len][d_model];
float value[seq_len][d_model];

//Fully Connected Function
/*
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
*/
void FullyConnected(){
 
}


//Dropout Function
void Dropout(float p){
    
}

// Transpose Function
float **transpose(float **matrix, int rows, int cols) {
    // 동적 메모리 할당
    float **trans = (float **)malloc(cols * sizeof(float *));
    for (int i = 0; i < cols; i++) {
        trans[i] = (float *)malloc(rows * sizeof(float));
    }

    // 전치 행렬 계산
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            trans[j][i] = matrix[i][j];
        }
    }

    return trans;
}

// Scaled-dot Attention Function
void attention(){

    // query 전치행렬 만들기
    float **query_t = (float **)malloc(d_model*sizeof(float *));
    for(int i=0; i<d_model; i++) query_t[i] = (float *)malloc(seq_len*sizeof(float));

    query_t = transpose(query,d_model,seq_len);

    // attention score 저장
    float **attn_score = (float **)malloc(d_model*sizeof(float *));
    for(int i=0; i<d_model; i++) query_t[i] = (float *)malloc(seq_len*sizeof(float));    

}

void MultiHeadAttn(){
    int d_k = (int)(d_model/head);
}

int main(void){
    
    // random 생성
    for(int j=0; j<seq_len; j++){
        for(int k=0; k<d_model; k++){
            query[j][k] = ((float)rand() / RAND_MAX) * 0.1f;
            key[j][k] = ((float)rand() / RAND_MAX) * 0.1f;
            value[j][k] = ((float)rand() / RAND_MAX) * 0.1f;
        }
    }
    

}