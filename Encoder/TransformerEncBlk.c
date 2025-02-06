#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TransformerData.h"

int batch_size, seq_len, d_model, header, d_k;


void LinearMapping(Linear *linear, float ***input, float ***output){
    float O;

    for(int i=0; i<batch_size; i++){
        for(int j=0; j<seq_len; j++){
            for(int k=0; k<d_model; k++){
                O = linear->bias[k];
                for(int l=0; l<d_model; l++){
                    O += input[i][j][l] * linear->weight[k][l];
                }
                output[i][j][k] = O;
            }
        }
    }
}

void Softmax(float **score){
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

void Attention_h(float ***query, float ***key, float ***value, 
                float output[header][seq_len][d_k]) {
    
    float score, scale = 1.0f / sqrt((float)d_k);
    float ***tmp;
    data_allocate_3d(&tmp, header, seq_len, seq_len);

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
                    score += tmp[i][j][l] * value[i][l][k];
                }
                output[i][j][k] = score;
            }
        }
    }
}

void MultiHeadAttention(float ***input, float ***output,
                        Linear *linear_q, Linear *linear_k, Linear *linear_v, Linear *linear_last){
    Results results;
    allocate_results(&results, batch_size, header, seq_len, d_k);
    
    float output_tmp[batch_size][header][seq_len][d_k];
    float ***tmp, ***output_tmp2;
    float ****head1, ****head2;
    data_allocate_4d(&head1,batch_size,seq_len,header,d_k);
    data_allocate_4d(&head2,batch_size,header,seq_len,d_k);
    data_allocate_3d(&tmp, batch_size, seq_len, d_model);
    data_allocate_3d(&output_tmp2, batch_size, seq_len, d_model);

    for(int a=0; a<3; a++){
        if(a==0) LinearMapping(linear_q,input,tmp);
        if(a==1) LinearMapping(linear_k,input,tmp);
        if(a==2) LinearMapping(linear_v,input,tmp); 

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
                        head2[i][k][j][l] = head1[i][j][k][l];
                    }
                }
            }
        }
        if(a==0) memcpy(results.Q[0][0][0], head2, batch_size * header * seq_len * d_k * sizeof(float));
        if(a==1) memcpy(results.K[0][0][0], head2, batch_size * header * seq_len * d_k * sizeof(float));
        if(a==2) memcpy(results.V[0][0][0], head2, batch_size * header * seq_len * d_k * sizeof(float));
    }      
    
    /*attention*/
    for(int i=0; i<batch_size; i++){
        Attention_h(results.Q[i],results.K[i],results.V[i],output_tmp[i]);
    }

    /*concat*/
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int h = 0; h < header; h++) { // header 별 d_k 연결
                for (int l = 0; l < d_k; l++) {
                    output_tmp2[i][j][h * d_k + l] = output_tmp[i][h][j][l]; 
                }
            }
        }
    }                   
    data_print_3D(output_tmp2,batch_size,seq_len,d_model);
    LinearMapping(linear_last,output_tmp2,output);

}

void TransformerBlock(float ***input, float ***output, int batch, int seq_length, int dim_model, int ffn_hidden, int n_head){
    batch_size = batch, seq_len = seq_length, d_model = dim_model, header = n_head, d_k = d_model/n_head;
    
    Linear linear_q, linear_k, linear_v, linear_last;
    data_allocate_2d(&linear_q.weight,d_model,d_model); data_allocate_1d(&linear_q.bias, d_model); 
    data_allocate_2d(&linear_k.weight,d_model,d_model); data_allocate_1d(&linear_k.bias, d_model); 
    data_allocate_2d(&linear_v.weight,d_model,d_model); data_allocate_1d(&linear_v.bias, d_model); 

    WeightBias_load("./Data/query_linear_weights.txt", "./Data/query_linear_biases.txt", &linear_q, d_model, d_model);
    WeightBias_load("./Data/key_linear_weights.txt", "./Data/key_linear_biases.txt", &linear_k, d_model, d_model);
    WeightBias_load("./Data/value_linear_weights.txt", "./Data/value_linear_biases.txt", &linear_v, d_model, d_model);
        
    MultiHeadAttention(input, output,&linear_q, &linear_k, &linear_v,&linear_last);
    data_print_3D(output, batch, seq_len, d_model);
}