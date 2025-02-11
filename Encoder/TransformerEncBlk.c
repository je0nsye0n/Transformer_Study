#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TransformerData.h"

int batch_size, seq_len, d_model, header, d_k, hidden;

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

void layer_norm(float* x, float* gamma, float* beta, float* output, int feature_dim, float epsilon) {
    float mean = 0.0, var = 0.0;

    for (int i = 0; i < feature_dim; i++) {
        mean += x[i];
    }
    mean /= feature_dim;

    for (int i = 0; i < feature_dim; i++) {
        var += (x[i] - mean) * (x[i] - mean);
    }
    var /= feature_dim;

    for (int i = 0; i < feature_dim; i++) {
        output[i] = ((x[i] - mean) / sqrt(var + epsilon)) * gamma[i] + beta[i];
    }
}

void Attention_h(float ***query, float ***key, float ***value, float ***output) {
    
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
                        Linear *linear_q, Linear *linear_k, Linear *linear_v, Linear *linear_concat){
    Results results;
    allocate_results(&results, batch_size, header, seq_len, d_k);
    
    float ****output_tmp, ***output_tmp2, ***tmp;
    float ****head1, ****head2;
    data_allocate_4d(&head1,batch_size,seq_len,header,d_k);
    data_allocate_4d(&head2,batch_size,header,seq_len,d_k);
    data_allocate_4d(&output_tmp,batch_size,header,seq_len,d_k);
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
       
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < header; j++) {
                for (int k = 0; k < seq_len; k++) {
                    if(a==0) memcpy(results.Q[i][j][k], head2[i][j][k], d_k * sizeof(float));
                    if(a==1) memcpy(results.K[i][j][k], head2[i][j][k], d_k * sizeof(float));
                    if(a==2) memcpy(results.V[i][j][k], head2[i][j][k], d_k * sizeof(float));
                }
            }
        }
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

    //data_print_3D(output_tmp2,batch_size,seq_len,d_model);
    LinearMapping(linear_concat,output_tmp2,output);
}

/* feedforward 함수 */
void feedforward(Linear *linear1, Linear *linear2, 
    float ***input, float ***output) {

    float hidden[batch_size][seq_len][hidden];

    // 1단계: 확장 (Feedforward1) 및 ReLU 적용
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < hidden; k++) {
                float O = linear1->bias[k];

                for (int l = 0; l < d_model; l++) {
                    O += input[i][j][l] * linear1->weight[k][l];
                }

                hidden[i][j][k] = fmax(O, 0.0);  // ReLU 적용
            }
        }
    }

    // 2단계: 축소 (Feedforward2)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < d_model; k++) {
                float O = linear2->bias[k];

                for (int l = 0; l < hidden; l++) {
                    O += hidden[i][j][l] * linear2->weight[k][l];
                }

                output[i][j][k] = O;
            }
        }
    }
}

void LoadTransformerWeights(Linear *linear_q, Linear *linear_k, Linear *linear_v, Linear *linear_concat,
                            Linear *linear_ff1, Linear *linear_ff2, float **gamma, float **beta) {
    data_allocate_2d(&linear_q->weight, d_model, d_model); data_allocate_1d(&linear_q->bias, d_model);
    data_allocate_2d(&linear_k->weight, d_model, d_model); data_allocate_1d(&linear_k->bias, d_model);
    data_allocate_2d(&linear_v->weight, d_model, d_model); data_allocate_1d(&linear_v->bias, d_model);
    data_allocate_2d(&linear_concat->weight, d_model, d_model); data_allocate_1d(&linear_concat->bias, d_model);
    data_allocate_2d(&linear_ff1->weight, hidden, d_model); data_allocate_1d(&linear_ff1->bias, hidden);
    data_allocate_2d(&linear_ff2->weight, d_model, hidden); data_allocate_1d(&linear_ff2->bias, d_model);
    data_allocate_1d(gamma, d_model);
    data_allocate_1d(beta, d_model);
    
    WeightBias_load("./Data/query_weights.txt", "./Data/query_biases.txt", linear_q, d_model, d_model);
    WeightBias_load("./Data/key_weights.txt", "./Data/key_biases.txt", linear_k, d_model, d_model);
    WeightBias_load("./Data/value_weights.txt", "./Data/value_biases.txt", linear_v, d_model, d_model);
    WeightBias_load("./Data/concat_weights.txt", "./Data/concat_biases.txt", linear_concat, d_model, d_model);
    WeightBias_load("./Data/ff1_weights.txt", "./Data/ff1_biases.txt", linear_ff1, hidden, d_model);
    WeightBias_load("./Data/ff2_weights.txt", "./Data/ff2_biases.txt", linear_ff2, d_model, hidden);
    WeightBias_load("./Data/LayerNorm_gamma.txt", NULL, NULL, gamma, d_model);
    WeightBias_load("./Data/LayerNorm_beta.txt", NULL, NULL, beta, d_model);
}

void TransformerBlock(float ***input, float ***output, int batch, int seq_length, int dim_model, int ffn_hidden, int n_head){
    batch_size = batch, seq_len = seq_length, d_model = dim_model, header = n_head, d_k = d_model/n_head, hidden=ffn_hidden;
    
    Linear linear_q, linear_k, linear_v, linear_concat, linear_ff1, linear_ff2;
    float *gamma, *beta;
    LoadTransformerWeights(&linear_q, &linear_k, &linear_v, &linear_concat, &linear_ff1, &linear_ff2, &gamma, &beta);
    
    /* Multi-Head Attention */
    float ***attn_output;
    data_allocate_3d(&attn_output, batch_size, seq_len, d_model);
    MultiHeadAttention(input, attn_output, &linear_q, &linear_k, &linear_v, &linear_concat);
    
    /* Residual Connection & Layer Norm */
    float ***norm1_output;
    data_allocate_3d(&norm1_output, batch_size, seq_len, d_model);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < d_model; k++) {
                attn_output[i][j][k] += input[i][j][k];
            }
            layer_norm(attn_output[i][j], gamma, beta, norm1_output[i][j], d_model, 1e-5);
        }
    }
    
    /* Feed Forward Network */
    float ***ffn_output;
    data_allocate_3d(&ffn_output, batch_size, seq_len, d_model);
    feedforward(&linear_ff1, &linear_ff2, norm1_output, ffn_output);
    
    /* Residual Connection & Layer Norm */
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < d_model; k++) {
                ffn_output[i][j][k] += norm1_output[i][j][k];
            }
            layer_norm(ffn_output[i][j], gamma, beta, output[i][j], d_model, 1e-5);
        }
    }

}

