#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TransformerData.h"
#include "TransformerDecBlk.h"

/* Hyperparameters 설정 */
#define enc_voc_size 100
#define max_len 16
#define d_model 16
#define ffn_hidden 128
#define n_head 4
#define n_layers 1
#define batch_size 2
#define seq_len 10

int **input;
float **tok_emb, **pos_emb, ***emb_output, ***output;

void compute_positional_encoding(float **pos_emb) {
    for (int t = 0; t < max_len; t++) {
        for (int d = 0; d < d_model; d += 2) { // 짝수 인덱스
            float div_term = exp(-log(10000.0) * (float)d / d_model);
            float angle = t * div_term;
            pos_emb[t][d] = sin(angle);
            if (d + 1 < d_model) { // 홀수 인덱스
                pos_emb[t][d + 1] = cos(angle);
            }
        }
    }
}

void TransformerEmbedding() {
   
    compute_positional_encoding(pos_emb);

    // 입력 시퀀스를 토큰 임베딩으로 변환
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int token_id = (int)input[b][t]; // 정수형 토큰 ID
            for (int d = 0; d < d_model; d++) {
                emb_output[b][t][d] = tok_emb[token_id][d] + pos_emb[t][d];
            }
        }
    }
}

/* Transformer Encoder 실행 */
void Transformer_Decoder() {
    TransformerEmbedding();
    data_print_3D(emb_output,batch_size,seq_len,d_model);
    for(int i=0; i<n_layers; i++){
        DecoderBlock(emb_output, output, batch_size, seq_len, d_model, ffn_hidden, n_head);
    }
}

/* 메인 실행 */
int main() {
    /* 필요한 변수들 동적 할당 및 데이터 로드 */
    data_allocate_int_2d(&input, batch_size, seq_len);
    data_allocate_2d(&tok_emb, enc_voc_size, d_model);
    data_allocate_2d(&pos_emb, max_len, d_model);  // Positional Encoding 추가
    data_allocate_3d(&emb_output, batch_size, seq_len, d_model);
    data_allocate_3d(&output, batch_size, seq_len, d_model);

    data_load_int_2D("./Data/trg.bin", input, batch_size, seq_len);
    data_load_float_2D("./Data/tok_emb.bin", tok_emb, enc_voc_size, d_model);

    printf("input\n");
    for(int i=0; i<batch_size; i++){
        for(int j=0; j<seq_len; j++){
            printf("%d ",input[i][j]);
        }
        printf("\n");
    }

    Transformer_Decoder();

    return 0;
}
