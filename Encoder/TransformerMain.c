#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TransformerEncBlk.h"
#include "TransformerData.h"

/* Hyperparameters 설정 */
#define enc_voc_size 100
#define max_len 50
#define d_model 16
#define ffn_hidden 128
#define n_head 4
#define n_layers 1

/* input data 설정 */
#define batch_size 2
#define seq_len 5

float **input, **tok_emb, **pos_emb, ***emb_output, ***output;

/* 2D 데이터 로드 함수 */
void data_load_2D(const char *filename, float **data, int row, int col) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fscanf(file, "%f", &data[i][j]);
        }
    }

    fclose(file);
}

void TransformerEmbedding() {
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            float denominator = pow(10000.0, (float)i / d_model);
            pos_emb[pos][i] = sin(pos / denominator);
            if (i + 1 < d_model) {
                pos_emb[pos][i + 1] = cos(pos / denominator);
            }
        }
    }
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            int token_id = (int)input[i][j];  // 정수 ID 가져오기
            if (token_id >= enc_voc_size || token_id < 0) {
                printf("Error: token_id %d out of range\n", token_id);
                exit(1);
            }

            // 임베딩 행을 복사하고 Positional Encoding 더하기
            for (int k = 0; k < d_model; k++) {
                emb_output[i][j][k] = tok_emb[token_id][k] + pos_emb[j][k];
            }
        }
    }
}

/* Transformer Encoder 실행 */
void Transformer_Encoder() {
    TransformerEmbedding();
    for(int i=0; i<n_layers; i++){
        TransformerBlock(emb_output, output, batch_size, seq_len, d_model, ffn_hidden, n_head);
    }
}

/* 메인 실행 */
int main() {
    /* 필요한 변수들 동적 할당 및 데이터 로드 */
    data_allocate_2d(&input, batch_size, seq_len);
    data_allocate_2d(&tok_emb, enc_voc_size, d_model);
    data_allocate_2d(&pos_emb, max_len, d_model);  // Positional Encoding 추가
    data_allocate_3d(&emb_output, batch_size, seq_len, d_model);
    data_allocate_3d(&output, batch_size, seq_len, d_model);

    data_load_2D("./Data/input.txt", input, batch_size, seq_len);
    data_load_2D("./Data/embedding_weights.txt", tok_emb, enc_voc_size, d_model);

    // Transformer Encoder 실행
    Transformer_Encoder();

    // 메모리 해제
    data_free_2d(input, batch_size);
    data_free_2d(tok_emb, enc_voc_size);
    data_free_2d(pos_emb, max_len);  // Positional Encoding 해제
    data_free_3d(emb_output, batch_size, seq_len);

    return 0;
}
