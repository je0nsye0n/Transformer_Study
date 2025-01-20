#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int h;         // 어텐션 헤드 수
    int d_model;   // 모델의 전체 차원
    int d_k;       // 헤드별 차원(d_model/h)
    float ****linears; // 쿼리(Query), 키(Key), 값(Value)와 출력 변환에 사용되는 가중치 배열
    float dropout; 
} MultiHeadedAttention;

/*4차원 배열 동적 할당 - 랜덤 초기화 범위 [-0.1,0.1]로 설정*/
float ****allocate_4d_array(int a, int b, int c, int d) {
    float ****array = (float ****)malloc(a * sizeof(float ***));
    for (int i = 0; i < a; i++) {
        array[i] = (float ***)malloc(b * sizeof(float **));
        for (int j = 0; j < b; j++) {
            array[i][j] = (float **)malloc(c * sizeof(float *));
            for (int k = 0; k < c; k++) {
                array[i][j][k] = (float *)malloc(d * sizeof(float));
                for (int l = 0; l < d; l++) {
                    array[i][j][k][l] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f; 
                }
            }
        }
    }
    return array;
}

void free_4d_array(float ****array, int a, int b, int c) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int k = 0; k < c; k++) {
                free(array[i][j][k]);
            }
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}

MultiHeadedAttention *create_multi_head_attention(int h, int d_model, float dropout) {
    MultiHeadedAttention *attn = (MultiHeadedAttention *)malloc(sizeof(MultiHeadedAttention));
    attn->h = h;
    attn->d_model = d_model;
    attn->d_k = d_model / h;
    attn->dropout = dropout;
    attn->linears = allocate_4d_array(4, h, attn->d_k, d_model); 
    return attn;
}

void free_multi_head_attention(MultiHeadedAttention *attn) {
    free_4d_array(attn->linears, 4, attn->h, attn->d_k);
    free(attn);
}

// Dot product function 
void matmul(float *A, float *B, float *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// Scaled dot-product attention
void attention(float *query, float *key, float *value, 
                float *output, int seq_len, int d_k) {
    float *scores = (float *)malloc(seq_len * seq_len * sizeof(float));
    float scale = 1.0f / sqrt((float)d_k);

    matmul(query, key, scores, seq_len, d_k, seq_len);
    for (int i = 0; i < seq_len * seq_len; i++) {
        scores[i] *= scale;
    }

    // Softmax with numerical stability
    for (int i = 0; i < seq_len; i++) {
        float max_score = scores[i * seq_len];
        for (int j = 1; j < seq_len; j++) {
            if (scores[i * seq_len + j] > max_score) {
                max_score = scores[i * seq_len + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = exp(scores[i * seq_len + j] - max_score);
            sum += scores[i * seq_len + j];
        }
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] /= sum;
        }
    }

    matmul(scores, value, output, seq_len, seq_len, d_k);
    free(scores);
}

// Multi-Head Attention 
void multi_head_attention_forward(MultiHeadedAttention *attn, 
                                float *query, float *key, float *value, 
                                float *output, int batch_size, int seq_len) {
    int d_k = attn->d_k;
    int h = attn->h;

    memset(output, 0, sizeof(float) * batch_size * seq_len * attn->d_model); // Ensure output is cleared

    for (int i = 0; i < h; i++) {
        float *query_slice = query + i * seq_len * d_k;
        float *key_slice = key + i * seq_len * d_k;
        float *value_slice = value + i * seq_len * d_k;
        float *head_output = (float *)malloc(seq_len * d_k * sizeof(float));

        attention(query_slice, key_slice, value_slice, head_output, seq_len, d_k);

        // Combine head outputs into the final output
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < d_k; k++) {
                output[j * attn->d_model + i * d_k + k] = head_output[j * d_k + k];
            }
        }

        free(head_output);
    }
}

int main() {
    int batch_size = 2;
    int seq_len = 10;
    int d_model = 6;
    int h = 2;

    MultiHeadedAttention *attn = create_multi_head_attention(h, d_model, 0.1);

    // query, key, value
    float query[batch_size * seq_len * d_model];
    float key[batch_size * seq_len * d_model];
    float value[batch_size * seq_len * d_model];
    float output[batch_size * seq_len * d_model];

    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        query[i] = ((float)rand() / RAND_MAX);
        key[i] = ((float)rand() / RAND_MAX);
        value[i] = ((float)rand() / RAND_MAX);
    }

    memset(output, 0, sizeof(output)); // Initialize output to 0

    multi_head_attention_forward(attn, query, key, value, output, batch_size, seq_len);

    // Print output to console
    printf("Output:\n");
    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        printf("%f ", output[i]);
        if ((i + 1) % d_model == 0) printf("\n");
    }

    // Save output to file
    FILE *output_file = fopen("./data/c_output.txt", "w");
    if (output_file == NULL) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        fprintf(output_file, "%f\n", output[i]);
    }
    fclose(output_file);

    free_multi_head_attention(attn);
    return 0;
}
