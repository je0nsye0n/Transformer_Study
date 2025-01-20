#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int h;         // Number of heads
    int d_model;   // Model dimension
    int d_k;       // Dimension per head
    float ****linears; // Weights for query, key, value, and output
    float dropout; 
} MultiHeadedAttention;


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
void attention(float *query, float *key, float *value, float *output, int seq_len, int d_k) {
    float *scores = (float *)malloc(seq_len * seq_len * sizeof(float));
    float scale = 1.0f / sqrt((float)d_k);

    matmul(query, key, scores, seq_len, d_k, seq_len);
    for (int i = 0; i < seq_len * seq_len; i++) {
        scores[i] *= scale;
    }

    // Softmax 
    for (int i = 0; i < seq_len; i++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = exp(scores[i * seq_len + j]);
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
void multi_head_attention_forward(MultiHeadedAttention *attn, float *query, float *key, float *value, float *output, int batch_size, int seq_len) {
    int d_k = attn->d_k;
    int h = attn->h;

    float *head_output = (float *)malloc(seq_len * d_k * sizeof(float));
    for (int i = 0; i < h; i++) {
        attention(query, key, value, head_output, seq_len, d_k);
    }

    // Combine heads 
    memcpy(output, head_output, seq_len * d_k * sizeof(float));
    free(head_output);
}

int main() {
    int batch_size = 1;
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

    multi_head_attention_forward(attn, query, key, value, output, batch_size, seq_len);

    // Print output
    printf("Output:\n");
    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        printf("%f ", output[i]);
        if ((i + 1) % d_model == 0) printf("\n");
    }

    free_multi_head_attention(attn);
    return 0;
}
