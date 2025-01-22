#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BATCH_SIZE 2
#define SEQ_LEN 4
#define D_MODEL 8
#define HIDDEN 32

typedef struct {
    float weights[D_MODEL][HIDDEN];
    float biases[HIDDEN];
} Linear1;

typedef struct {
    float weights[HIDDEN][D_MODEL];
    float biases[D_MODEL];
} Linear2;

void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0, x[i]);
    }
}
// void linear_forward1(Linear1 *linear, float input[BATCH_SIZE][SEQ_LEN][D_MODEL], float output[BATCH_SIZE][SEQ_LEN][HIDDEN]) {
//     for (int b = 0; b < BATCH_SIZE; b++) { // 배치별 반복
//         for (int s = 0; s < SEQ_LEN; s++) { // 시퀀스별 반복
//             for (int h = 0; h < HIDDEN; h++) { // 출력 차원별 계산
//                 // biases[h]로 초기화
//                 float output_v = linear->biases[h];

//                 // 입력과 가중치의 곱
//                 for (int i = 0; i < D_MODEL; i++) {
//                     output_v += input[b][s][i] * linear->weights[i][h];
//                 }

//                 // 출력 저장
//                 output[b][s][h] = output_v;

//                 // 디버깅 출력
//                 printf("%.4f ", output_v);
//             }
//             printf("\n");
//         }
//     }
// }

void linear_forward1(Linear1 *linear, float input[BATCH_SIZE][SEQ_LEN][D_MODEL], float output[BATCH_SIZE][SEQ_LEN][HIDDEN]) {

    float O, I;

    for (int b = 0; b < BATCH_SIZE; b++) { // 배치별 (2)
        printf("[");
        for (int s = 0; s < SEQ_LEN; s++) { // 시퀀스별 (4)
            printf("[");
            for (int h = 0; h < HIDDEN; h++) { // 출력 차원 (32)

                O = linear->biases[h];

                for(int i=0; i<D_MODEL; i++){
                    O += input[b][s][i] * linear->weights[h][i];
                }

                output[b][s][h] = O;
                printf("%.4f ",O);

            }
            printf("],\n");
        }
        printf("]\n");
    }
}


void linear_forward2(Linear2 *linear, float input[BATCH_SIZE][SEQ_LEN][HIDDEN], float output[BATCH_SIZE][SEQ_LEN][D_MODEL]) {
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int d = 0; d < D_MODEL; d++) {
                output[b][s][d] = linear->biases[d];
                for (int h = 0; h < HIDDEN; h++) {
                    output[b][s][d] += input[b][s][h] * linear->weights[h][d];
                }
            }
        }
    }
}

void load_weights(const char *filename, float *data, int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%f", &data[j * rows + i]); // Adjust for transpose
           // printf("Loaded weight: %.10f\n", data[j * rows + i]);

        }
    }
    fclose(file);
}

void load_biases(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        fscanf(file, "%f", &data[i]);
        //printf("Loaded bias: %.10f\n", data[i]);
    }
    fclose(file);
}

void print_tensor(const char *label, float *tensor, int dim1, int dim2, int dim3) {
    //printf("%s:\n", label);
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                //printf("%.4f ", tensor[(i * dim2 * dim3) + (j * dim3) + k]);
            }
            //printf("\n");
        }
        //printf("\n");
    }
}

void print_weights(const char *label, float weights[D_MODEL][HIDDEN]) {
    printf("%s Weights:\n", label);
    for (int i = 0; i < D_MODEL; i++) {
        for (int j = 0; j < HIDDEN; j++) {
            printf("%8.4f ", weights[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main() {
    Linear1 linear1;
    Linear2 linear2;

    // Load weights and biases
    load_weights("data2/linear1_weights.txt", &linear1.weights[0][0], D_MODEL, HIDDEN);
    load_biases("data2/linear1_biases.txt", linear1.biases, HIDDEN);
    load_weights("data2/linear2_weights.txt", &linear2.weights[0][0], HIDDEN, D_MODEL);
    load_biases("data2/linear2_biases.txt", linear2.biases, D_MODEL);

    // Load input data
    float input[BATCH_SIZE][SEQ_LEN][D_MODEL];
    FILE *input_file = fopen("data2/input_data.txt", "r");
    if (!input_file) {
        perror("Failed to open input file");
        exit(1);
    }

    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; i++) {
        fscanf(input_file, "%f", &((float *)input)[i]);
    }
    fclose(input_file);

    // Forward pass
    float hidden_output[BATCH_SIZE][SEQ_LEN][HIDDEN] = {0};
    float output[BATCH_SIZE][SEQ_LEN][D_MODEL] = {0};

    // Linear1 -> ReLU
    linear_forward1(&linear1, input, hidden_output);
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            relu(hidden_output[b][s], HIDDEN);
        }
    }
    // Linear2
    linear_forward2(&linear2, hidden_output, output);

    // Print results
   // print_tensor("Input Tensor", &input[0][0][0], BATCH_SIZE, SEQ_LEN, D_MODEL);
   // print_tensor("Hidden Output After Linear1 and ReLU", &hidden_output[0][0][0], BATCH_SIZE, SEQ_LEN, HIDDEN);
   //print_tensor("Final Output Tensor", &output[0][0][0], BATCH_SIZE, SEQ_LEN, D_MODEL);

    return 0;
}
