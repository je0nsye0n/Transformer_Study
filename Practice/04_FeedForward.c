#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BATCH_SIZE 2
#define SEQ_LEN 4
#define D_MODEL 8
#define HIDDEN 32

typedef struct
{
    float **weights;
    float *biases;
} Linear;

void load_input(float input[BATCH_SIZE][SEQ_LEN][D_MODEL])
{
    FILE *input_file = fopen("data2/input_data.txt", "r");
    if (!input_file)
    {
        perror("Failed to open input file");
        exit(1);
    }

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < SEQ_LEN; j++)
        {
            for (int k = 0; k < D_MODEL; k++)
            {
                fscanf(input_file, "%f", &input[i][j][k]);
            }
        }
    }

    fclose(input_file);
}

void load_data(const char *filename, const char *filename2, Linear *linear, int row, int col)
{
    /*
        linear1의 경우 : row(hidden), col(d_model)
        linear2의 경우 : row(d_model), col(hidden)
    */

    FILE *file = fopen(filename, "r");
    FILE *file2 = fopen(filename2, "r");
    if (!file || !file2)
    {
        perror("Failed to open file");
        exit(1);
    }

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            fscanf(file, "%f", &linear->weights[i][j]);
        }
        fscanf(file2, "%f", &linear->biases[i]);
    }

    fclose(file);
    fclose(file2);
}

void print_output(float output[BATCH_SIZE][SEQ_LEN][D_MODEL])
{
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        printf("[");
        for (int j = 0; j < SEQ_LEN; j++)
        {
            printf("[");
            for (int k = 0; k < D_MODEL; k++)
            {
                printf("%f ", output[i][j][k]);
            }
            printf("]\n");
        }
        printf("]\n");
    }
}

// feedforward1 : 확장
void feedforward1(Linear *linear, float input[BATCH_SIZE][SEQ_LEN][D_MODEL], float output[BATCH_SIZE][SEQ_LEN][HIDDEN])
{

    float O;
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < SEQ_LEN; j++)
        {
            for (int k = 0; k < HIDDEN; k++)
            {
                O = linear->biases[k];

                for (int l = 0; l < D_MODEL; l++)
                {
                    O += input[i][j][l] * linear->weights[k][l];
                }

                output[i][j][k] = fmax(O, 0.0);
            }
        }
    }
}

// feedforward2 : 축소
void feedforward2(Linear *linear, float input[BATCH_SIZE][SEQ_LEN][HIDDEN], float output[BATCH_SIZE][SEQ_LEN][D_MODEL])
{

    float O;
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < SEQ_LEN; j++)
        {
            for (int k = 0; k < D_MODEL; k++)
            {
                O = linear->biases[k];

                for (int l = 0; l < HIDDEN; l++)
                {
                    O += input[i][j][l] * linear->weights[k][l];
                }

                output[i][j][k] = O;
            }
        }
    }
}

int main()
{

    // load input
    float input[BATCH_SIZE][SEQ_LEN][D_MODEL];
    load_input(input);

    // linear 생성 및 초기화화
    Linear linear1, linear2;

    linear1.weights = (float **)malloc(sizeof(float *) * HIDDEN);
    for (int i = 0; i < HIDDEN; i++)
        linear1.weights[i] = (float *)malloc(sizeof(float) * D_MODEL);
    linear1.biases = (float *)malloc(sizeof(float) * HIDDEN);

    linear2.weights = (float **)malloc(sizeof(float *) * D_MODEL);
    for (int i = 0; i < D_MODEL; i++)
        linear2.weights[i] = (float *)malloc(sizeof(float) * HIDDEN);
    linear2.biases = (float *)malloc(sizeof(float) * D_MODEL);

    // load weights and biases
    load_data("data2/linear1_weights.txt", "data2/linear1_biases.txt", &linear1, HIDDEN, D_MODEL);
    load_data("data2/linear2_weights.txt", "data2/linear2_biases.txt", &linear2, D_MODEL, HIDDEN);

    // feedforward
    float hidden_output[BATCH_SIZE][SEQ_LEN][HIDDEN] = {0};
    float output[BATCH_SIZE][SEQ_LEN][D_MODEL] = {0};
    feedforward1(&linear1, input, hidden_output);
    feedforward2(&linear2, hidden_output, output);

    // output 출력
    print_output(output);

    // 메모리 할당 해제
    for (int i = 0; i < HIDDEN; i++)
    {
        free(linear1.weights[i]);
    }
    free(linear1.weights);

    free(linear1.biases);

    for (int i = 0; i < D_MODEL; i++)
    {
        free(linear2.weights[i]);
    }
    free(linear2.weights);

    free(linear2.biases);

    return 0;
}