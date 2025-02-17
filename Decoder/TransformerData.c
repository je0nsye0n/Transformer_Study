#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*linear에 필요한 구조체*/
typedef struct {
    float **weight;
    float *bias;
}Linear;

typedef struct{
    float ****Q;
    float ****K;
    float ****V;
} Results;

/* 데이터 출력 함수 - 검증용 */
void data_print_1D(float *data, int dim1) {
    for (int i = 0; i < dim1; i++) {
        printf("%.4f ", data[i]);
    }
    printf("\n");
}

void data_print_2D(float **data, int dim1, int dim2) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            printf("%.4f ", data[i][j]);
        }
        printf("\n");
    }
}

void data_print_3D(float ***data, int dim1, int dim2, int dim3) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                printf("%.4f ", data[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void data_print_4D(float ****data, int dim1, int dim2, int dim3, int dim4) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int l = 0; l < dim4; l++) {
                    printf("%.4f ", data[i][j][k][l]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

/* 데이터 동적 할당 */

void data_allocate_int_2d(int ***data, int dim1, int dim2) {
    *data = (int **)malloc(dim1 * sizeof(int *));
    for (int i = 0; i < dim1; i++) {
        (*data)[i] = (int *)malloc(dim2 * sizeof(int));
    }
}


void data_allocate_1d(float **data, int dim1) {
    *data = (float *)malloc(dim1 * sizeof(float));
}

void data_allocate_2d(float ***data, int dim1, int dim2) {
    *data = (float **)malloc(dim1 * sizeof(float *));
    for (int i = 0; i < dim1; i++) {
        (*data)[i] = (float *)malloc(dim2 * sizeof(float));
    }
}

void data_allocate_3d(float ****data, int dim1, int dim2, int dim3) {
    *data = (float ***)malloc(dim1 * sizeof(float **));
    for (int i = 0; i < dim1; i++) {
        (*data)[i] = (float **)malloc(dim2 * sizeof(float *));
        for (int j = 0; j < dim2; j++) {
            (*data)[i][j] = (float *)malloc(dim3 * sizeof(float));
        }
    }
}

void data_allocate_4d(float *****data, int dim1, int dim2, int dim3, int dim4) {
    *data = (float ****)malloc(dim1 * sizeof(float ***));
    for (int i = 0; i < dim1; i++) {
        (*data)[i] = (float ***)malloc(dim2 * sizeof(float **));
        for (int j = 0; j < dim2; j++) {
            (*data)[i][j] = (float **)malloc(dim3 * sizeof(float *));
            for (int k = 0; k < dim3; k++) {
                (*data)[i][j][k] = (float *)malloc(dim4 * sizeof(float));
            }
        }
    }
}

/* 데이터 해제 */
void data_free_1d(float *data) {
    free(data);
}

void data_free_2d(float **data, int dim1) {
    for (int i = 0; i < dim1; i++) {
        free(data[i]);
    }
    free(data);
}

void data_free_3d(float ***data, int dim1, int dim2) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            free(data[i][j]);
        }
        free(data[i]);
    }
    free(data);
}

void data_free_4d(float ****data, int dim1, int dim2, int dim3) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                free(data[i][j][k]);
            }
            free(data[i][j]);
        }
        free(data[i]);
    }
    free(data);
}


void WeightBias_load(const char *filename1, const char *filename2, Linear *linear, int row, int col){
    FILE *file1 = fopen(filename1, "r");
    FILE *file2 = fopen(filename2, "r");
    if(!file1||!file2){
        perror("Failed to open file");
        exit(1);
    }

    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            fscanf(file1,"%f",&linear->weight[i][j]);
        }
        fscanf(file2,"%f",&linear->bias[i]);
    }

    fclose(file1);
    fclose(file2);
}

void allocate_results(Results *results, int batch_size, int header, int seq_len, int d_k) {
    results->Q = (float ****)malloc(batch_size * sizeof(float ***));
    results->K = (float ****)malloc(batch_size * sizeof(float ***));
    results->V = (float ****)malloc(batch_size * sizeof(float ***));

    for (int i = 0; i < batch_size; i++) {
        results->Q[i] = (float ***)malloc(header * sizeof(float **));
        results->K[i] = (float ***)malloc(header * sizeof(float **));
        results->V[i] = (float ***)malloc(header * sizeof(float **));

        for (int j = 0; j < header; j++) {
            results->Q[i][j] = (float **)malloc(seq_len * sizeof(float *));
            results->K[i][j] = (float **)malloc(seq_len * sizeof(float *));
            results->V[i][j] = (float **)malloc(seq_len * sizeof(float *));

            for (int k = 0; k < seq_len; k++) {
                results->Q[i][j][k] = (float *)malloc(d_k * sizeof(float));
                results->K[i][j][k] = (float *)malloc(d_k * sizeof(float));
                results->V[i][j][k] = (float *)malloc(d_k * sizeof(float));
            }
        }
    }
}

void data_load_int_2D(const char *filename, int **data, int row, int col) {
    FILE *file = fopen(filename, "rb");  // 바이너리 모드로 파일 열기
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }

    // 데이터 로드
    for (int i = 0; i < row; i++) {
        if (fread(data[i], sizeof(int), col, file) != col) {
            perror("Error reading file");
            fclose(file);
            exit(1);
        }
    }

    fclose(file);
}

void data_load_float_2D(const char *filename, float **data, int row, int col) {
    FILE *file = fopen(filename, "rb");  // 바이너리 모드로 파일 열기
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }

    // 데이터 로드
    for (int i = 0; i < row; i++) {
        if (fread(data[i], sizeof(float), col, file) != col) {
            perror("Error reading file");
            fclose(file);
            exit(1);
        }
    }

    fclose(file);
}