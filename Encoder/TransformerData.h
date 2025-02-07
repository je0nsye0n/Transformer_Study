#ifndef TRANSFORMER_ENCODER_H
#define TRANSFORMER_ENCODER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Linear 구조체 정의 */
typedef struct {
    float **weight;
    float *bias;
} Linear;

/* Attention 결과 구조체 정의 */
typedef struct {
    float ****Q;
    float ****K;
    float ****V;
} Results;

/* 외부에서 사용할 전역 변수 */
extern float ***query, ***key, ***value;

/* 데이터 출력 함수 */
void data_print_1D(float *data, int dim1);
void data_print_2D(float **data, int dim1, int dim2);
void data_print_3D(float ***data, int dim1, int dim2, int dim3);
void data_print_4D(float ****data, int dim1, int dim2, int dim3, int dim4);

/* 데이터 동적 할당 함수 */
void data_allocate_1d(float **data, int dim1);
void data_allocate_2d(float ***data, int dim1, int dim2);
void data_allocate_3d(float ****data, int dim1, int dim2, int dim3);
void data_allocate_4d(float *****data, int dim1, int dim2, int dim3, int dim4);

/* 데이터 해제 함수 */
void data_free_1d(float *data);
void data_free_2d(float **data, int dim1);
void data_free_3d(float ***data, int dim1, int dim2);
void data_free_4d(float ****data, int dim1, int dim2, int dim3);

/* Weight와 Bias 로드 함수 */
void WeightBias_load(const char *filename1, const char *filename2, Linear *linear, int row, int col);

/* Results 구조체 메모리 할당 함수 */
void allocate_results(Results *results, int batch_size, int header, int seq_len, int d_k);

#endif /* TRANSFORMER_ENCODER_H */
