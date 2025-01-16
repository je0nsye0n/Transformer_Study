// dataset.h
#ifndef HEADER_H
#define HEADER_H

typedef struct {
    float **data;   // 리뷰 데이터 (2D 배열)
    int *labels;  // 레이블 (0 또는 1)
    int size;     // 데이터 크기
} Dataset;

#endif
