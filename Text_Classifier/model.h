#ifndef MODEL_H
#define MODEL_H

#include "header.h"

typedef struct {
    int input_dim;
    int output_dim;
    float **w, *b;
} transLayer;

transLayer *create_fclayer(int input_dim, int output_dim);
float *classifier(transLayer *model, float *data);

#endif
