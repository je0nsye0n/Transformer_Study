#ifndef MODEL_H
#define MODEL_H

#include "header.h"

typedef struct {
    int input_dim;
    int output_dim;
    float **w, *b;
} fclayer;

fclayer *create_fclayer(int input_dim, int output_dim);
float *forward_fc(fclayer *layer, float *input);

fclayer *classifier(Dataset train_data, Dataset test_data);

#endif
