#include <stdio.h>
#include <math.h>

#define d_model 8
#define max_len 20
#define batch 2
#define seq_len 10

float input[batch][seq_len] = {
    {-0.5122, -1.7005, -1.4030, -0.3759, 0.8815, 0.6126, 0.4119, 1.4686, 0.3029, 0.6220},
    {1.3683, -0.3435, -0.1593, -0.2343, -0.1361, -0.3176, -0.3475, 0.0671, 1.4763, 0.4753}
};
float encoding[max_len][d_model];

void pos_enc() {
    for (int i = 0; i < max_len; i++) {
        for (int j = 0; j < d_model; j++) {
            if (j % 2 == 0) 
                encoding[i][j] = sin(i / pow(10000.0, (j / (float)d_model)));
            else 
                encoding[i][j] = cos(i / pow(10000.0, (j / (float)d_model)));
        }
    }
}

int main() {
    pos_enc();

    printf("Input Tensor:\n");
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < seq_len; j++) {
            printf("%.4f ", input[i][j]);
        }
        printf("\n");
    }

    printf("Output:\n");
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < seq_len; j++) {
            printf("[ ");
            for (int k = 0; k < d_model; k++) {
                printf("%.4f ", input[i][j] + encoding[j][k]);
            }
            printf("]\n");
        }
        printf("\n");
    }

    return 0;
}
