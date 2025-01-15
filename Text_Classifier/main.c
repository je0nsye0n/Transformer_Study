#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEN 100 // 각 리뷰의 최대 길이
#define VOCAB_SIZE 2000 // 허용할 단어의 최대 개수

typedef struct {
    int **data;   // 리뷰 데이터 (2D 배열)
    int *labels;  // 레이블 (0 또는 1)
    int size;     // 데이터 크기
} Dataset;

// CSV 파일에서 데이터셋 로드
Dataset load_dataset(const char *file_path, int max_len) {
    Dataset dataset;
    dataset.size = 0;

    // 파일 열기
    FILE *file = fopen(file_path, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    // 데이터 크기 계산
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), file)) {
        dataset.size++;
    }

    // 데이터 메모리 할당
    dataset.data = malloc(dataset.size * sizeof(int *));
    dataset.labels = malloc(dataset.size * sizeof(int));

    rewind(file); // 파일 포인터를 다시 시작으로 이동

    int idx = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        dataset.data[idx] = malloc(max_len * sizeof(int));
        
        char *token = strtok(buffer, ",");
        int word_idx = 0;

        // 리뷰 데이터 읽기
        while (token && word_idx < max_len) {
            dataset.data[idx][word_idx++] = atoi(token);
            token = strtok(NULL, ",");
        }

        // 남은 공간을 0으로 패딩
        while (word_idx < max_len) {
            dataset.data[idx][word_idx++] = 0;
        }

        // 레이블 읽기 (마지막 값)
        if (token) {
            dataset.labels[idx] = atoi(token);
        }

        idx++;
    }

    fclose(file);
    return dataset;
}

// 데이터셋 메모리 해제
void free_dataset(Dataset dataset) {
    for (int i = 0; i < dataset.size; i++) {
        free(dataset.data[i]);
    }
    free(dataset.data);
    free(dataset.labels);
}

int main() {
    // 데이터셋 로드
    Dataset train_data = load_dataset("data/train_data.csv", MAX_LEN);
    Dataset test_data = load_dataset("data/test_data.csv", MAX_LEN);

    // 데이터 크기 출력
    printf("Train Dataset Size: %d\n", train_data.size);
    printf("Test Dataset Size: %d\n", test_data.size);

    // 첫 번째 샘플 확인
    printf("First Train Sample (Padded):\n");
    for (int i = 0; i < MAX_LEN; i++) {
        printf("%d ", train_data.data[0][i]);
    }
    printf("\nLabel: %d\n", train_data.labels[0]);

    // 메모리 해제
    free_dataset(train_data);
    free_dataset(test_data);

    return 0;
}
