#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#if defined(__linux__) && !defined(__NVCC__)
#include <linux/time.h>
#else
#include <time.h>
#endif

// SIZE * SIZE の行列を生成
// これを変数にするとすごく遅くなる。引数で可変にしようとか思わないように。
#ifndef SIZE
#define SIZE 1024
#endif

#define MAX_VALUE 100 // 行列の各要素の最大値。実際の最大値はMAX_VALUE-1

#ifdef __NVCC__
  #ifndef THREAD
  #define THREAD 16
  #endif
#endif

#ifndef __NVCC__
  #ifdef __linux__
  typedef __clockid_t clockid_t;
  #endif
extern int clock_gettime(clockid_t clk_id, struct timespec *tp);
#else
int block_num  = SIZE / THREAD; // この変数はmain()とCUDA版のto_squaring()で使用するが、to_squaring()の引数をCPU版とCUDA版で揃えたいためグローバルで宣言
#endif

typedef int matrix_t;

int mem_size = sizeof(matrix_t) * SIZE * SIZE; // 行列一つに必要なメモリサイズ

void print_matrix(matrix_t* matrix, int digit) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%*d ", digit, matrix[i * SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init_matrix(matrix_t* matrix, matrix_t* matrix_ans) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i * SIZE + j] = rand() % MAX_VALUE;
            matrix_ans[i * SIZE + j] = 0;
        }
    }
}

int get_digit(int num) {
    return log10(num) + 1;
}

double get_time() {
    struct timespec time;

    if (clock_gettime(CLOCK_REALTIME, &time) < 0) {
        perror("error: clock_gettime");
        exit(1);
    }

    double sec  = time.tv_sec;
    double nsec = time.tv_nsec / 1000000000.0; // ナノ秒なので10億で割る

    return sec + nsec;
}

#ifdef __NVCC__

__global__
void kernel(matrix_t* matrix, matrix_t* matrix_ans) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= SIZE || j >= SIZE) { return; }

    for (int k = 0; k < SIZE; k++) {
        matrix_ans[i + SIZE * j] += matrix[i + SIZE * k] * matrix[k + SIZE * j];
    }
}

static inline void to_squaring(matrix_t* matrix, matrix_t* matrix_ans) {
    matrix_t* d_matrix;
    matrix_t* d_matrix_ans;

    cudaMalloc((void**)&d_matrix,     mem_size);
    cudaMalloc((void**)&d_matrix_ans, mem_size);
    cudaMemcpy(d_matrix,     matrix,     mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_ans, matrix_ans, mem_size, cudaMemcpyHostToDevice);

    if (SIZE % THREAD != 0) { block_num++; }

    dim3 thread = dim3(THREAD, THREAD, 1);
    dim3 block  = dim3(block_num,  block_num,  1);
    kernel<<<block, thread>>>(d_matrix, d_matrix_ans);
    cudaThreadSynchronize();

    cudaMemcpy(matrix_ans, d_matrix_ans, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_matrix_ans);
}

#else

static inline void to_squaring(matrix_t* matrix, matrix_t* matrix_ans) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                matrix_ans[i * SIZE + j] += matrix[i * SIZE + k] * matrix[k * SIZE + j];
            }
        }
    }
}

#endif

int main(int argc, char* argv[]) {
    bool do_print = false;
    if (argc == 2) {
        if (strcmp(argv[1], "-p") == 0) {
            do_print = true;
        }
    }

    matrix_t* matrix     = (matrix_t*)malloc(mem_size);
    matrix_t* matrix_ans = (matrix_t*)malloc(mem_size);

    init_matrix(matrix, matrix_ans);

    if (do_print) {
        printf("元の行列\n");
        print_matrix(matrix, get_digit(MAX_VALUE));
    }

    double begin = get_time();

    to_squaring(matrix, matrix_ans);

    double end = get_time();

    free(matrix);

    if (do_print) {
        printf("2乗後の行列\n");
        print_matrix(matrix_ans, get_digit(MAX_VALUE) * 2);
    }

    int top = matrix_ans[0];
    int tail = matrix_ans[(SIZE - 1) * SIZE + (SIZE - 1)];
    free(matrix_ans);

    #ifdef __NVCC__
    printf("Use CUDA: THREAD=%d, BLOCK=%d\n", THREAD, block_num);
    #else
    printf("Use CPU only\n");
    #endif
    printf("Matrix size: %d * %d\n", SIZE, SIZE);
    printf("matrix_ans[0][0]       = %d\n", top);
    printf("matrix_ans[SIZE][SIZE] = %d\n", tail);
    printf("\nProcessing time: %lf sec\n", end - begin);

    if (top <= 0 || tail <= 0) {
        fprintf(stderr, "error: calculation fails\n");
        exit(1);
    }
    return 0;
}
