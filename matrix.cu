#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__linux__) && !defined(__NVCC__)
#include <linux/time.h>
#else
#include <time.h>
#endif

#if defined(__linux__) && !defined(__NVCC__)
typedef __clockid_t clockid_t;
#endif

//#define PRINT_VALUE // 行列の内容を出力

// SIZE * SIZE の行列を生成
// これを変数にするとすごく遅くなる。引数で可変にしようとか思わないように。
#ifndef SIZE
#define SIZE 1024
#endif
#define MEM_SIZE SIZE * SIZE

#define MAX_VALUE 100 // 行列の各要素の最大値。実際の最大値はMAX_VALUE-1

#ifdef __NVCC__
#ifndef THREAD
#define THREAD 16
#endif
#define BLOCK (SIZE / THREAD)
#endif

#ifndef __NVCC__
extern int clock_getres(clockid_t clk_id, struct timespec *res);
extern int clock_gettime(clockid_t clk_id, struct timespec *tp);
#endif

void init_matrix(int *matrix, int *matrix_ans) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i * SIZE + j] = rand() % MAX_VALUE;
            matrix_ans[i * SIZE + j] = 0;
        }
    }
}

#ifdef PRINT_VALUE

int get_digit(int num) {
    return log10(num) + 1;
}

void print_matrix(int* matrix, int digit) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%*d ", digit, matrix[i * SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

#endif

#ifdef __NVCC__

__global__
void kernel(int* matrix, int* matrix_ans){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j  = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > SIZE || j > SIZE) { return; }

    for(int k = 0; k < SIZE; k++){
        matrix_ans[i + SIZE * j] += matrix[i + SIZE * k] * matrix[k + SIZE * j];
    }
}

static inline void to_squaring(int *matrix, int *matrix_ans) {
    int* d_matrix;
    int* d_matrix_ans;

    cudaMalloc((void**)&d_matrix,     sizeof(int) * MEM_SIZE);
    cudaMalloc((void**)&d_matrix_ans, sizeof(int) * MEM_SIZE);
    cudaMemcpy(d_matrix,     matrix,     sizeof(int) * MEM_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_ans, matrix_ans, sizeof(int) * MEM_SIZE, cudaMemcpyHostToDevice);

    dim3 thread = dim3(THREAD,THREAD,1);
    dim3 block = dim3(BLOCK ,BLOCK, 1);
    kernel<<<block, thread>>>(d_matrix, d_matrix_ans);
    cudaThreadSynchronize();

    cudaMemcpy(matrix_ans, d_matrix_ans, sizeof(int) * MEM_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_matrix_ans);
}

#else

static inline void to_squaring(int *matrix, int *matrix_ans) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                matrix_ans[i * SIZE + j] += matrix[i * SIZE + k] * matrix[k * SIZE + j];
            }
        }
    }
}

#endif

double get_time() {
    struct timespec time;

    if(clock_gettime(CLOCK_REALTIME,&time) < 0){
        perror("clock_gettime");
        exit(1);
    }

    double sec = time.tv_sec;
    double nsec = time.tv_nsec / 1000000000.0; // ナノ秒なので10億で割る

    return sec + nsec;
}

int main() {
    int  mem_size   = sizeof(int) * SIZE * SIZE;
    int* matrix     = (int*)malloc(sizeof(int) * mem_size);
    int* matrix_ans = (int*)malloc(sizeof(int) * mem_size);

    init_matrix(matrix, matrix_ans);
    #ifdef PRINT_VALUE
    printf("元の行列\n");
    print_matrix(matrix, get_digit(MAX_VALUE));
    #endif

    double begin = get_time();

    to_squaring(matrix, matrix_ans);

    double end = get_time();

    #ifdef PRINT_VALUE
    printf("2乗後の行列\n");
    print_matrix(matrix_ans, (get_digit(MAX_VALUE)) * 2);
    #endif

    free(matrix);
    free(matrix_ans);

    #ifdef __NVCC__
    printf("Use CUDA: THREAD=%d, BLOCK=%d\n",THREAD, BLOCK);
    #else
    printf("Use CPU only\n");
    #endif
    printf("Matrix size: %d * %d\n", SIZE, SIZE);
    printf("Processing time: %lf sec\n",end - begin);

    return 0;
}
