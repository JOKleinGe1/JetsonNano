#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>

#define N 1024 // Taille des matrices N x N

// Fonction pour initialiser une matrice avec des valeurs aléatoires
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Noyau CUDA pour la multiplication de matrices
__global__ void matrixMultiply(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

int main() {
    int size = N * N;
    int matrixSize = size * sizeof(float);

    // Allouer de la mémoire sur l'hôte
    float *h_A = (float *)malloc(matrixSize);
    float *h_B = (float *)malloc(matrixSize);
    float *h_C = (float *)malloc(matrixSize);

    // Initialiser les matrices avec des valeurs aléatoires
    initializeMatrix(h_A, size);
    initializeMatrix(h_B, size);

    // Allouer de la mémoire sur le périphérique
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);
for (int ExperienceNum = 0; ExperienceNum<20; ExperienceNum++){
    // Copier les matrices de l'hôte vers le périphérique
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // Définir la grille et les blocs de threads
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Créer des événements pour mesurer le temps
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Lancer le noyau et mesurer le temps
    cudaEventRecord(start);
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copier le résultat du périphérique vers l'hôte
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // Calculer la performance en GFLOPS
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = flops / (milliseconds * 1e6);
    printf("Temps d'exécution: %.2f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);
}
    // Libérer la mémoire
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

