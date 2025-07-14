#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// Noyau CUDA pour inverser les pixels de la moitié droite de l'image
__global__ void invertRightHalfKernel(uchar* inputImage, uchar* outputImage, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows) {
        if (col > cols / 2) {
            outputImage[row * cols + col] = (uchar)(255 - inputImage[row * cols + col]);
        } else {
            outputImage[row * cols + col] = inputImage[row * cols + col];
        }
    }
}

int main() {
    cv::Mat inputImageMat = cv::imread("input_image.png", 0);
    cv::Mat outputImageMat = inputImageMat.clone();

    int rows = inputImageMat.rows;
    int cols = inputImageMat.cols;

    // Allouer de la mémoire sur le GPU
    uchar* d_inputImage;
    uchar* d_outputImage;
    cudaMalloc((void**)&d_inputImage, rows * cols * sizeof(uchar));
    cudaMalloc((void**)&d_outputImage, rows * cols * sizeof(uchar));

    // Copier les données de l'image vers le GPU
    cudaMemcpy(d_inputImage, inputImageMat.data, rows * cols * sizeof(uchar), cudaMemcpyHostToDevice);

    // Configurer la grille et les blocs pour le noyau CUDA
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Lancer le noyau CUDA
    invertRightHalfKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, rows, cols);

    // Copier les données de l'image de sortie du GPU vers le CPU
    cudaMemcpy(outputImageMat.data, d_outputImage, rows * cols * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Afficher les images
    cv::imshow("input image", inputImageMat);
    cv::imshow("output image", outputImageMat);

    // Attendre une touche
    while (true) {
        int keycode = cv::waitKey(10) & 0xff;
        if (keycode == 27) break;
    }

    // Sauvegarder l'image de sortie
    cv::imwrite("output_image.png", outputImageMat);

    // Libérer la mémoire du GPU
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Détruire toutes les fenêtres
    cv::destroyAllWindows();

    return 0;
}
