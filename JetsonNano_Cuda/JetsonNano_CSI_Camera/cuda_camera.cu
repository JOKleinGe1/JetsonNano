// simple_camera.cpp
// MIT License
// Copyright (c) 2019-2022 JetsonHacks
// See LICENSE for OpenCV license and additional information
// Using a CSI camera (such as the Raspberry Pi Version 2) connected to a 
// NVIDIA Jetson Nano Developer Kit using OpenCV
// Drivers for the camera and OpenCV are included in the base image

#include <opencv2/opencv.hpp>
#define ROWS_NUM 720
#define COLS_NUM 1280


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


using namespace cv;

std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main()
{
    int capture_width = COLS_NUM ;
    int capture_height = ROWS_NUM ;
    int display_width = COLS_NUM ;
    int display_height = ROWS_NUM ;
    int rows = capture_height;
    int cols = capture_width;

    int framerate = 30 ;
    int flip_method = 0 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
	capture_height,
	display_width,
	display_height,
	framerate,
	flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";
 
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }

   cv::namedWindow("Input Video", cv::WINDOW_AUTOSIZE);
   cv::namedWindow("Output Video", cv::WINDOW_AUTOSIZE);
   cv::Mat inputImageMat;
   cv::Mat outputImageMat ;

    std::cout << "Hit ESC to exit" << "\n" ;
    /* alloue la memoire pour CUDA */
 
    // Allouer de la mémoire sur le GPU
    uchar* d_inputImage;
    uchar* d_outputImage;
    cudaMalloc((void**)&d_inputImage, rows * cols * sizeof(uchar));
    cudaMalloc((void**)&d_outputImage, rows * cols * sizeof(uchar));

    while(true)
    {
    	if (!cap.read(inputImageMat)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	}
	// outputImageMat = inputImageMat.clone();
	cv::cvtColor(inputImageMat,inputImageMat,COLOR_BGR2GRAY);
	outputImageMat = inputImageMat.clone();

/* ******************************start of image processing******************/
     // Copier les données de l'image vers le GPU
    cudaMemcpy(d_inputImage, inputImageMat.data, rows * cols * sizeof(uchar), cudaMemcpyHostToDevice);

    // Configurer la grille et les blocs pour le noyau CUDA
    dim3 blockSize(32, 32);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Lancer le noyau CUDA
    invertRightHalfKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, rows, cols);

    // Copier les données de l'image de sortie du GPU vers le CPU
    cudaMemcpy(outputImageMat.data, d_outputImage, rows * cols * sizeof(uchar), cudaMemcpyDeviceToHost);

/**********end of image processing************** */

	cv::imshow("Input Video",inputImageMat);
	cv::imshow("Output Video",outputImageMat);
	int keycode = cv::waitKey(10) & 0xff ; 
        if (keycode == 27) break ;
    }
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    cap.release();
    cv::imwrite("input_image.png", inputImageMat);
    cv::imwrite("output_image.png", outputImageMat);
    cv::destroyAllWindows() ;
    return 0;

 }


