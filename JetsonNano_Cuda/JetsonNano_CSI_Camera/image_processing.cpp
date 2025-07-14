// image_processing.cpp

#include <opencv2/opencv.hpp>
#include <iostream> 

using namespace cv;

int main()
{
    cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("output image", cv::WINDOW_AUTOSIZE);
    cv::Mat inputImageMat, outputImageMat;
    inputImageMat = cv::imread("img2.png", 0);
    std::cout << "inputImageMat.size (rxc): "<< inputImageMat.size<<"\n";

    outputImageMat = inputImageMat.clone();

    for (int col = 0; col<outputImageMat.cols; col++)
	for (int row = 0; row<outputImageMat.rows; row++)
		if(col> inputImageMat.cols/2) outputImageMat.at<uchar>(row,col) = (uchar) (255- outputImageMat.at<uchar>(row,col));


    cv::imshow("input image",inputImageMat);
    cv::imshow("output image",outputImageMat);
   
    while(true)
    {
  	int keycode = cv::waitKey(10) & 0xff ; 
        if (keycode == 27) break ;
    }

    cv::imwrite("img3.png", outputImageMat);
    cv::destroyAllWindows() ;
    return 0;
}


