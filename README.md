# JetsonNano
Test software with Jetson Nano baord Cuda programming and CSI-Camera
##hello : 
The very simple "hello cuda" project fo JetsonNano. Compile with "make" and launch with "./test"
##JetsonNano_CSI_Camera : 
Jetson Nano projects for CSI-Camera and Cuda basic programming
###simple_camera.cpp
Connection to CSI-Camera to acquire images, convert color to grayscale and show result in a wihdow, save the last images (before and after processing). 
###image_processing.cpp
Load an image, process each pixel (invert black/white pixel in the half right part), without using Cuda (Kernel) nor Camera.
###cuda_image_processing.cu
Load an image, process each pixel (invert black/white pixel in the half right part) using Cuda Kernel, without Camera.
###cuda_camera.cu : 
Connection to CSI-Camera to acquire images, convert color to grayscale, process each pixel (invert black/white pixel in the half right part) using Cuda Kernel and show result in a wihdow, save the last images (before and after processing). 
