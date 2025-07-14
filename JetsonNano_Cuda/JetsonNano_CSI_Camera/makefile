all : image_processing simple_camera cuda_image_processing cuda_camera 

simple_camera : simple_camera.cpp  
	g++ -std=c++11 -Wall -I/usr/lib/opencv -I/usr/include/opencv4 $^ -L/usr/lib  -lopencv_core -lopencv_highgui -lopencv_videoio  -lopencv_imgproc -lopencv_imgcodecs -o $@
image_processing : image_processing.cpp
	g++ -std=c++11 -Wall -I/usr/lib/opencv -I/usr/include/opencv4 $^ -L/usr/lib  -lopencv_core -lopencv_highgui -lopencv_videoio  -lopencv_imgproc -lopencv_imgcodecs -o $@
cuda_image_processing : cuda_image_processing.cu
	 nvcc -I/usr/lib/opencv -I/usr/include/opencv4  -L/usr/lib  -lopencv_core -lopencv_highgui -lopencv_videoio  -lopencv_imgproc -lopencv_imgcodecs  $^ -o  $@

cuda_camera : cuda_camera.cu
	 nvcc -I/usr/lib/opencv -I/usr/include/opencv4  -L/usr/lib  -lopencv_core -lopencv_highgui -lopencv_videoio  -lopencv_imgproc -lopencv_imgcodecs  $^ -o  $@

clean : 
	rm -f image_processing simple_camera cuda_image_processing cuda_camera 
