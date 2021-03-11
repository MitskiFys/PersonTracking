#include <iostream>
#include "HelpEntities.cpp"
#include "ObjDetect.h"
	
int main()
{
	std::string modelPath = "C:/Users/mitsk/source/repos/OpenCV/x64/Release/yolov4-tiny.weights";
	std::string configPath = "C:/Users/mitsk/source/repos/OpenCV/x64/Release/yolov4-tiny.cfg";
	pt::QueueFPS<pt::detectedBounds> bounds;
	auto objDetecter = ObjDetect::createInstance(bounds);
	objDetecter->initNet(modelPath, configPath, cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::DNN_TARGET_OPENCL);
	objDetecter->setInput(0);
	objDetecter->setClasses("C:/Users/mitsk/source/repos/OpenCV/x64/Release/coco.names");
	objDetecter->start();
	std::cout << "end" << std::endl;
	ObjDetect::deleteInstance();
	return 0;
}