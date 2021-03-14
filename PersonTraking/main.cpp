#include <iostream>
#include "ObjTracker.h"
#include "ObjDetect.h"

const int CNUM = 20;

int main()
{
	std::string modelPath = "C:/Users/mitsk/source/repos/OpenCV/x64/Release/yolov4-tiny.weights";
	std::string configPath = "C:/Users/mitsk/source/repos/OpenCV/x64/Release/yolov4-tiny.cfg";
	pt::QueueFPS<pt::detectedBounds> bounds;
	pt::QueueFPS<pt::detectedBounds> resultBounds;

	pt::QueueFPS<pt::detectedBounds> copyBounds;
	auto objDetecter = ObjDetect::createInstance(bounds);
	objDetecter->initNet(modelPath, configPath, cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::DNN_TARGET_OPENCL_FP16);
	objDetecter->setInput(0);
	objDetecter->setClasses("C:/Users/mitsk/source/repos/OpenCV/x64/Release/coco.names");
	objDetecter->start();
	
	ObjTracker tracker(bounds, resultBounds, copyBounds);
	tracker.start();

	// Create a window
	static const std::string kWinName = "Deep learning object detection in OpenCV";
	cv::startWindowThread();
	namedWindow(kWinName, WINDOW_NORMAL);

	// 0. randomly generate colors, only for display
	cv::RNG rng(0xFFFFFFFF);
	cv::Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
	vector<Rect_<float>> detFrameData;
	cv::startWindowThread();
	while (cv::waitKey(1) < 0)
	{
		if (resultBounds.size() == 0)
		{
			continue;
		}

		auto frameWithBounds = resultBounds.get();
		auto bounds = frameWithBounds.getMapWithBoxes();
		auto frame = frameWithBounds.getFrame();
		for (const auto& box : bounds)
		{
			cv::rectangle(frame, box.second, randColor[box.first % CNUM], 2, 8, 0);
		}
		cv::imshow(kWinName, frame);
	}
	ObjDetect::deleteInstance();
	return 0;
}