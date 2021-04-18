#pragma once
#include "ObjDetect.h"
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>

class HumanIdentification
{
public: 
	HumanIdentification(ObjDetect* objDetect);
	void setSource(const std::string& source);
	void setSource(const int camera);
	void startTraining();
private:
	void extractFeatures(std::vector<cv::Mat> batch);
	void setSourcebounds(pt::QueueFPS<pt::detectedBounds>& bounds);
	std::vector<std::vector<float>> features;
	cv::dnn::Net net;
	pt::QueueFPS<pt::detectedBounds> bounds;
	ObjDetect* objDetect;
};

