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
	void trainZeroFile();
private:
	void extractFeatures(std::vector<cv::Mat>& batch);
	void extractFeaturesfromPrepairedImage(const std::string& filepath);
	void trainSVMModelFromPrepairedImages(std::vector<std::string>& filepath);
	void setSourcebounds(pt::QueueFPS<pt::detectedBounds>& bounds);
	void trainClassifier();
	std::vector<std::vector<float>> features;
	std::vector<std::vector<float>> otherFeatures;
	cv::Ptr<cv::ml::SVM> classifierSVM;
	cv::dnn::Net net;
	pt::QueueFPS<pt::detectedBounds> bounds;
	ObjDetect* objDetect;
};

