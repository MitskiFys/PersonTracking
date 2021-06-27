#pragma once
#include "ObjDetect.h"
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <atomic>

class HumanIdentification
{
public: 
	HumanIdentification(ObjDetect* objDetect = nullptr);
	void setSource(const std::string& source);
	void setSource(const int camera);
	void startTraining();
	void trainZeroFile();
	void findPersonOnImage(pt::detectedBounds& frameWithBounds, bool inMainthread);
	bool isFindPerson();
	void personIsLost();
	int getPersonId();
	void readPretrainedModel(const std::string& filepath);

private:
	void extractFeatures(std::vector<cv::Mat>& batch);
	void extractFeaturesfromPrepairedImage(const std::string& filepath);
	void trainSVMModelFromPrepairedImages(std::vector<std::string>& filepath);
	void setSourcebounds(pt::QueueFPS<pt::detectedBounds>& bounds);
	void trainClassifier();
	void threadFunctionFindPerson(pt::detectedBounds frameWithBounds);

	std::mutex m_mutex;

	std::atomic<bool> m_threadIsWorking;
	std::atomic<int> m_personId;
	std::atomic<bool> m_isFind;

	std::vector<std::vector<float>> m_features;
	std::vector<std::vector<float>> otherFeatures;
	cv::Ptr<cv::ml::SVM> m_classifierSVM;
	cv::dnn::Net net;
	pt::QueueFPS<pt::detectedBounds> bounds;
	ObjDetect* objDetect;
};

