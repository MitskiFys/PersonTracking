#pragma once

#include "HelpEntities.cpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/ocl.hpp"
#include <memory>

class ObjDetect
{
public:

	static ObjDetect* Instance();
	static ObjDetect* createInstance(pt::QueueFPS<pt::detectedBounds>& bounds);
	static void deleteInstance();
	void setInput(int camera);
	void setInput(const std::string& filePath);
	void setClasses(const std::string filepath);
	void setImageWidth(const int width);
	void setImageHeight(const int height);

	void initNet(const std::string modelFilePath, const std::string configFilePath, const int backend, const int target);

	void setFps(const int fps);

	void start();

protected:
	ObjDetect(pt::QueueFPS<pt::detectedBounds>& bounds);
	~ObjDetect();
private:
	ObjDetect() = delete;
	ObjDetect(const ObjDetect& other) = delete;
	ObjDetect& operator=(const ObjDetect& other) = delete;
	ObjDetect(ObjDetect&& other) = delete;
	ObjDetect& operator=(ObjDetect&& other) = delete;

	std::vector <std::thread> threadPool;

	void processingVideoCapture();
	void processingObjecting();
	void preprocessing(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale, const cv::Scalar& mean, bool swapRB);
	void postprocessing();
	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

	pt::QueueFPS<cv::Mat> framesQueue;
	pt::QueueFPS<cv::Mat> processedFramesQueue;
	pt::QueueFPS<std::vector<cv::Mat> > predictionsQueue;
	cv::Scalar mean;
	bool process;
	bool swapRB;
	cv::VideoCapture videcapture;
	cv::dnn::Net net;
	pt::QueueFPS<pt::detectedBounds>* bounds;
	int fpsDelay;
	int imageHeight;
	int imageWidth;
	int backend;
	int target;
	float scale;
	float nmsThreshold;
	std::vector<cv::String> outNames;
	std::vector<std::string> classes;
	static ObjDetect* _instance;
};