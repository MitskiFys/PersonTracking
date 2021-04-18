#include "HumanIdentification.h"
#include <thread>
#include <chrono>

namespace
{
	const int _batchSize = 32;
	const int _resizeHeight = 256;
	const int _resizeWidth = 128;
	cv::Mat preprocess(const cv::Mat& img)
	{
		const double mean[3] = { 0.485, 0.456, 0.406 };
		const double std[3] = { 0.229, 0.224, 0.225 };//https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
		cv::Mat ret = cv::Mat(img.rows, img.cols, CV_32FC3);
		for (int y = 0; y < ret.rows; y++)
		{
			for (int x = 0; x < ret.cols; x++)
			{
				for (int c = 0; c < 3; c++)
				{
					ret.at<cv::Vec3f>(y, x)[c] = (float)((img.at<cv::Vec3b>(y, x)[c] / 255.0 - mean[2 - c]) / std[2 - c]);
				}
			}
		}
		return ret;
	}

	cv::Mat getImageInBounds(pt::detectedBounds& inputData)
	{
		auto frames = inputData.getMapWithBoxes();
		if (frames.size() == 1)
		{
			auto bounds = (*frames.begin()).second;
			auto frame = inputData.getFrame();
			if (bounds.x <= 0)
				bounds.x = 0;
			if (bounds.y <= 0)
				bounds.y = 0;
			if (bounds.x + bounds.width > frame.cols)
				bounds.width += frame.cols - (bounds.x + bounds.width);
			if (bounds.y + bounds.height > frame.rows)
				bounds.height += frame.rows - (bounds.y + bounds.height);
			return preprocess(frame(bounds));
		}
		return cv::Mat();
	}

	static std::vector<float> normalization(const std::vector<double>& feature)
	{
		std::vector<float> ret2;
		const auto  minElem = *std::min_element(feature.begin(), feature.end());
		const auto maxElem = *std::max_element(feature.begin(), feature.end());

		for (const auto& elem : feature)
		{
			ret2.push_back((elem - minElem) / (maxElem - minElem));
		}

		return ret2;
	}
}

HumanIdentification::HumanIdentification(ObjDetect* objDetect)
{
	this->objDetect = objDetect;
	setSourcebounds(this->bounds);
	/*net = cv::dnn::readNet("C:/Develop/PersonTraking/PersonTraking/youtu_reid_baseline_lite.onnx");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);*/
}

void HumanIdentification::setSource(const std::string& source)
{
	this->objDetect->setInput(source);
}

void HumanIdentification::setSource(const int camera)
{
	this->objDetect->setInput(camera);
}

void HumanIdentification::startTraining()
{
	objDetect->start();
	std::vector<cv::Mat> batch;
	while (objDetect->getWorkingStatus() /*|| batch.size() > 0*/) 
	{
		cv::waitKey(1);
		//if (!bounds.empty())
		//{
		//	do
		//	{
		//		auto image = getImageInBounds(bounds.get());
		//		if (!image.empty())
		//		{
		//			//cv::imshow("asdasd", image);
		//			//cv::waitKey(1);
		//			batch.push_back(image);
		//		}
		//	} while (batch.size() != _batchSize && !bounds.empty());
		//}
		//if (batch.size() == _batchSize || (!objDetect->getWorkingStatus()))
		//{
		//	//extractFeatures(batch);
		//	batch.clear();
		//}
	}
	std::cout << "Find!" << std::endl;
}

void HumanIdentification::extractFeatures(std::vector<cv::Mat> batch)
{
	cv::Mat blob = cv::dnn::blobFromImages(batch, 1.0, cv::Size(_resizeWidth, _resizeHeight), cv::Scalar(0.0, 0.0, 0.0), true, false, CV_32F);
	net.setInput(blob);
	auto e1 = cv::getTickCount();
	cv::Mat out = net.forward();
	auto e2 = cv::getTickCount();
	auto time = (e2 - e1) / cv::getTickFrequency();
	std::cout << time << std::endl;
	for (int i = 0; i < (int)out.size().height; i++)
	{
		std::vector<double> temp_feature;
		for (int j = 0; j < (int)out.size().width; j++)
		{
			temp_feature.push_back(out.at<float>(i, j));
		}
		//normal feature
		features.push_back(normalization(temp_feature));
	}
}

void HumanIdentification::setSourcebounds(pt::QueueFPS<pt::detectedBounds>& bounds)
{
	objDetect->setBoundsOutput(bounds);
}
