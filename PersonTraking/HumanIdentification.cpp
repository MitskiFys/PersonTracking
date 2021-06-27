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

	cv::Mat getImageInBounds(const cv::Mat& frame, cv::Rect_<float> bounds)
	{
		if (bounds.empty())
		{
			return cv::Mat();
		}
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

	static float similarity(const std::vector<float>& feature1, const std::vector<float>& feature2)
	{
		float result = 0.0;
		for (int i = 0; i < (int)feature1.size(); i++)
		{
			result += feature1[i] * feature2[i];
		}
		return result;
	}

	static float avg(std::vector<float>& v)
	{
		float return_value = 0.0;
		int n = v.size();

		for (int i = 0; i < n; i++)
		{
			return_value += v[i];
		}

		return (return_value / n);
	}

	static void getTopK(const std::vector<std::vector<float>>& queryFeatures, const std::vector<std::vector<float>>& galleryFeatures, const int& topk, std::vector<std::vector<int>>& result)
	{
		for (int i = 0; i < (int)queryFeatures.size(); i++)
		{
			std::vector<float> similarityList;
			std::vector<int> index;
			for (int j = 0; j < (int)galleryFeatures.size(); j++)
			{
				similarityList.push_back(similarity(queryFeatures[i], galleryFeatures[j]));
				index.push_back(j);
			}
			const auto avarage = avg(similarityList);
			sort(index.begin(), index.end(), [&](int x, int y) {return similarityList[x] > similarityList[y]; });
			std::vector<int> topk_result;
			for (int j = 0; j < cv::min(topk, (int)index.size()); j++)
			{
				topk_result.push_back(index[j]);
			}
			result.push_back(topk_result);
		}
		return;
	}
}

HumanIdentification::HumanIdentification(ObjDetect* objDetect):
	m_threadIsWorking(false),
	m_personId(0),
	m_isFind(false)
{
	if (objDetect)
	{
		this->objDetect = objDetect;
		setSourcebounds(this->bounds);
	}
	net = cv::dnn::readNet("C:/Develop/PersonTraking/PersonTraking/youtu_reid_baseline_lite.onnx");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	m_classifierSVM = cv::ml::SVM::create();
	m_classifierSVM->setType(cv::ml::SVM::C_SVC);
	m_classifierSVM->setKernel(cv::ml::SVM::RBF);

	m_classifierSVM->setTermCriteria(cv::TermCriteria(cv::TermCriteria::Type::MAX_ITER,
		1000, 1e-5));
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
		if (!bounds.empty())
		{
			do
			{
				auto e1 = cv::getTickCount();
				auto image = getImageInBounds(bounds.get());
				auto e2 = cv::getTickCount();
				auto time = (e2 - e1) / cv::getTickFrequency();
				std::cout << time << std::endl;
				if (!image.empty())
				{
					cv::imshow("asdasd", image);
					cv::waitKey(300);
					batch.push_back(image);
				}
			} while (batch.size() != _batchSize && !bounds.empty());
		}
		if (batch.size() == _batchSize || (!objDetect->getWorkingStatus()))
		{
			extractFeatures(batch);
			batch.clear();
		}
	}
	std::cout << "Find!" << std::endl;
}

void HumanIdentification::trainSVMModelFromPrepairedImages(std::vector<std::string>& filepath)
{
	auto rowCount = 0;
	auto columnCount = 0;
	auto group = 0;
	cv::Mat groups;
	std::vector<std::vector<std::vector<float>>> vecOfFeatures;
	for (const auto& path : filepath)
	{
		extractFeaturesfromPrepairedImage(path);
		rowCount += m_features.size();
		columnCount = m_features.at(0).size();
		for (int j = 0; j < m_features.size(); j++)
		{
			groups.push_back(group);
		}
		vecOfFeatures.push_back(m_features);
		m_features.clear();
		++group;
	}

	cv::Mat samples(rowCount, columnCount, CV_32F);

	int filledRows = 0;
	for (const auto& vecFeature : vecOfFeatures)
	{
		for (int i = filledRows; i < vecFeature.size() + filledRows; i++)
		{
			for (int j = 0; j < columnCount; j++)
			{
				samples.at<float>(i, j) = vecFeature.at(i - filledRows).at(j);
			}
		}
		filledRows += vecFeature.size();
	}


	m_classifierSVM->trainAuto(samples, cv::ml::ROW_SAMPLE, groups);
	m_classifierSVM->save("trainingData.yml");
}


void HumanIdentification::trainZeroFile()
{
	std::vector<std::string> paths;
	paths.push_back("C:/Develop/PersonTraking/Market-1501-v15.09.15/filtredImages/img_%04d.jpg");
	paths.push_back("C:/Develop/PersonTraking/Market-1501-v15.09.15/novikov_train/img_%04d.jpg");
	trainSVMModelFromPrepairedImages(paths);

	extractFeaturesfromPrepairedImage("C:/Develop/PersonTraking/Market-1501-v15.09.15/novikov_test/img_%04d.jpg");

	for (const auto& feature : m_features)
	{
		std::cout << m_classifierSVM->predict(feature) << std::endl;
	}
	m_features.clear();
}



void HumanIdentification::extractFeaturesfromPrepairedImage(const std::string& filepath)
{
	cv::VideoCapture videcapture;
	videcapture.open(filepath);
	if (!videcapture.isOpened())
	{
		assert(false);
	}
	bool isHaveImage = true;
	cv::Mat frame;
	std::vector<cv::Mat> batch;
	m_features.clear();
	while (isHaveImage)
	{
		try
		{
			isHaveImage = videcapture.read(frame);
		}
		catch (std::exception exp)
		{
			isHaveImage = false;
			std::cout << exp.what() << std::endl;
		}
		if (!frame.empty())
		{
			if (batch.size() < _batchSize)
			{
				batch.push_back(preprocess(frame));
			}
			else
			{
				extractFeatures(batch);
				batch.clear();
			}
		}
	}
	if (batch.size() > 0)
	{
		extractFeatures(batch);
		batch.clear();
	}

}

void HumanIdentification::extractFeatures(std::vector<cv::Mat>& batch)
{
	cv::Mat blob = cv::dnn::blobFromImages(batch, 1.0, cv::Size(_resizeWidth, _resizeHeight), cv::Scalar(0.0, 0.0, 0.0), true, false, CV_32F);
	net.setInput(blob);
	auto e1 = cv::getTickCount();
	cv::Mat out;
	try
	{
		out = net.forward();
	}
	catch (...)
	{
		return;
	}
	auto e2 = cv::getTickCount();
	auto time = (e2 - e1) / cv::getTickFrequency();
	std::cout << "Extract feature time - " <<  time << std::endl;
	auto test = out.size;
	for (int i = 0; i < (int)out.size[0]; i++)
	{
		std::vector<double> temp_feature;
		for (int j = 0; j < (int)out.size[1]; j++)
		{
			temp_feature.push_back(out.at<float>(i, j, 0));
		}
		//normal feature
		m_features.push_back(normalization(temp_feature));
	}
}

void HumanIdentification::setSourcebounds(pt::QueueFPS<pt::detectedBounds>& bounds)
{
	objDetect->setBoundsOutput(bounds);
}

void HumanIdentification::trainClassifier()
{

}

void HumanIdentification::threadFunctionFindPerson(pt::detectedBounds frameWithBounds)
{
	if (m_isFind.load())
		return;
	m_threadIsWorking.store(true);
	const auto frame = frameWithBounds.getFrame();
	const auto bounds = frameWithBounds.getMapWithBoxes();
	if (bounds.empty())
	{
		return;
	}
	std::vector<cv::Mat> person;
	std::vector<int> numeration;
	for (const auto& bound : bounds)
	{
		numeration.push_back(bound.first);
		person.push_back(getImageInBounds(frame, bound.second));
	}
	extractFeatures(person);
	if (m_features.empty())
	{
		return;
	}
	auto iter = 0;
	for (const auto& feature : m_features)
	{
		const auto id = m_classifierSVM->predict(feature);
		if (id == 1)
		{
			m_isFind.store(true);
			m_personId.store(numeration[iter]);
			m_threadIsWorking.store(false);
			return;
		}
		iter++;
	}

	m_features.clear();
	m_isFind.store(false);
	m_threadIsWorking.store(false);
}

void HumanIdentification::findPersonOnImage(pt::detectedBounds& frameWithBounds, bool inMainthread)
{
	if (!m_threadIsWorking.load())
	{
		auto thread = std::thread(&HumanIdentification::threadFunctionFindPerson, this, frameWithBounds);
		if (inMainthread)
		{
			thread.join();
		}
		else
		{
			thread.detach();
		}
	}
}

bool HumanIdentification::isFindPerson()
{
	return m_isFind.load();
}

void HumanIdentification::personIsLost()
{
	m_isFind.store(false);
}

int HumanIdentification::getPersonId()
{
	return m_personId.load();
}

void HumanIdentification::readPretrainedModel(const std::string& filepath)
{
	m_classifierSVM = m_classifierSVM->load(filepath);
}
