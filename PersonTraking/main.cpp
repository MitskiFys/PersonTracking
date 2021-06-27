#include <iostream>
#include "ObjTracker.h"
#include "ObjDetect.h"
#include "HumanIdentification.h"
#include <QtCore/qstring.h>
#include <QtCore/QCoreApplication>
#include "NetworkManager.h"
#include "WebSocket.h"
const int CNUM = 20;


namespace
{
	const std::string _sourcePath = "C:/Develop/PersonTraking/Market-1501-v15.09.15/findHere/img_%04d.jpg";
	const std::string _modelPath = "C:/Develop/PersonTraking/models/yolov4-tiny.weights";
	const std::string _configPath = "C:/Develop/PersonTraking/models/yolov4-tiny.cfg";
	const std::string _classesPath = "C:/Develop/PersonTraking/models/coco.names";
	const auto _backend = cv::dnn::DNN_BACKEND_DEFAULT;
	const auto _target = cv::dnn::DNN_TARGET_OPENCL;
}

int main(int argc, char* argv[])
{
	QCoreApplication a(argc, argv);
	QWebSocket webSocket;
	EchoClient client(webSocket, true);
	webSocket.open(QUrl("ws://10.110.15.232:8765/request"));
	while (!client.isGetHttpPort())
	{
		cv::waitKey(100);
	}

	pt::QueueFPS<cv::Mat> frameSourse;
	NetworkManager net(frameSourse);
	net.setAccessToken(client.getAccesToken());
	net.setHttpPort(client.getHttpPort());
	net.makeGetRequest();
	cv::waitKey(100);


	pt::QueueFPS<pt::detectedBounds> bounds;
	pt::QueueFPS<pt::detectedBounds> resultBounds;
	pt::QueueFPS<pt::detectedBounds> copyBounds;

	auto objDetecter = ObjDetect::createInstance();
	objDetecter->initNet(_modelPath, _configPath, _backend, _target);
	objDetecter->setInput(frameSourse);
	objDetecter->setSkipFrames(true);
	objDetecter->setClasses(_classesPath);
	objDetecter->setBoundsOutput(bounds);
	
	auto humanIdentification = HumanIdentification();
	humanIdentification.readPretrainedModel("trainingData.yml");
	//humanIdentification.trainZeroFile();

	objDetecter->start();
	ObjTracker tracker(bounds, resultBounds, copyBounds);
	tracker.start();

	// Create a window
	static const std::string kWinName = "Deep learning object detection in OpenCV";
	//cv::startWindowThread();
	namedWindow(kWinName, WINDOW_NORMAL);

	// 0. randomly generate colors, only for display
	cv::RNG rng(0xFFFFFFFF);
	cv::Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
	vector<Rect_<float>> detFrameData;
	cv::startWindowThread();
	
	bool ptzMove = false;


	while (cv::waitKey(1) < 0)
	{
		if (resultBounds.size() == 0)
		{
			continue;
		}

		auto frameWithBounds = resultBounds.get();
		auto bounds = frameWithBounds.getMapWithBoxes();
		auto frame = frameWithBounds.getFrame();

		int personId = -1;
		static bool isFind = false;

		if (!isFind)
		{
			humanIdentification.findPersonOnImage(frameWithBounds, false);
		}

		if (humanIdentification.isFindPerson())
		{
			isFind = true;
			personId = humanIdentification.getPersonId();
		}

		if (isFind)
		{
			bool isFindPerson = bounds.count(personId);
			if (!isFindPerson)
			{
				isFind = false;
				humanIdentification.personIsLost();
			}
		}

		static bool ptzMove = false;
		if (ptzMove && bounds.size() == 0)
		{
			client.ptzStop();
			ptzMove = false;
		}
		for (const auto& box : bounds)
		{
			cv::rectangle(frame, box.second, randColor[isFind ? box.first == personId ? 1 : 5 : 5], 2, 8, 0);

			if (isFind && box.first == personId)
			{
				const int xCent = box.second.x + (box.second.width / 2);
				const int yCent = box.second.y + (box.second.height / 2);
				cv::circle(frame, cv::Point(xCent, yCent), 10, (0, 0, 255), -1);
				if (xCent < 480)
				{
					client.ptzLeft();
					ptzMove = true;
				}
				else if (xCent > 1440)
				{
					client.ptzRight();
					ptzMove = true;
				}
				else
				{
					if (ptzMove)
					{
						client.ptzStop();
						ptzMove = false;
					}
				}
			}
		}

		cv::rectangle(frame, cv::Rect(480, 270, 960, 540), randColor[1 % CNUM], 2, 8, 0);

		cv::imshow(kWinName, frame);
	}
	ObjDetect::deleteInstance();
	return 0;
}