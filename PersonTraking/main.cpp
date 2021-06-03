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
	
}

void threadWebSocketConnect()
{

}

int main(int argc, char* argv[])
{
	QCoreApplication a(argc, argv);
	QWebSocket webSocket;
	EchoClient client(webSocket, true);
	webSocket.open(QUrl("ws://127.0.0.1:8765/request"));
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



	std::string modelPath = "C:/Develop/PersonTraking/models/yolov4-tiny.weights";
	std::string configPath = "C:/Develop/PersonTraking/models/yolov4-tiny.cfg";
	pt::QueueFPS<pt::detectedBounds> bounds;
	pt::QueueFPS<pt::detectedBounds> resultBounds;
	pt::QueueFPS<pt::detectedBounds> copyBounds;

	auto objDetecter = ObjDetect::createInstance();
	objDetecter->initNet(modelPath, configPath, cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::DNN_TARGET_OPENCL_FP16);
	objDetecter->setInput(frameSourse);
	objDetecter->setSkipFrames(true);
	objDetecter->setClasses("C:/Develop/PersonTraking/models/coco.names");
	objDetecter->setBoundsOutput(bounds);
	objDetecter->start();
	
	//auto humanIdentification = HumanIdentification(ObjDetect::Instance());
	//humanIdentification.trainZeroFile();
	//humanIdentification.setSource("C:/Develop/PersonTraking/Market-1501-v15.09.15/filtredImages/img_%04d.jpg");

	//humanIdentification.startTraining();

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

		if (!bounds.empty())
		{
			const auto firstBounds = *bounds.begin();

			const int xCent = firstBounds.second.x + (firstBounds.second.width / 2);
			const int yCent = firstBounds.second.y + (firstBounds.second.height / 2);
			
			cv::circle(frame, cv::Point(xCent, yCent), 10, (0, 0, 255), -1);
			if (xCent < 515)
			{
				client.ptzRight();
				ptzMove = true;
			}
			else if (xCent > 765)
			{
				client.ptzLeft();
				ptzMove = true;
			}
			else if (yCent < 235)
			{
				client.ptzUp();
				ptzMove = true;
			}
			else if (yCent > 485)
			{
				client.ptzDown();
				ptzMove = true;
			}
			else
			{
				if (ptzMove)
				{
					client.ptzStop();
				}
			}

		}
		else
		{
			if (ptzMove)
			{
				client.ptzStop();
			}
		}

		

		for (const auto& box : bounds)
		{
			cv::rectangle(frame, box.second, randColor[box.first % CNUM], 2, 8, 0);
		}

		cv::rectangle(frame, cv::Rect(515, 235, 250, 250), randColor[1 % CNUM], 2, 8, 0);

		cv::imshow(kWinName, frame);
	}
	ObjDetect::deleteInstance();
	return 0;
}