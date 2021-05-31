#include "ObjDetect.h"
#include <fstream>
#include <opencv2/core/base.hpp>

ObjDetect* ObjDetect::_instance = nullptr;
static const std::string kWinName = "Deep learning object detection in OpenCV";
static float confThreshold = 0.3;
const int width = 416;
const int height = 416;
ObjDetect* ObjDetect::Instance()
{
	if (_instance)
	{
		return _instance;
	}
	else
	{
		assert(false);
		return nullptr;
	}
}

ObjDetect* ObjDetect::createInstance()
{
	if (!_instance)
	{
		_instance = new ObjDetect();
	}
    return _instance;
}

void ObjDetect::setInput(int camera)
{
	videcapture.open(camera);
	if (!videcapture.isOpened())
	{
		assert(false);
	}
    videcapture.set(cv::CAP_PROP_FRAME_WIDTH, 416);
    videcapture.set(cv::CAP_PROP_FRAME_HEIGHT, 416);
}

void ObjDetect::setInput(const std::string& filePath)
{
	videcapture.open(filePath);
	if (!videcapture.isOpened())
	{
		assert(false);
	}
}

void ObjDetect::setClasses(const std::string filepath)
{
    std::ifstream ifs(filepath.c_str());
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "File " + filepath + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
}

void ObjDetect::setImageWidth(const int width)
{
	imageWidth = width;
}

void ObjDetect::setImageHeight(const int height)
{
	imageHeight = height;
}

void ObjDetect::setBoundsOutput(pt::QueueFPS<pt::detectedBounds>& bounds)
{
    this->bounds = &bounds;
}

void ObjDetect::setSkipFrames(bool skip)
{
    this->skipFrames = skip;
}

bool ObjDetect::getWorkingStatus()
{
    return isWorking.load();
}

void ObjDetect::initNet(const std::string modelFilePath, const std::string configFilePath, const int backend, const int target)
{
	net = cv::dnn::readNet(configFilePath, modelFilePath);
    this->backend = backend;
	net.setPreferableBackend(backend);
    this->target = target;
	net.setPreferableTarget(target);
	outNames = net.getUnconnectedOutLayersNames();
}

void ObjDetect::setFps(const int fps)
{
	fpsDelay = 1000 / fps;
}

void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}

void ObjDetect::start()
{
    //cv::startWindowThread();
    //cv::namedWindow(kWinName, cv::WINDOW_AUTOSIZE);
   //int initialConf = (int)(confThreshold * 100);
    //cv::createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);
    process = true;
    this->isWorking.store(true);
    threadPool.push_back(std::thread(&ObjDetect::processingVideoCapture, this));
    threadPool.push_back(std::thread(&ObjDetect::processingObjecting, this));
    threadPool.push_back(std::thread(&ObjDetect::postprocessing, this));
}

void ObjDetect::stop()
{
    process = false;
    predictionsQueue.clear();
}

ObjDetect::ObjDetect():
	process {false},
	mean(0,0,0,0),
	swapRB(true),
	imageHeight(height),
	imageWidth(width),
	scale(1.0 / 255.0),
    nmsThreshold(0.4)
{
	setFps(30);
}

void ObjDetect::deleteInstance()
{
    delete _instance;
}

ObjDetect::~ObjDetect()
{
    for (auto& thread : threadPool)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

void ObjDetect::processingVideoCapture()
{
	cv::Mat frame;
	while (process)
	{
        try
        {
		    videcapture >> frame;
        }
        catch (std::exception exp)
        {
            std::cout << exp.what() << std::endl;
        }
		if (!frame.empty())
		{
			framesQueue.push(frame.clone());
		}	
		else
		{
            process = false;
			break;
		}
		cv::waitKey(fpsDelay);
	}
}

void ObjDetect::processingObjecting()
{
    cv::Mat blob;
    while (isWorking)
    {
        // Get a next frame
        cv::Mat frame;
        {
            if (!framesQueue.empty())
            {
                frame = framesQueue.get();
                if (skipFrames)
                {
                    framesQueue.clear();  // Skip the rest of frames 
                }  
            }
        }
        // Process the frame
        if (!frame.empty())
        {
			preprocessing(frame, net, cv::Size(imageWidth, imageHeight), scale, mean, swapRB);
            processedFramesQueue.push(frame);
			std::vector<cv::Mat> outs;
            auto e1 = cv::getTickCount();
			net.forward(outs, outNames);
            auto e2 = cv::getTickCount();
            auto time = (e2 - e1) / cv::getTickFrequency();
            //std::cout << time << std::endl;
			predictionsQueue.push(outs);
        }
        else if (process == false)
        {
            isWorking.store(false);
        }
    }
}

void ObjDetect::preprocessing(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale, const cv::Scalar& mean, bool swapRB)
{
	static cv::Mat blob;
	// Create a 4D blob from a frame.
	if (inpSize.width <= 0) inpSize.width = frame.cols;
	if (inpSize.height <= 0) inpSize.height = frame.rows;
	cv::dnn::blobFromImage(frame, blob, 1.0, inpSize, cv::Scalar(), swapRB, false, CV_8U);

	// Run a model.
	net.setInput(blob, "", scale, mean);
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		resize(frame, frame, inpSize);
		cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
		net.setInput(imInfo, "im_info");
	}
}

void ObjDetect::postprocessing()
{
    while (cv::waitKey(1) < 0 && isWorking)
    {
        if (predictionsQueue.empty())
            continue;

        std::vector<cv::Mat> outs = predictionsQueue.get();
        cv::Mat frame = processedFramesQueue.get();

        postprocess(frame, outs, net, backend);

        /*if (predictionsQueue.counter > 1)
        {
            std::string label = cv::format("Camera: %.2f FPS", framesQueue.getFPS());
            putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

            label = cv::format("Network: %.2f FPS", predictionsQueue.getFPS());
            putText(frame, label, cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

            label = cv::format("Skipped frames: %d", framesQueue.counter - predictionsQueue.counter);
            putText(frame, label, cv::Point(0, 45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        }
        cv::imshow(kWinName, frame)*/;
    }
}

void ObjDetect::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left = (int)data[i + 3];
                    int top = (int)data[i + 4];
                    int right = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2)
                    {
                        left = (int)(data[i + 3] * frame.cols);
                        top = (int)(data[i + 4] * frame.rows);
                        right = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }
    else
       CV_Error(cv::Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (outLayers.size() > 1 || (outLayerType == "Region" && backend != cv::dnn::DNN_BACKEND_OPENCV))
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;
        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<cv::Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }

    pt::detectedBounds newBounds(frame);
    std::map<int, cv::Rect_<float>> rectangles;
    int addItem = 0;
    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        cv::Rect box = boxes[idx];
        
        if (classIds[idx] == 0)
        {
            auto rect = cv::Rect_<float>(cv::Point_<float>(box.x, box.y), cv::Point_<float>(box.x + box.width, box.y + box.height));
            rectangles.insert(std::make_pair(addItem, rect));
            addItem++;
        }
        
    }
    newBounds.setMapBoxes(rectangles);
    bounds->push(newBounds);
}

void ObjDetect::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = cv::max(top, labelSize.height);
    rectangle(frame, cv::Point(left, top - labelSize.height),
        cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}
