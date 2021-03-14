#pragma once
#include "HelpEntities.cpp"
#include "KalmanTracker.h"

class ObjTracker
{
public:

	ObjTracker(pt::QueueFPS<pt::detectedBounds>& bounds, pt::QueueFPS<pt::detectedBounds>& resultBounds, pt::QueueFPS<pt::detectedBounds>& copyBounds);
	void start();
	~ObjTracker();
private:
	void processing();
	bool process;
	pt::QueueFPS<pt::detectedBounds>* bounds;
	pt::QueueFPS<pt::detectedBounds>* resultBounds;
	pt::QueueFPS<pt::detectedBounds>* copyBounds;
	std::vector <std::thread> threadPool;
};

