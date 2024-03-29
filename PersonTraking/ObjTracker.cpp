#include "ObjTracker.h"
#include "HungarianAlgorithm.h"

namespace
{

	const double iouThreshold = 0.1;

	const int min_hits = 3;
}

// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

ObjTracker::ObjTracker(pt::QueueFPS<pt::detectedBounds>& bounds, pt::QueueFPS<pt::detectedBounds>& resultBounds, pt::QueueFPS<pt::detectedBounds>& copyBounds):
	process{false}
{
	this->bounds = &bounds;
	this->resultBounds = &resultBounds;
	this->copyBounds = &copyBounds;
}

void ObjTracker::start()
{
	process = true;
	threadPool.push_back(std::thread(&ObjTracker::processing, this));
}

ObjTracker::~ObjTracker()
{
	for (auto& thread : threadPool)
	{
		if (thread.joinable())
		{
			thread.join();
		}
	}
}

void ObjTracker::processing()
{
	std::vector<KalmanTracker> trackers;
	int64 start_time = 0;
	int frame_count = 0;
	int max_age = 1;
	double cycle_time = 0.0;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	vector<Rect_<float>> predictedBoxes;
	vector<Rect_<float>> detFrameData;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	vector<cv::Point> matchedPairs;


	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;


	std::map<int, cv::Rect_<float>> frameTrackingResult;

	while (cv::waitKey(1) < 0 && process)
	{
		if (bounds->size() == 0)
		{
			continue;
		}
		frame_count++;
		start_time = getTickCount();

		pt::detectedBounds detOneFrameData = bounds->get();
		std::map<int, cv::Rect_<float>> framesBoxes = detOneFrameData.getMapWithBoxes();

		detFrameData.clear();
		for (std::map<int, cv::Rect_<float>>::iterator it = framesBoxes.begin(); it != framesBoxes.end(); ++it) {
			detFrameData.push_back(it->second);
		}

		if (trackers.size() == 0) // the first frame met
		{
			// initialize kalman trackers using first detections.
			for (const auto& iter : framesBoxes)
			{
				KalmanTracker trk = KalmanTracker(iter.second);
				trackers.push_back(trk);
			}
			// output the first frame detections
			std::map<int, cv::Rect_<float>> rectangles;
			for (unsigned int id = 0; id < framesBoxes.size(); id++)
			{
				auto rect = cv::Rect_<float>(cv::Point_<float>(framesBoxes[id].x, framesBoxes[id].y), cv::Point_<float>(framesBoxes[id].x + framesBoxes[id].width, framesBoxes[id].y + framesBoxes[id].height));
				rectangles.insert(std::make_pair(id, rect));
			}
			detOneFrameData.setMapBoxes(rectangles);
			resultBounds->push(detOneFrameData);
			continue;
		}

		///////////////////////////////////////
		// 3.1. get predicted locations from existing trackers.
		predictedBoxes.clear();

		for (auto it = trackers.begin(); it != trackers.end();)
		{
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0)
			{
				predictedBoxes.push_back(pBox);
				it++;
			}
			else
			{
				it = trackers.erase(it);
				//cerr << "Box invalid at frame: " << frame_count << endl;
			}
		}

		///////////////////////////////////////
		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		trkNum = predictedBoxes.size();
		detNum = detFrameData.size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[j]);
			}
		}

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else
			;

		// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
			{
				std::cout << "delete " << 1 - iouMatrix[i][assignment[i]] << std::endl;
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}

		///////////////////////////////////////
		// 3.3. updating trackers

		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(detFrameData[detIdx]);
		}

		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(detFrameData[umd]);
			trackers.push_back(tracker);
		}

		// get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
			{
				frameTrackingResult.insert(std::make_pair((*it).m_id + 1, (*it).get_state()));
				//std::cout << (*it).m_id + 1 << " " << (*it).get_state().tl() << std::endl;
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}
		cycle_time = (double)(getTickCount() - start_time);

		detOneFrameData.setMapBoxes(frameTrackingResult);
		resultBounds->push(detOneFrameData);
	}
}
