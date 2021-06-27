#pragma once
#include <mutex>
#include <queue>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>

namespace pt
{
    struct detectedBounds
    {
        detectedBounds(cv::Mat frame) :
            frame(frame)
        {

        }

        void setMapBoxes(std::map<int, cv::Rect_<float>> boxes)
        {
            this->boxes = boxes;
        }

        std::map<int, cv::Rect_<float>>& getMapWithBoxes()
        {
            return boxes;
        }

        cv::Mat& getFrame()
        {
            return frame;
        }

    private:

        std::map<int, cv::Rect_<float>> boxes;
        cv::Mat frame;
    };

    template <typename T>
    class QueueFPS : public std::queue<T>
    {
    public:
        QueueFPS() : counter(0) {}

        void push(const T& entry)
        {
            std::lock_guard<std::mutex> lock(mutex);

            std::queue<T>::push(entry);
            counter += 1;
            if (counter == 1)
            {
                // Start counting from a second frame (warmup).
                tm.reset();
                tm.start();
            }
        }

        T get()
        {
            std::lock_guard<std::mutex> lock(mutex);
            T entry = this->front();
            this->pop();
            return entry;
        }

        float getFPS()
        {
            tm.stop();
            double fps = counter / tm.getTimeSec();
            tm.start();
            return static_cast<float>(fps);
        }

        void clear()
        {
            std::lock_guard<std::mutex> lock(mutex);
            while (!this->empty())
                this->pop();
        }

        unsigned int counter;

    private:
        cv::TickMeter tm;
        std::mutex mutex;
    };
}

