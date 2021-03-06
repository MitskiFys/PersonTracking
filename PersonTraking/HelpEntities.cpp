#pragma once
#include <mutex>
#include <queue>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>

namespace pt
{
    struct detectedBounds
    {
        detectedBounds(int x, int y, int width, int height, cv::Mat person) :
            x(x),
            y(y),
            width(width),
            height(height),
            person(person)
        {
            id = 0;
        }

        int getId() { return id; }
        void setId(int id) { this->id = id; }

    private:

        int x;
        int y;
        int height;
        int width;
        cv::Mat person;
        int id;
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

