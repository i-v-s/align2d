#ifndef PYRAMID_H
#define PYRAMID_H
#include <vector>
#include <opencv2/core.hpp>
#include "utils.h"


template<typename T>
class Pyramid
{
    size_t levelCount;

public:
    std::vector<cv::Mat> levels;
    Pyramid(size_t levelCount) : levelCount(levelCount) {}
    void update(const cv::Mat &image) {
        if(levels.empty()) {
            levels.push_back(image);
            int cols = image.cols, rows = image.rows, type = image.type();
            for (int l = 1; l < levelCount; ++l) {
                cols /= 2;
                rows /= 2;
                levels.push_back(cv::Mat(rows, cols, type));
            }
        }
        levels[0] = image;
        for (int l = 1; l < levelCount; ++l) {
            shrink<T>(levels[l - 1], levels[l]);
        }
    }
    inline bool empty() const {return levels.empty();}
    inline cv::Mat & operator[](int i) {return levels[i];}
};

#endif // PYRAMID_H
