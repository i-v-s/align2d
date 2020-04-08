#ifndef ALIGNER_H
#define ALIGNER_H
#include <memory>
#include <opencv2/core.hpp>


struct AlignerParams {
    bool xx, xy, x;
    bool yx, yy, y;
    int levels, max_steps;
    int jakobiType = CV_16S;
};


class Aligner
{
public:
    Aligner()
    {
    }
    typedef std::unique_ptr<Aligner> Ptr;
    static Ptr create(const AlignerParams &params, int channels);
    virtual cv::Mat align(const cv::Mat &a, const cv::Mat &b) = 0;
};

#endif // ALIGNER_H
