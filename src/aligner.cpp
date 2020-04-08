#include "aligner.h"
#include "pyramid.h"
#include "jacobi.h"


template<class Pixel>
class AlignerT: public Aligner
{
    std::unique_ptr<Pyramid<Pixel>> pa, pb;
    GaussNewton gn;
    std::vector<Jacobi::Ptr> jacobi;
    AlignerParams params;

public:
    AlignerT(const AlignerParams &params):
        pa(std::make_unique<Pyramid<Pixel>>(params.levels)),
        pb(std::make_unique<Pyramid<Pixel>>(params.levels)),
        gn(0.1, params.max_steps),
        params(params)
    {

    }
    cv::Mat align(const cv::Mat &a, const cv::Mat &b)
    {
        using namespace std;
        if (b.empty())
            pa.swap(pb);
        pa->update(a);
        if (!b.empty())
            pb->update(b);
        cv::Mat pose = cv::Mat::eye(2, 3, CV_64F);
        if (pb->empty())
            return pose;
        if (jacobi.empty()) {
            for (int i = 0; i < params.levels; ++i)
                jacobi.push_back(Jacobi::create(params.xx, params.xy, params.x, params.yx, params.yy, params.y, params.jakobiType));
        }
        auto i0 = pa->levels.rbegin(), i1 = pb->levels.rbegin();
        for(auto j = jacobi.rbegin(); j != jacobi.rend(); j++, i0++, i1++)
        {
            pose.at<double>(0, 2) *= 2;
            pose.at<double>(1, 2) *= 2;

            gn.solve(*i0, *i1, j->get(), pose);
        }
        return pose;
    }
};


Aligner::Ptr Aligner::create(const AlignerParams &params, int channels)
{
    using namespace std;
    switch(channels) {
    case 1:
        return make_unique<AlignerT<uint8_t>>(params);
    case 3:
        return make_unique<AlignerT<Eigen::Matrix<uint8_t, 3, 1>>>(params);
    case 4:
        return make_unique<AlignerT<Eigen::Matrix<uint8_t, 4, 1>>>(params);
    default:
        return nullptr;
    }
}
