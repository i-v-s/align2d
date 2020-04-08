#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <memory>

#include "utils.h"


class Jacobi
{
protected:
    Jacobi(){};
    cv::Mat gx_, gy_; // Gradients
    double scale_;
public:
    inline const cv::Mat & gx() const { return gx_;}
    inline const cv::Mat & gy() const { return gy_;}
    inline double scale() const {return scale_;}
    virtual void set(const cv::Mat &f, double scale = 1.0) = 0;
    virtual void shrink(const Jacobi * p) = 0;
    virtual void solve(const cv::Mat & e, cv::Mat & dt) const = 0;
    virtual ~Jacobi(){};
    enum Type
    {
        jtShift, jtStereoX, jtAffine
    };
    typedef std::unique_ptr<Jacobi> Ptr;
    static Ptr create(Type jType, int cvType = CV_16S);
    static Ptr create(bool xx = false, bool xy = false, bool x = true,
                      bool yx = false, bool yy = false, bool y = true,
                      int cvType = CV_16S);
};



//typedef Jacobi<int16_t> JacobiShift;
//typedef Jacobi<int16_t, true, true, true, false, false, false> JacobiStereoX;
//typedef Jacobi<int16_t, true, true, true, true, true, true> JacobiAffine;

class GaussNewton
{
public:
    int maxStepCount_;
    int stepCount_;
    double maxError_;
    double thres_;
    GaussNewton(double maxError = 0.1, int maxStepCount = 20) :
        maxError_(maxError), maxStepCount_(maxStepCount), thres_(5) {}
    template<class T, bool out = false> double calcError(cv::Mat & me)
    {
        assert(me.type() == cv_type<T>());
        double e = 0;
        //T thres = T (std::numeric_limits<T>::max() * thres_);
        T thres = (T) thres_;
        for(T * p = (T *) me.data, * end = (T *) me.dataend; p < end; p++)
        {
            if(abs(*p) > thres) {e++; if(out) *p = 255;}
            else *p = 0;
        }
        return e / (me.cols * me.rows);
    }

    template<int writeOut = 0> double solve(const cv::Mat & f0, const cv::Mat & f1, const Jacobi * j, cv::Mat & pose, std::vector<cv::Mat> * out = 0)
    {
        cv::Mat du(2, 3, CV_64F);
        int h = f0.rows, w = f0.cols;
        cv::Mat whiteBox(f0.rows, f0.cols, CV_8U, cv::Scalar(100)), mask;
        double e = 1E10;
        cv::Mat p = pose.clone();
        double step = 1.0;

        for(int x = 0; x < maxStepCount_; x++)
        {
            cv::Mat f0t;
            cv::warpAffine(f0,        f0t, p, cv::Size(w, h), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(255));
            cv::warpAffine(whiteBox, mask, p, cv::Size(w, h), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar(0));

            cv::Mat em;
            cv::subtract(f1, f0t, em, mask, CV_16S);
            j->solve(em, du);
            double te = calcError<int16_t, true>(em);//cv::norm(em);

            if(writeOut & 2)
            {
                std::cout << "--- it " << x + 1 << std::endl;
                std::cout << "u = " << std::endl << p << std::endl << std::endl;
                std::cout << "e = " << te << std::endl;
                std::cout << "du = " << std::endl << du << std::endl << std::endl;
            }

            if((writeOut & 1) && out)
            {
                out->push_back(f0t.clone());
                out->push_back(em.clone());
            }

            //if(te < e)
            {
                p.copyTo(pose);
                if(te < maxError_)
                {
                    stepCount_ = x;
                    return te;
                }
                e = te;
                step = 1.0;
            }
            p += du * step;
        }
        stepCount_ = maxStepCount_;
        return e;
    }
};

#endif // FRAME_H
