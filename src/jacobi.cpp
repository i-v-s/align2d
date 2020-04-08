#include "jacobi.h"
#include <iostream>
#include <random>


template<class T, bool usegxx = false, bool usegxy = false, bool usegx = true,
                  bool usegyx = false, bool usegyy = false, bool usegy = true>
class JacobiT: public Jacobi
{
public:
	static constexpr int size() { return (usegx?1:0) + (usegxx?1:0) + (usegxy?1:0) + (usegy?1:0) + (usegyx?1:0) + (usegyy?1:0);}
    typedef Eigen::Matrix<double, (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0), (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0)> Matrix;
    typedef Eigen::Matrix<double, (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0), 1> Vector;
private:
    int kernel_;
    Eigen::Matrix<double, (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0), (usegx?1:0)+(usegxx?1:0)+(usegxy?1:0)+(usegy?1:0)+(usegyx?1:0)+(usegyy?1:0)> invA_;
    void calcA()
    {
        Matrix a = Matrix::Zero();
        const T * p1, * p2;
        double gx, gy;
        Vector v;
        if(usegx) p1 = (const T *) gx_.data;
        if(usegy) p2 = (const T *) gy_.data;
        int w = usegx ? gx_.cols : gy_.cols;
        int h = usegx ? gx_.rows : gy_.rows;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
            {
                double * r = v.data();
                if(usegx) gx = *(p1++);
                if(usegy) gy = *(p2++);
                if(usegxx) *(r++) = gx * x;
                if(usegxy) *(r++) = gx * y;
                if(usegx) *(r++) = gx;

                if(usegyx) *(r++) = gy * x;
                if(usegyy) *(r++) = gy * y;
                if(usegy) *(r++) = gy;
                assert(r - v.data() == size());
                a += v * v.transpose();
            }
        a *= scale_ * scale_;
        invA_ = a.inverse();
    }
    template<class Te> void calcB(Vector & b, const cv::Mat & e) const
    {
        assert(!usegx || (e.rows == gx_.rows && e.cols == gx_.cols));
        assert(!usegy || (e.rows == gy_.rows && e.cols == gy_.cols));
        //std::cout << "cv_type<Te>() = " << cv_type<Te>() << "; e.type() = " << e.type() << std::endl;
        assert(cv_type<Te>() == e.type());
        b.setZero();
        const T * p1, * p2;
        double gx, gy;
        Vector v;
        if(usegx) p1 = (const T *) gx_.data;
        if(usegy) p2 = (const T *) gy_.data;
        const Te * pe = (const Te *) e.data;
        int w = e.cols;
        int h = e.rows;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
            {
                double * r = v.data();
                if(usegx) gx = *(p1++);
                if(usegy) gy = *(p2++);
                if(usegxx) *(r++) = gx * x;
                if(usegxy) *(r++) = gx * y;
                if(usegx) *(r++) = gx;

                if(usegyx) *(r++) = gy * x;
                if(usegyy) *(r++) = gy * y;
                if(usegy) *(r++) = gy;
                assert(r - v.data() == size());
                b += v * *(pe++);
            }
        b *= scale_;
    }

public:
    JacobiT(int kernel = cv::FILTER_SCHARR): kernel_(kernel) {}
    void set(const cv::Mat &f, double scale = 1.0)
    {
        switch(kernel_)
        {
        case cv::FILTER_SCHARR:
            scale_ = scale / 32;
            break;
        case 1:
            scale_ = scale / 2;
            break;
        case 3:
            scale_ = scale / 8;
            break;
        case 5:
            scale_ = scale / 58;
            break;
        //case 7: scale /=
        }
        cv::Mat gx, gy;
        if(usegx)
        {
            if(f.type() == CV_8U && cv_type<T>() == CV_32S)
            {
                cv::Sobel(f, gx, CV_16S, 1, 0, kernel_);
                gx.convertTo(gx_, cv_type<T>());
            }
            else
                cv::Sobel(f, gx_, cv_type<T>(), 1, 0, kernel_);
        }
        if(usegy)
        {
            if(f.type() == CV_8U && cv_type<T>() == CV_32S)
            {
                cv::Sobel(f, gy, CV_16S, 0, 1, kernel_);
                gy.convertTo(gy_, cv_type<T>());
            }
            else
                cv::Sobel(f, gy_, cv_type<T>(), 0, 1, kernel_);
        }
        calcA();
    }
    void shrink(const Jacobi * p)
    {
        if(usegx)
        {
            if(gx_.empty()) gx_ = cv::Mat(p->gx().rows / 2, p->gx().cols / 2, cv_type<T>());
            ::shrink<T>(p->gx(), gx_);
        }
        if(usegy)
        {
            if(gy_.empty()) gy_ = cv::Mat(p->gy().rows / 2, p->gy().cols / 2, cv_type<T>());
            ::shrink<T>(p->gy(), gy_);
        }
        scale_ = p->scale() * 2;
        calcA();
    }
    inline void solve(const cv::Mat & e, Vector & r) const
    {
        Vector b;
        switch(e.type())
        {
        case CV_16S:
            calcB<int16_t>(b, e);
            break;
        case CV_16U:
            calcB<uint16_t>(b, e);
            break;
        case CV_32S:
            calcB<int32_t>(b, e);
            break;
        default:
            assert(!"Unknown error type");
        }
        r = (invA_ * b);
    }
    void solve(const cv::Mat & e, cv::Mat & dt) const
    {
        assert(dt.rows == 2 && dt.cols == 3 && dt.type() == CV_64F);
        Vector v;
        solve(e, v);
        const double * pv = v.data();
        double * pd = (double *) dt.data;
        *(pd++) = usegxx ? *(pv++) : 0;
        *(pd++) = usegxy ? *(pv++) : 0;
        *(pd++) = usegx ? *(pv++) : 0;
        *(pd++) = usegyx ? *(pv++) : 0;
        *(pd++) = usegyy ? *(pv++) : 0;
        *(pd++) = usegy ? *(pv++) : 0;
    }
};

template<class T> Jacobi::Ptr createT(Jacobi::Type jType)
{
    switch(jType)
    {
    case Jacobi::jtShift:
        return Jacobi::Ptr(new JacobiT<T, false, false, true, false, false, true>());
    case Jacobi::jtAffine:
        return Jacobi::Ptr(new JacobiT<T, true, true, true, true, true, true>());
    case Jacobi::jtStereoX:
        return Jacobi::Ptr(new JacobiT<T, true, true, true, false, false, false>());
    default:
        assert(!"Unknown type");
		return nullptr;
    }
}

Jacobi::Ptr Jacobi::create(Jacobi::Type jType, int cvType)
{
    switch(cvType)
    {
    case CV_16S:
        return createT<int16_t>(jType);
    case CV_32S:
        return createT<int32_t>(jType);
    default:
        assert(!"Unknown type");
		return nullptr;
    }
}


template<class T, bool xx, bool xy, bool x> inline Jacobi::Ptr createT(bool yx, bool yy, bool y)
{
    if (yx) {
        if (yy) {
            if (y)
                return std::make_unique<JacobiT<T, xx, xy, x, true, true, true>>();
            else
                return std::make_unique<JacobiT<T, xx, xy, x, true, true, false>>();
        } else {
            if (y)
                return std::make_unique<JacobiT<T, xx, xy, x, true, false, true>>();
            else
                return std::make_unique<JacobiT<T, xx, xy, x, true, false, false>>();
        }
    } else {
        if (yy) {
            if (y)
                return std::make_unique<JacobiT<T, xx, xy, x, false, true, true>>();
            else
                return std::make_unique<JacobiT<T, xx, xy, x, false, true, false>>();
        } else {
            if (y)
                return std::make_unique<JacobiT<T, xx, xy, x, false, false, true>>();
            else {
                if constexpr (xx || xy || x)
                    return std::make_unique<JacobiT<T, xx, xy, x, false, false, false>>();
                else
                    return nullptr;
            }
        }
    }
}


template<class T> inline Jacobi::Ptr createT(bool xx, bool xy, bool x, bool yx, bool yy, bool y)
{
    if (xx) {
        if (xy) {
            if (x)
                return createT<T, true, true, true>(yx, yy, y);
            else
                return createT<T, true, true, false>(yx, yy, y);
        } else {
            if (x)
                return createT<T, true, false, true>(yx, yy, y);
            else
                return createT<T, true, false, false>(yx, yy, y);

        }
    } else {
        if (xy) {
            if (x)
                return createT<T, false, true, true>(yx, yy, y);
            else
                return createT<T, false, true, false>(yx, yy, y);
        } else {
            if (x)
                return createT<T, false, false, true>(yx, yy, y);
            else
                return createT<T, false, false, false>(yx, yy, y);
        }
    }
}


Jacobi::Ptr Jacobi::create(bool xx, bool xy, bool x, bool yx, bool yy, bool y, int cvType)
{
    switch(cvType)
    {
    case CV_16S:
        return createT<int16_t>(xx, xy, x, yx, yy, y);
    case CV_32S:
        return createT<int32_t>(xx, xy, x, yx, yy, y);
    default:
        assert(!"Unknown type");
        return nullptr;
    }
}
