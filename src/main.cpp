#include <stdio.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <Eigen/Core>
#include "aligner.h"


typedef struct {
    PyObject_HEAD
    Aligner::Ptr aligner;
    AlignerParams params;
} ImageAlign2dObject;



static int ImageAlign2d_init(ImageAlign2dObject *self, PyObject *args, PyObject *kwds)
{
    static const char *kwlist[] = {"xx", "xy", "x", "yx", "yy", "y", "levels", "max_steps", NULL};
    auto &p = self->params;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ppppppii", const_cast<char**>(kwlist),
                                     &p.xx, &p.xy, &p.x, &p.yx, &p.yy, &p.y, &p.levels, &p.max_steps))
        return -1;
    return 0;
}


static void ImageAlign2d_dealloc(ImageAlign2dObject *self)
{
    self->aligner = nullptr;
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject* align_method(ImageAlign2dObject *self, PyObject *args)
{
    PyObject *arg1 = nullptr, *arg2 = nullptr;
    if (!PyArg_ParseTuple(args, "O|O", &arg1, &arg2))
        return nullptr;

    if (arg1->ob_type != &PyArray_Type || (arg2 && arg2->ob_type != &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Both align arguments must be numpy arrays");
        return nullptr;
    }

    PyArrayObject
            *arr1 = reinterpret_cast<PyArrayObject *>(arg1),
            *arr2 = (arg2 ? reinterpret_cast<PyArrayObject *>(arg2) : nullptr);

    int nd = PyArray_NDIM(arr1);
    if(arr2 && nd != PyArray_NDIM(arr2)) {
        PyErr_SetString(PyExc_ValueError, "Both align arguments must have same shapes");
        return nullptr;
    }

    if(nd < 2 || nd > 3) {
        PyErr_SetString(PyExc_ValueError, "Both align arguments must have 2 or 3 dimensions");
        return nullptr;
    }

    npy_intp *s1 = PyArray_SHAPE(arr1), *s2 = arr2 ? PyArray_SHAPE(arr2) : nullptr;
    int shape[3] = {0, 0, 1};
    for (int n = 0; n < nd; ++n) {
        int d = s1[n];
        if (s2 && d != s2[n]) {
            PyErr_SetString(PyExc_ValueError, "Both align arguments must have same shapes");
            return nullptr;
        }
        shape[n] = d;
    }

    auto type = PyArray_TYPE(arr1);
    if (type != NPY_UBYTE || (arr2 && PyArray_TYPE(arr2) != NPY_UBYTE)) {
        PyErr_SetString(PyExc_TypeError, "Both align arguments must have np.uint8 type");
        return nullptr;
    }
    auto [h, w, c] = shape;
    if (c > 4 || c < 1 || c == 2) {
        PyErr_SetString(PyExc_ValueError, "Wrong channels count");
        return nullptr;
    }

    if (!PyArray_IS_C_CONTIGUOUS(arr1) || !PyArray_ISALIGNED(arr1)) {
        auto o = PyArray_FromAny(reinterpret_cast<PyObject *>(arr1), nullptr, 0, 0, NPY_ARRAY_IN_ARRAY, nullptr);
        Py_DECREF(arr1);
        arr1 = reinterpret_cast<PyArrayObject *>(o);
    }

    if (arr2 && (!PyArray_IS_C_CONTIGUOUS(arr2) || !PyArray_ISALIGNED(arr2))) {
        auto o = PyArray_FromAny(reinterpret_cast<PyObject *>(arr2), nullptr, 0, 0, NPY_ARRAY_IN_ARRAY, nullptr);
        Py_DECREF(arr2);
        arr2 = reinterpret_cast<PyArrayObject *>(o);
    }

    static int types[] = {CV_8U, -1, CV_8UC3, CV_8UC4};
    type = types[c - 1];

    if (!self->aligner)
        self->aligner = Aligner::create(self->params, c);

    using namespace cv;
    Mat ma = Mat(h, w, type, PyArray_DATA(arr1));
    Mat mb = arr2 ? Mat(h, w, type, PyArray_DATA(arr2)) : Mat();
    cv::Mat cv_pose = self->aligner->align(ma, mb);

    const npy_intp s[] = {2, 3, 0};
    PyObject *pose = PyArray_EMPTY(2, s, NPY_DOUBLE, 0);
    memcpy(PyArray_DATA((PyArrayObject*)pose), cv_pose.data, 6 * sizeof (double));

    //Py_DECREF(arr1);
    //if (arr2)
    //    Py_DECREF(arr2);

    return pose;
}


static PyMethodDef ImageAlign2d_methods[] = {
    {"align", (PyCFunction) align_method, METH_VARARGS,
     "Return the name, combining the first and last name"
    },
    {NULL}  /* Sentinel */
};


static PyTypeObject ImageAlign2dType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "align2d.ImageAlign2d",
    sizeof(ImageAlign2dObject),
    0,
    (destructor)ImageAlign2d_dealloc
};


static PyModuleDef align2d_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "align2d",
    .m_doc = "A Python module that alignes two images.",
    .m_size = -1,
};


PyMODINIT_FUNC PyInit_align2d(void)
{
    import_array();
    ImageAlign2dType.tp_doc = "Custom objects";
    ImageAlign2dType.tp_flags = Py_TPFLAGS_DEFAULT;
    ImageAlign2dType.tp_new = PyType_GenericNew;
    ImageAlign2dType.tp_init = (initproc) ImageAlign2d_init;
    ImageAlign2dType.tp_methods = ImageAlign2d_methods;

    PyObject *m;
    if (PyType_Ready(&ImageAlign2dType) < 0)
        return NULL;

    m = PyModule_Create(&align2d_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ImageAlign2dType);
    if (PyModule_AddObject(m, "ImageAlign2d", (PyObject *) &ImageAlign2dType) < 0) {
        Py_DECREF(&ImageAlign2dType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
