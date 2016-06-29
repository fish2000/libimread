
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_PYMETHODS_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_PYMETHODS_HH_

#include "detect.hh"
#include "structcode_parse.hh"
#include "typecheck.hh"
#include "buffermethods.hh"
#include "imagemethods.hh"

namespace py {
    
    namespace functions {
        
        PyObject* image_check(PyObject* self, PyObject* args);
        PyObject* buffer_check(PyObject* self, PyObject* args);
        PyObject* imagebuffer_check(PyObject* self, PyObject* args);
        PyObject* array_check(PyObject* self, PyObject* args);
        PyObject* arraybuffer_check(PyObject* self, PyObject* args);
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_PYMETHODS_HH_