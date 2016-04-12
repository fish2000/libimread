
#define NO_IMPORT_ARRAY
#include "halideimage.hh"
#include "detail.hpp"

namespace py {
    
    namespace functions {
        
        PyObject* image_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return NULL; }
            return py::boolean(ImageModel_Check(evaluee));
        }
        
        PyObject* buffer_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return NULL; }
            return py::boolean(BufferModel_Check(evaluee));
        }
        
        PyObject* imagebuffer_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return NULL; }
            return py::boolean(ImageBufferModel_Check(evaluee));
        }
        
    }
}