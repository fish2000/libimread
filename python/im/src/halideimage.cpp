
#define NO_IMPORT_ARRAY
#include "halideimage.hh"
#include "detail.hpp"

namespace py {
    
    namespace ext {
        
        PyObject* check(PyTypeObject* type, PyObject* evaluee) {
            return py::boolean(type == Py_TYPE(evaluee));
        }
        
    }
    
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
        
        PyObject* array_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return NULL; }
            return py::boolean(ArrayModel_Check(evaluee));
        }
        
        PyObject* arraybuffer_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return NULL; }
            return py::boolean(ArrayBufferModel_Check(evaluee));
        }
        
    }
}