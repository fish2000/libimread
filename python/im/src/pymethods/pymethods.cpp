
#define NO_IMPORT_ARRAY
#include "pymethods/typecheck.hh"
#include "pymethods/pymethods.hh"
#include "detail.hpp"

namespace py {
    
    namespace functions {
        
        PyObject* image_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return nullptr; }
            return py::boolean(ImageModel_Check(evaluee));
        }
        
        PyObject* buffer_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return nullptr; }
            return py::boolean(BufferModel_Check(evaluee));
        }
        
        PyObject* imagebuffer_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return nullptr; }
            return py::boolean(ImageBufferModel_Check(evaluee));
        }
        
        PyObject* array_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return nullptr; }
            return py::boolean(ArrayModel_Check(evaluee));
        }
        
        PyObject* arraybuffer_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return nullptr; }
            return py::boolean(ArrayBufferModel_Check(evaluee));
        }
        
    }
}