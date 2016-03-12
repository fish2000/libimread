
#define NO_IMPORT_ARRAY
#include "halideimage.hh"

namespace py {
    
    namespace functions {
        
        PyObject* image_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return NULL; }
            return Py_BuildValue("O", ImageModel_Check(evaluee) ? Py_True : Py_False);
        }
        
        PyObject* buffer_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return NULL; }
            return Py_BuildValue("O", BufferModel_Check(evaluee) ? Py_True : Py_False);
        }
        
        PyObject* imagebuffer_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return NULL; }
            return Py_BuildValue("O", ImageBufferModel_Check(evaluee) ? Py_True : Py_False);
        }
        
    }
}