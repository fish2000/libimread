
#define NO_IMPORT_ARRAY
#include "hybridimage.hh"
#include "detail.hpp"

namespace py {
    
    namespace functions {
        
        PyObject* hybridimage_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return nullptr; }
            return py::boolean(HybridImage_Check(evaluee));
        }
        
    }
}