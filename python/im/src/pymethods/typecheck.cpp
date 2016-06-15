
#define NO_IMPORT_ARRAY
#include "pymethods/typecheck.hh"
// #include "detail.hpp"

namespace py {
    
    namespace ext {
        
        PyObject* typecheck(PyTypeObject* type, PyObject* evaluee) {
            // return py::boolean(type == Py_TYPE(evaluee));
            return Py_BuildValue("O", type == Py_TYPE(evaluee) ? Py_True : Py_False);
        }
        
    }
}