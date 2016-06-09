
#define NO_IMPORT_ARRAY
#include "models/models.hh"
#include "detail.hpp"

namespace py {
    
    namespace ext {
        
        PyObject* check(PyTypeObject* type, PyObject* evaluee) {
            return py::boolean(type == Py_TYPE(evaluee));
        }
        
    }

}