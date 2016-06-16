
#define NO_IMPORT_ARRAY
#include "pymethods/typecheck.hh"

namespace py {
    
    namespace ext {
        
        PyObject* typecheck(PyTypeObject* type, PyObject* evaluee) {
            return Py_BuildValue("O", type == Py_TYPE(evaluee) ? Py_True : Py_False);
        }
        
        PyObject* subtypecheck(PyTypeObject* type, PyObject* evaluee) {
            return Py_BuildValue("O", PyObject_TypeCheck(evaluee, type) ? Py_True : Py_False);
        }
        
    }
}