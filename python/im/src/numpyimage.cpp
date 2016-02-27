
#define NO_IMPORT_ARRAY
#include "numpyimage.hh"

namespace py {
    
    namespace functions {
        
        PyObject* structcode_parse(PyObject* self, PyObject* args) {
            char const* code;
            if (!PyArg_ParseTuple(args, "s", &code)) { return NULL; }
            return Py_BuildValue("O",
                im::detail::structcode_to_dtype(code));
        }
        
        PyObject* numpyimage_check(PyObject* self, PyObject* args) {
            PyObject* evaluee;
            if (!PyArg_ParseTuple(args, "O", &evaluee)) { return NULL; }
            return Py_BuildValue("O", NumpyImage_Check(evaluee) ? Py_True : Py_False);
        }
        
    }
}