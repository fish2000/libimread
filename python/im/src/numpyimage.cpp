
#define NO_IMPORT_ARRAY
#include "numpyimage.hh"

namespace py {
    
    namespace methods {
        
        PyObject* structcode_parse(PyObject* self, PyObject* args) {
            char const* code;
            
            if (!PyArg_ParseTuple(args, "s", &code)) {
                PyErr_SetString(PyExc_ValueError,
                    "cannot get structcode string");
                return NULL;
            }
            
            return Py_BuildValue("O",
                im::detail::structcode_to_dtype(code));
        }
        
    }
}