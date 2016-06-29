
#define NO_IMPORT_ARRAY
#include "pymethods/structcode_parse.hh"
#include "detail.hpp"

namespace py {
    
    namespace functions {
        
        PyObject* structcode_parse(PyObject* self, PyObject* args) {
            char const* code;
            if (!PyArg_ParseTuple(args, "s", &code)) { return nullptr; }
            return py::detail::structcode_to_dtype(code);
        }
        
    }
}