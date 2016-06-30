
#define NO_IMPORT_ARRAY
#include "pymethods/structcode_parse.hh"
#include "detail.hpp"
#include "structcode.hpp"

namespace py {
    
    namespace functions {
        
        PyObject* structcode_parse(PyObject* self, PyObject* args) {
            char const* code;
            if (!PyArg_ParseTuple(args, "s", &code)) { return nullptr; }
            return py::detail::structcode_to_dtype(code);
        }
        
        PyObject* structcode_convert(PyObject* self, PyObject* args) {
            char const* code;
            if (!PyArg_ParseTuple(args, "s", &code)) { return nullptr; }
            auto tuple = structcode::parse(code);
            return py::convert(tuple);
        }
        
    }
}