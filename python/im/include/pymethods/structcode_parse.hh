
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_STRUCTCODE_PARSE_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_STRUCTCODE_PARSE_HH_

#include <Python.h>

namespace py {
    
    namespace functions {
        
        PyObject* structcode_parse(PyObject*, PyObject*);
        PyObject* structcode_convert(PyObject*, PyObject*);
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_STRUCTCODE_PARSE_HH_