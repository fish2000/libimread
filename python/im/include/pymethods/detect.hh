
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_DETECT_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_DETECT_HH_

#include <string>
#include <Python.h>

namespace py {
    
    namespace functions {
        
        namespace detail {
            
            std::string detect(char const*);
            std::string detect(Py_buffer const&);
            std::string detect(PyObject*);
            
        }
        
        PyObject* detect(PyObject*, PyObject*, PyObject*);
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_DETECT_HH_