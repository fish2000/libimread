
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_TYPECHECK_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_TYPECHECK_HH_

#include <Python.h>

namespace py {
    
    namespace ext {
        
        /// typecheck() has a forward declaration!
        PyObject* typecheck(PyTypeObject* type, PyObject* evaluee);
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_TYPECHECK_HH_