
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_

#include <cstddef>
#include <Python.h>

namespace py {
    
    namespace gil {
        
        struct with {
            PyThreadState*  state;
            PyFileObject*   source;
            FILE*           file;
            bool            active;
            
            with(PyObject* fileobject);
            with(PyFileObject* fileobject);
            with(std::nullptr_t no = nullptr);
            ~with();
            
            void init();
            void restore();
        };
        
    } /* namespace gil */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_