
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_

#include <cstddef>
#include <memory>
#include <Python.h>
#include <libimread/libimread.hpp>
#include <libimread/filehandle.hh>

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
            
            std::unique_ptr<im::handle::source> source() const;
            std::unique_ptr<im::handle::sink> sink() const;
        };
        
    } /* namespace gil */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_