
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_

#include <Python.h>

namespace py {
    
    namespace gil {
        
        struct with {
            PyThreadState*  state;
            PyFileObject*   source;
            FILE*           file;
            bool            active;
            
            with(PyObject* fileobject)
                :state(nullptr), source(reinterpret_cast<PyFileObject*>(fileobject))
                ,file(nullptr),  active(PyFile_Check(fileobject))
                {
                    if (active) { init(); }
                }
            
            with(PyFileObject* fileobject)
                :state(nullptr), source(fileobject)
                ,file(nullptr),  active(PyFile_Check(fileobject))
                {
                    if (active) { init(); }
                }
            
            with(std::nullptr_t no = nullptr)
                :state(nullptr), source(nullptr)
                ,file(nullptr),  active(false)
                {}
            
            ~with() {
                if (active) { restore(); }
            }
            
            void init() {
                Py_INCREF(source);
                file = PyFile_AsFile(reinterpret_cast<PyObject*>(source));
                PyFile_IncUseCount(source);
                state = PyEval_SaveThread();
            }
            
            void restore() {
                PyEval_RestoreThread(state);
                PyFile_DecUseCount(source);
                Py_DECREF(source);
                file = nullptr;
                active = false;
            }
        };
        
    } /* namespace gil */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_