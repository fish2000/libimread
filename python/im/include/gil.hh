
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_GIL_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_GIL_HH_

#include <Python.h>

namespace py {
    
    namespace gil {
        
        struct release {
            PyThreadState*  state;
            bool            active;
            
            release()
                :state(PyEval_SaveThread()), active(true)
                {}
            
            ~release() {
                if (active) { restore(); }
            }
            
            void restore() {
                PyEval_RestoreThread(state);
                active = false;
            }
        };
        
        struct ensure {
            PyGILState_STATE    state;
            bool                active;
            
            ensure()
                :state(PyGILState_Ensure()), active(true)
                {}
            
            ~ensure() {
                if (active) { restore(); }
            }
            
            void restore() {
                PyGILState_Release(state);
                active = false;
            }
        };
        
    } /* namespace gil */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_GIL_HH_