
#include "gil.hpp"

namespace py {
    
    namespace gil {
        
        release::release()
            :state(PyEval_SaveThread()), active(true)
            {}
        
        release::~release() {
            if (active) { restore(); }
        }
        
        void release::restore() {
            PyEval_RestoreThread(state);
            active = false;
        }
        
        ensure::ensure()
            :state(PyGILState_Ensure()), active(true)
            {}
        
        ensure::~ensure() {
            if (active) { restore(); }
        }
        
        void ensure::restore() {
            PyGILState_Release(state);
            active = false;
        }
        
    } /* namespace gil */
    
} /* namespace py */
