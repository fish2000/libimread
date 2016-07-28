
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
        
        /// rock the GIL / or else I'm gonna
        ///         stab a knife through your eye 
        ///    -- S. Combs (licensed and paraphrased):
        ///          DIDDYS KIDS REMEMBER TO GIL UP ON THIS FUNC,
        ///          SAFE INTERPRETER REENTRANCY IS NO ACCIDENT !!!
        ///          REMEMBER TO TELL A GROWNUP IF YOUR GIL CODE HURTS YOU
        ///          REPORT UNCAUGHT EXCEPTIONS THROWN BY STRANGERS
        
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
