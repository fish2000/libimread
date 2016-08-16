
#include <algorithm>
#include "exceptions.hpp"
#include "detail.hpp"

namespace py {
    
    namespace ex {
        
        idx::idx(PyObject* error) {
            get().emplace_back(error);
            get().back().inc();
        }
        
        objectvec_t& idx::get() {
            static objectvec_t internal;
            return internal;
        }
        
        
#define REDEFINE_PYTHON_ERROR(errorname)                                \
        PyObject* errorname = PyExc_##errorname##Error;                 \
        idx initialize_##errorname(errorname);
        
#define DEFINE_PYTHON_ERROR(errorname)                                  \
        idx initialize_##errorname(errorname);
        
        PyObject* BaseException = PyExc_BaseException;
        PyObject* Exception = PyExc_Exception;
        
        REDEFINE_PYTHON_ERROR(Standard);
        REDEFINE_PYTHON_ERROR(Arithmetic);
        REDEFINE_PYTHON_ERROR(Lookup);
        REDEFINE_PYTHON_ERROR(Assertion);
        REDEFINE_PYTHON_ERROR(Attribute);
        // REDEFINE_PYTHON_ERROR(EOF);
        REDEFINE_PYTHON_ERROR(Environment);
        REDEFINE_PYTHON_ERROR(FloatingPoint);
        REDEFINE_PYTHON_ERROR(IO);
        REDEFINE_PYTHON_ERROR(Import);
        REDEFINE_PYTHON_ERROR(Index);
        REDEFINE_PYTHON_ERROR(Key);
        
        PyObject* KeyboardInterrupt = PyExc_KeyboardInterrupt;
        DEFINE_PYTHON_ERROR(KeyboardInterrupt);
        
        REDEFINE_PYTHON_ERROR(Memory);
        REDEFINE_PYTHON_ERROR(Name);
        REDEFINE_PYTHON_ERROR(NotImplemented);
        REDEFINE_PYTHON_ERROR(OS);
        REDEFINE_PYTHON_ERROR(Overflow);
        REDEFINE_PYTHON_ERROR(Reference);
        REDEFINE_PYTHON_ERROR(Runtime);
        REDEFINE_PYTHON_ERROR(Syntax);
        REDEFINE_PYTHON_ERROR(System);
        
        PyObject* SystemExit = PyExc_SystemExit;
        DEFINE_PYTHON_ERROR(SystemExit);
        
        REDEFINE_PYTHON_ERROR(Type);
        REDEFINE_PYTHON_ERROR(Value);
        // REDEFINE_PYTHON_ERROR(Windows);
        REDEFINE_PYTHON_ERROR(ZeroDivision);
        
        /// stow these last:
        DEFINE_PYTHON_ERROR(BaseException);
        DEFINE_PYTHON_ERROR(Exception);
        
    }
    
    bool ErrorOccurred(void) {
        return !!PyErr_Occurred();
    }
    
    /// ... in retrospect, I may have named too many methods `get()`:
    PyObject* LastError(void) {
        if (!py::ErrorOccurred()) { return nullptr; }
        auto const& iter = std::find_if(ex::idx::get().begin(),
                                        ex::idx::get().end(),
                                     [](py::ref& exc) -> bool {
            return PyErr_ExceptionMatches(exc.get());
        });
        return iter != std::end(ex::idx::get()) ? iter->get() : nullptr;
    }
    
    
}

#undef REDEFINE_PYTHON_ERROR
