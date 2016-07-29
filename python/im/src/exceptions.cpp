
#include "exceptions.hpp"

namespace py {
    
    namespace ex {
        
        PyObject* BaseException = PyExc_BaseException;
        PyObject* Exception = PyExc_Exception;
        
        #define REDEFINE_PYTHON_ERROR(name) \
            PyObject* name = PyExc_##name##Error;
        
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
        
        REDEFINE_PYTHON_ERROR(Type);
        REDEFINE_PYTHON_ERROR(Value);
        // REDEFINE_PYTHON_ERROR(Windows);
        REDEFINE_PYTHON_ERROR(ZeroDivision);
        
        #undef REDEFINE_PYTHON_ERROR
    }
    
}
