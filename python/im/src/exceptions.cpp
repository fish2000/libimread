
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
        
        objectvec_t::iterator       idx::begin()       { return std::begin(get());  }
        objectvec_t::const_iterator idx::begin() const { return std::cbegin(get()); }
        objectvec_t::iterator       idx::end()         { return std::end(get());    }
        objectvec_t::const_iterator idx::end() const   { return std::cend(get());   }
        
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
    
    std::string ErrorName(void) {
        if (!py::ErrorOccurred()) { return "<undefined>"; }
        py::ref name = PyObject_GetAttrString(PyErr_Occurred(), "__name__");
        return name.to_string();
    }
    
    std::string ErrorMessage(void) {
        if (!py::ErrorOccurred()) { return "<undefined>"; }
        PyObject* type;
        PyObject* value;
        PyObject* traceback;
        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);
        py::ref message = PyObject_GetAttrString(value, "message");
        std::string out = message.to_string();
        PyErr_Restore(type, value, traceback);
        return out;
    }
    
    void ClearError(void) {
        PyErr_Clear();
    }
    
    /// ... in retrospect, I may have named too many methods `get()`:
    PyObject* LastError(void) {
        using namespace ex;
        if (!py::ErrorOccurred()) { return nullptr; }
        auto const& iter = std::find_if(idx::begin(),
                                        idx::end(),
                                     [](py::ref& exc) -> bool {
            return PyErr_ExceptionMatches(exc.get());
        });
        return iter != idx::end() ? iter->get() : nullptr;
    }
    
    
}

#undef REDEFINE_PYTHON_ERROR
