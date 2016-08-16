
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_EXCEPTIONS_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_EXCEPTIONS_HPP_

#include <string>
#include <vector>
#include <type_traits>
#include <Python.h>

namespace py {
    
    /// forward-declare py::ref
    class ref;
    
    namespace ex {
        
        /// vector of exception pointer wrappers:
        using objectvec_t = std::vector<py::ref>;
        
        /// static-storage containment unit,
        /// for an exception pointer vector:
        struct idx {
            explicit idx(PyObject*);
            static objectvec_t& get();
        };
        
        /// This list is adapted unabashedly right from the docs:
        /// https://docs.python.org/2/c-api/exceptions.html#standard-exceptions
        
        extern PyObject* BaseException;
        extern PyObject* Exception;
        
        #define REDECLARE_PYTHON_ERROR(name)                                            \
            extern PyObject* name;
        
        REDECLARE_PYTHON_ERROR(Standard);
        REDECLARE_PYTHON_ERROR(Arithmetic);
        REDECLARE_PYTHON_ERROR(Lookup);
        REDECLARE_PYTHON_ERROR(Assertion);
        REDECLARE_PYTHON_ERROR(Attribute);
        // REDECLARE_PYTHON_ERROR(EOF);
        REDECLARE_PYTHON_ERROR(Environment);
        REDECLARE_PYTHON_ERROR(FloatingPoint);
        REDECLARE_PYTHON_ERROR(IO);
        REDECLARE_PYTHON_ERROR(Import);
        REDECLARE_PYTHON_ERROR(Index);
        REDECLARE_PYTHON_ERROR(Key);
        
        extern PyObject* KeyboardInterrupt;
        
        REDECLARE_PYTHON_ERROR(Memory);
        REDECLARE_PYTHON_ERROR(Name);
        REDECLARE_PYTHON_ERROR(NotImplemented);
        REDECLARE_PYTHON_ERROR(OS);
        REDECLARE_PYTHON_ERROR(Overflow);
        REDECLARE_PYTHON_ERROR(Reference);
        REDECLARE_PYTHON_ERROR(Runtime);
        REDECLARE_PYTHON_ERROR(Syntax);
        REDECLARE_PYTHON_ERROR(System);
        
        extern PyObject* SystemExit;
        
        REDECLARE_PYTHON_ERROR(Type);
        REDECLARE_PYTHON_ERROR(Value);
        // REDECLARE_PYTHON_ERROR(Windows);
        REDECLARE_PYTHON_ERROR(ZeroDivision);
        
        #undef REDECLARE_PYTHON_ERROR
    }
    
    /// use py::error() like so:
    ///
    /// if (something_bad_happened) {
    ///     return py::error("Something bad happened!"); /// defaults to RuntimeError
    ///     return py::error("Something bad happened!", py::ex::IO);
    /// }
    /// 
    /// ... if within a function with an odd return type, like an __init__ implementation:
    /// if (something_bad_happened) {
    ///     return py::error("Something bad happened!", py::ex::IO, 0);
    /// }
    using pyptr_t = std::add_pointer_t<PyObject>;
    
    template <typename rT = pyptr_t>
    rT error(std::string const& message,
             PyObject* exception = py::ex::Runtime,
             rT out = (pyptr_t)nullptr) {
        if (!exception) { exception = py::ex::Runtime; }
        PyErr_SetString(exception, message.c_str());
        return out;
    }
    
    #define DECLARE_ERROR_SHORTCUT(errorname)                                               \
        template <typename rT = pyptr_t>                                                    \
        rT errorname##Error(std::string const& message, rT out = (pyptr_t)nullptr) {        \
            PyErr_SetString(py::ex::errorname, message.c_str());                            \
            return out;                                                                     \
        }
    
    /// use error shortcuts like so:
    ///
    /// if (something_bad_specifically_happened) {
    ///     return py::ValueError("Something bad happened with a value!");
    /// } else if (something_else_specifically_happened) {
    ///     return py::AttributeError("Something bad happened with an attribute!");
    /// }
    
    DECLARE_ERROR_SHORTCUT(Lookup);
    DECLARE_ERROR_SHORTCUT(Assertion);
    DECLARE_ERROR_SHORTCUT(Attribute);
    // DECLARE_ERROR_SHORTCUT(EOF);
    DECLARE_ERROR_SHORTCUT(Environment);
    DECLARE_ERROR_SHORTCUT(FloatingPoint);
    DECLARE_ERROR_SHORTCUT(IO);
    DECLARE_ERROR_SHORTCUT(Import);
    DECLARE_ERROR_SHORTCUT(Index);
    DECLARE_ERROR_SHORTCUT(Key);
    DECLARE_ERROR_SHORTCUT(Memory);
    DECLARE_ERROR_SHORTCUT(Name);
    DECLARE_ERROR_SHORTCUT(NotImplemented);
    DECLARE_ERROR_SHORTCUT(OS);
    DECLARE_ERROR_SHORTCUT(Overflow);
    DECLARE_ERROR_SHORTCUT(Reference);
    DECLARE_ERROR_SHORTCUT(Runtime);
    DECLARE_ERROR_SHORTCUT(System);
    DECLARE_ERROR_SHORTCUT(Type);
    DECLARE_ERROR_SHORTCUT(Value);
    DECLARE_ERROR_SHORTCUT(ZeroDivision);
    
    #undef DECLARE_ERROR_SHORTCUT
    
    /// py::ErrorOccurred() just wraps PyErr_Occurred()
    /// such that it returns reliable and sensible booleans:
    bool ErrorOccurred(void);
    std::string ErrorName(void);
    std::string ErrorMessage(void);
    
    /// clear the last error:
    void ClearError(void);
    
    /// py::LastError() returns the last error that occurred,
    /// as a borrowed-reference PyObject*, without all the fuss/muss
    /// of the PyErr_{Fetch,NormalizeException,Restore}() API
    PyObject* LastError(void);
    
    /// Convenience templated-terniary-operator function,
    /// for juggling two typed values that
    template <typename CleanType = pyptr_t,
              typename ErrorType = pyptr_t,
              typename FinalType = typename std::common_type<CleanType, ErrorType>::type>
    FinalType for_error_state(CleanType clean_value,
                              ErrorType error_value = (pyptr_t)nullptr) {
        return py::ErrorOccurred() ? clean_value : error_value;
    }
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_EXCEPTIONS_HPP_