
#include "gil-io.hpp"

namespace py {
    
    namespace handle {
        
        source::source(FILE* fh)
            :im::handle::source(fh)
            {}
        
        /// handle + object ctor: requires unreleased GIL
        source::source(FILE* fh, PyObject* pyfh)
            :im::handle::source(fh)
            ,object(pyfh)
            {
                Py_INCREF(object);
                PyFile_IncUseCount((PyFileObject*)object);
            }
        
        sink::sink(FILE* fh)
            :im::handle::sink(fh)
            {}
        
        /// handle + object ctor: requires unreleased GIL
        sink::sink(FILE* fh, PyObject* pyfh)
            :im::handle::sink(fh)
            ,object(pyfh)
            {
                Py_INCREF(object);
                PyFile_IncUseCount((PyFileObject*)object);
            }
    
    }
    
    namespace gil {
        
        with::with(PyFileObject* fileobject)
            :state(nullptr), object(fileobject)
            ,file(nullptr),  active(PyFile_Check(fileobject))
            {
                if (active) { init(); }
            }
        
        with::with(PyObject* fileobject)
            :with(reinterpret_cast<PyFileObject*>(fileobject))
            {}
        
        with::with(char const* filepth)
            :with(PyFile_FromString(const_cast<char*>(filepth),
                                    const_cast<char*>("r+")))
            {}
        
        with::with(std::nullptr_t no)
            :state(nullptr), object(nullptr)
            ,file(nullptr),  active(false)
            {}
        
        with::~with() {
            if (active) { restore(); }
        }
        
        void with::init() {
            Py_INCREF(object);                                                  /// own the object;
            file = PyFile_AsFile(reinterpret_cast<PyObject*>(object));          /// get the file;
            PyFile_IncUseCount(object);                                         /// protect the filehandle;
            state = PyEval_SaveThread();                                        /// release GIL;
        }
        
        void with::restore() {
            PyEval_RestoreThread(state);                                        /// acquire GIL;
            PyFile_DecUseCount(object);                                         /// unprotect the filehandle;
            Py_DECREF(object);                                                  /// disown the object;
            file = nullptr;                                                     /// clear members;
            active = false;
        }
        
        with::source_t with::source() {
            PyEval_RestoreThread(state);                                        /// acquire the GIL;
            auto out = with::source_t(new py::handle::source(file,              /// wrap object + handle in handle::source,
                                      reinterpret_cast<PyObject*>(object)));    /// incrementing refcount and use count;
            state = PyEval_SaveThread();                                        /// release the GIL;
            return out;
        }
        
        with::sink_t with::sink() {
            PyEval_RestoreThread(state);                                        /// acquire the GIL;
            auto out = with::sink_t(new py::handle::sink(file,                  /// wrap object + handle in handle::sink,
                                    reinterpret_cast<PyObject*>(object)));      /// incrementing refcount and use count;
            state = PyEval_SaveThread();                                        /// release the GIL;
            return out;
        }
    
    } /* namespace gil */

} /* namespace py */
