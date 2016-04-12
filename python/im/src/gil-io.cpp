
#include "gil-io.hpp"
#include <libimread/errors.hh>

namespace py {
    
    namespace gil {
        
        with::with(PyObject* fileobject)
            :state(nullptr), object(reinterpret_cast<PyFileObject*>(fileobject))
            ,file(nullptr),  active(PyFile_Check(fileobject))
            {
                if (active) { init(); }
            }
        
        with::with(PyFileObject* fileobject)
            :state(nullptr), object(fileobject)
            ,file(nullptr),  active(PyFile_Check(fileobject))
            {
                if (active) { init(); }
            }
        
        with::with(std::nullptr_t no)
            :state(nullptr), object(nullptr)
            ,file(nullptr),  active(false)
            {}
        
        with::~with() {
            if (active) { restore(); }
        }
        
        void with::init() {
            Py_INCREF(object);
            file = PyFile_AsFile(reinterpret_cast<PyObject*>(object));
            PyFile_IncUseCount(object);
            state = PyEval_SaveThread();
        }
        
        void with::restore() {
            PyEval_RestoreThread(state);
            PyFile_DecUseCount(object);
            Py_DECREF(object);
            file = nullptr;
            active = false;
        }
        
        with::source_t with::source() const {
            if (!active) {
                imread_raise(CannotReadError,
                    "py::gil::with::source():",
                    "\tGIL guard not active");
            }
            return with::source_t(new py::handle::source(file,
                                  reinterpret_cast<PyObject*>(object)));
        }
        
        with::sink_t with::sink() const {
            if (!active) {
                imread_raise(CannotWriteError,
                    "py::gil::with::sink():",
                    "\tGIL guard not active");
            }
            return with::sink_t(new py::handle::sink(file,
                                reinterpret_cast<PyObject*>(object)));
        }
        
        
    } /* namespace gil */
    
} /* namespace py */
