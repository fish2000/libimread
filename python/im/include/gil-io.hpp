
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_

#include <cstddef>
#include <memory>
#include <Python.h>
#include <libimread/libimread.hpp>
#include <libimread/filehandle.hh>

namespace py {
    
    namespace handle {
        
        class source : public im::handle::source {
            public:
                PyObject* object = nullptr;
            
            public:
                source(FILE* fh)
                    :im::handle::source(fh)
                    {}
                source(FILE* fh, PyObject* pyfh)
                    :im::handle::source(fh)
                    ,object(pyfh)
                    {
                        Py_INCREF(object);
                        PyFile_IncUseCount((PyFileObject*)object);
                    }
        };
        
        class sink : public im::handle::sink {
            public:
                PyObject* object = nullptr;
            
            public:
                sink(FILE* fh)
                    :im::handle::sink(fh)
                    {}
                sink(FILE* fh, PyObject* pyfh)
                    :im::handle::sink(fh)
                    ,object(pyfh)
                    {
                        Py_INCREF(object);
                        PyFile_IncUseCount((PyFileObject*)object);
                    }
        };
        
    }
    
    namespace gil {
        
        struct with {
            PyThreadState*  state;
            PyFileObject*   object;
            FILE*           file;
            bool            active;
            
            template <typename F>
            struct fileclose {
                constexpr fileclose() noexcept = default;
                template <typename U> fileclose(fileclose<U> const&) noexcept {};
                void operator()(std::add_pointer_t<F> handle) {
                    PyFile_DecUseCount((PyFileObject*)handle->object);
                    Py_DECREF(handle->object);
                    delete handle;
                }
            };
            
            with(PyObject* fileobject);
            with(PyFileObject* fileobject);
            with(std::nullptr_t no = nullptr);
            ~with();
            
            void init();
            void restore();
            
            using source_t = std::unique_ptr<py::handle::source, fileclose<py::handle::source>>;
            using sink_t = std::unique_ptr<py::handle::sink, fileclose<py::handle::sink>>;
            
            source_t source() const;
            sink_t sink() const;
        };
        
    } /* namespace gil */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_