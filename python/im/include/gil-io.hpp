
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
                source(FILE* fh);
                source(FILE* fh, PyObject* pyfh);
        };
        
        class sink : public im::handle::sink {
            public:
                PyObject* object = nullptr;
                sink(FILE* fh);
                sink(FILE* fh, PyObject* pyfh);
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
                void operator()(std::add_pointer_t<F> source_sink) {
                    if (source_sink != nullptr) {
                        PyFile_DecUseCount((PyFileObject*)source_sink->object);
                        Py_DECREF(source_sink->object);
                        delete source_sink;
                    }
                }
            };
            
            with(PyObject* fileobject);
            with(PyFileObject* fileobject);
            with(char const* filepth);
            with(std::nullptr_t no = nullptr);
            virtual ~with();
            
            void init();
            void restore();
            
            using source_t = std::unique_ptr<py::handle::source,
                                   fileclose<py::handle::source>>;
            
            using sink_t = std::unique_ptr<py::handle::sink,
                                 fileclose<py::handle::sink>>;
            
            source_t source();
            sink_t sink();
        };
        
    } /* namespace gil */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_GIL_IO_HH_