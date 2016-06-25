
#include <memory>

#define NO_IMPORT_ARRAY
#include "pymethods/detect.hh"

#include "check.hh"
#include "gil.hpp"
#include "gil-io.hpp"
#include "detail.hpp"
#include "options.hpp"
#include "pybuffer.hpp"

#include <libimread/libimread.hpp>
#include <libimread/file.hh>
#include <libimread/imageformat.hh>
#include <libimread/formats.hh>

namespace py {
    
    namespace functions {
        
        using im::FileSource;
        using im::ImageFormat;
        
        namespace detail {
            
            std::string detect(char const* source) {
                std::unique_ptr<ImageFormat> format;
                std::unique_ptr<FileSource> input;
                
                try {
                    py::gil::release nogil;
                    input = std::make_unique<FileSource>(source);
                    format = im::for_source(input.get());
                    return format->get_suffix();
                } catch (im::FormatNotFound& exc) {
                    return "";
                }
            }
            
            std::string detect(Py_buffer const& view) {
                std::unique_ptr<ImageFormat> format;
                std::unique_ptr<py::buffer::source> input;
                
                try {
                    py::gil::release nogil;
                    input = std::make_unique<py::buffer::source>(view);
                    format = im::for_source(input.get());
                    return format->get_suffix();
                } catch (im::FormatNotFound& exc) {
                    return "";
                }
            }
            
            std::string detect(PyObject* file) {
                std::unique_ptr<ImageFormat> format;
                typename py::gil::with::source_t input;
                
                try {
                    py::gil::with iohandle(file);
                    input = iohandle.source();
                    format = im::for_source(input.get());
                    return format->get_suffix();
                } catch (im::FormatNotFound& exc) {
                    return "";
                }
            }
            
        }
        
        PyObject* detect(PyObject* _nothing_, PyObject* args, PyObject* kwargs) {
            PyObject* py_is_blob = nullptr;
            PyObject* file = nullptr;
            Py_buffer view;
            char const* keywords[] = { "source", "file", "is_blob", nullptr };
            bool is_blob = false;
            
            if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "|s*OO:__init__", const_cast<char**>(keywords),
                &view,                      /// "view", buffer with file path or image data
                &file,                      /// "file", possible file-like object
                &py_is_blob))               /// "is_blob", Python boolean specifying blobbiness
            {
                return nullptr;
            }
            
            /// test is necessary, the next line chokes on nullptr:
            is_blob = py::options::truth(py_is_blob);
            
            if (file) {
                /// file - check before passing:
                if (!PyFile_Check(file)) {
                    PyErr_SetString(PyExc_ValueError,
                        "File object isn't file-ish enough");
                    return nullptr;
                }
                return py::string(detail::detect(file));
            } else if (is_blob) {
                /// blob - pass buffer along:
                return py::string(detail::detect(view));
            } else {
                /// filename - extract from buffer:
                py::buffer::source source(view);
                return py::string(detail::detect(source.str().c_str()));
            }
            
        }
        
    }
}