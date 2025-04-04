/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_BUFFER_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_BUFFER_HPP_

#include <type_traits>
#include <Python.h>
#include "private/buffer_t.h"
#include "detail.hpp"

/// forward-declare PyArrayObject
typedef struct tagPyArrayObject PyArrayObject;

namespace im {
    
    namespace buffer {
        
        buffer_t* heapcopy(buffer_t const* buffer);
        buffer_t* heapcopy(buffer_t const* buffer, float scale); /// PLANAR ONLY!!
        buffer_t* heapcopy(Py_buffer const* pybuffer);
        buffer_t* heapcopy(PyArrayObject* array);
        void heapdestroy(buffer_t* buffer);
        
        template <typename BufferType>
        struct deleter_t {
            constexpr deleter_t() noexcept = default;
            template <typename U> deleter_t(deleter_t<U> const&) noexcept {}
            void operator()(std::add_pointer_t<BufferType> ptr) {
                im::buffer::heapdestroy(ptr);
            }
        };
        
        template <typename BufferType> inline
        Py_ssize_t ndims(BufferType const& buffer) {
            Py_ssize_t out = 0;
            out += buffer.extent[0] ? 1 : 0;
            out += buffer.extent[1] ? 1 : 0;
            out += buffer.extent[2] ? 1 : 0;
            out += buffer.extent[3] ? 1 : 0;
            return out;
        }
        
        template <typename BufferType> inline
        std::size_t length(BufferType const& buffer) {
            std::size_t out = static_cast<std::size_t>(buffer.elem_size);
            out *= buffer.extent[0] ? buffer.extent[0] : 1;
            out *= buffer.extent[1] ? buffer.extent[1] : 1;
            out *= buffer.extent[2] ? buffer.extent[2] : 1;
            out *= buffer.extent[3] ? buffer.extent[3] : 1;
            return out;
        }
        
        template <typename BufferType> inline
        PyObject* width(BufferType const& buffer) {
            return py::convert(buffer.extent[0] ? buffer.extent[0] : 0);
        }
        
        template <typename BufferType> inline
        PyObject* height(BufferType const& buffer) {
            return py::convert(buffer.extent[1] ? buffer.extent[1] : 0);
        }
        
        template <typename BufferType> inline
        PyObject* planes(BufferType const& buffer) {
            return py::convert(buffer.extent[2] ? buffer.extent[2] : 0);
        }
        
        template <typename BufferType> inline
        PyObject* shape(BufferType const& buffer) {
            switch (im::buffer::ndims(buffer)) {
                case 1:
                    return py::tuplize(buffer.extent[0]);
                case 2:
                    return py::tuplize(buffer.extent[0],
                                       buffer.extent[1]);
                case 3:
                    return py::tuplize(buffer.extent[0],
                                       buffer.extent[1],
                                       buffer.extent[2]);
                case 4:
                    return py::tuplize(buffer.extent[0],
                                       buffer.extent[1],
                                       buffer.extent[2],
                                       buffer.extent[3]);
                default:
                    return py::tuplize();
            }
            return py::tuplize();
        }
        
        template <typename BufferType> inline
        PyObject* strides(BufferType const& buffer) {
            switch (im::buffer::ndims(buffer)) {
                case 1:
                    return py::tuplize(buffer.stride[0]);
                case 2:
                    return py::tuplize(buffer.stride[0],
                                       buffer.stride[1]);
                case 3:
                    return py::tuplize(buffer.stride[0],
                                       buffer.stride[1],
                                       buffer.stride[2]);
                case 4:
                    return py::tuplize(buffer.stride[0],
                                       buffer.stride[1],
                                       buffer.stride[2],
                                       buffer.stride[3]);
                default:
                    return py::tuplize();
            }
            return py::tuplize();
        }
        
        template <typename BufferType> inline
        PyObject* min(BufferType const& buffer) {
            switch (im::buffer::ndims(buffer)) {
                case 1:
                    return py::tuplize(buffer.min[0]);
                case 2:
                    return py::tuplize(buffer.min[0],
                                       buffer.min[1]);
                case 3:
                    return py::tuplize(buffer.min[0],
                                       buffer.min[1],
                                       buffer.min[2]);
                case 4:
                    return py::tuplize(buffer.min[0],
                                       buffer.min[1],
                                       buffer.min[2],
                                       buffer.min[3]);
                default:
                    return py::tuplize();
            }
            return py::tuplize();
        }
        
    } /* namespace buffer */
    
} /* namespace im */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_BUFFER_HPP_