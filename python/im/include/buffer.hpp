/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_BUFFER_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_BUFFER_HPP_

#include <string>
#include <type_traits>
#include <Python.h>
#include "private/buffer_t.h"
#include <libimread/libimread.hpp>

namespace im {
    
    namespace buffer {
        
        buffer_t* heapcopy(buffer_t* buffer);
        void heapdestroy(buffer_t* buffer);
        
        template <typename BufferType>
        struct deleter {
            constexpr deleter() noexcept = default;
            template <typename U> deleter(deleter<U> const&) noexcept {}
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
        PyObject* shape(BufferType const& buffer) {
            switch (im::buffer::ndims(buffer)) {
                case 1:
                    return Py_BuildValue("(i)",     buffer.extent[0]);
                case 2:
                    return Py_BuildValue("(ii)",    buffer.extent[0],
                                                    buffer.extent[1]);
                case 3:
                    return Py_BuildValue("(iii)",   buffer.extent[0],
                                                    buffer.extent[1],
                                                    buffer.extent[2]);
                case 4:
                    return Py_BuildValue("(iiii)",  buffer.extent[0],
                                                    buffer.extent[1],
                                                    buffer.extent[2],
                                                    buffer.extent[3]);
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
        template <typename BufferType> inline
        PyObject* strides(BufferType const& buffer) {
            switch (im::buffer::ndims(buffer)) {
                case 1:
                    return Py_BuildValue("(i)",     buffer.stride[0]);
                case 2:
                    return Py_BuildValue("(ii)",    buffer.stride[0],
                                                    buffer.stride[1]);
                case 3:
                    return Py_BuildValue("(iii)",   buffer.stride[0],
                                                    buffer.stride[1],
                                                    buffer.stride[2]);
                case 4:
                    return Py_BuildValue("(iiii)",  buffer.stride[0],
                                                    buffer.stride[1],
                                                    buffer.stride[2],
                                                    buffer.stride[3]);
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
        template <typename BufferType> inline
        PyObject* min(BufferType const& buffer) {
            switch (im::buffer::ndims(buffer)) {
                case 1:
                    return Py_BuildValue("(i)",     buffer.min[0]);
                case 2:
                    return Py_BuildValue("(ii)",    buffer.min[0],
                                                    buffer.min[1]);
                case 3:
                    return Py_BuildValue("(iii)",   buffer.min[0],
                                                    buffer.min[1],
                                                    buffer.min[2]);
                case 4:
                    return Py_BuildValue("(iiii)",  buffer.min[0],
                                                    buffer.min[1],
                                                    buffer.min[2],
                                                    buffer.min[3]);
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
        
    } /* namespace buffer */
    
} /* namespace im */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_BUFFER_HPP_