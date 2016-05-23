/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "buffer.hpp"

namespace im {
    
    namespace buffer {
        
        buffer_t* heapcopy(buffer_t const* buffer) {
            buffer_t* out = new buffer_t{
                buffer->dev,
                buffer->host,
                { buffer->extent[0], buffer->extent[1], buffer->extent[2], buffer->extent[3] },
                { buffer->stride[0], buffer->stride[1], buffer->stride[2], buffer->stride[3] },
                { buffer->min[0],    buffer->min[1],    buffer->min[2],    buffer->min[3]    },
                buffer->elem_size,
                buffer->host_dirty,
                buffer->dev_dirty
            };
            return out;
        }
        
        buffer_t* heapcopy(Py_buffer const* pybuffer) {
            buffer_t* out = new buffer_t{
                0,
                reinterpret_cast<uint8_t*>(pybuffer->buf),
                { static_cast<int32_t>(pybuffer->shape[0]),
                  static_cast<int32_t>(pybuffer->shape[1]),
                  static_cast<int32_t>(pybuffer->shape[2]),
                  0 },
                { static_cast<int32_t>(pybuffer->strides[0]),
                  static_cast<int32_t>(pybuffer->strides[1]),
                  static_cast<int32_t>(pybuffer->strides[2]),
                  0 },
                { 0, 0, 0, 0 },
                static_cast<int32_t>(pybuffer->itemsize),
                false,
                false
            };
            return out;
        }
        
        buffer_t* heapcopy(PyArrayObject* array) {
            buffer_t* out;
            switch (PyArray_NDIM(array)) {
                case 1:
                    out = new buffer_t{ 0,
                        reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                        {
                            static_cast<int32_t>(PyArray_DIM(array, 0)),
                            0, 0, 0
                        },
                        {
                            static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                            0, 0, 0
                        },
                        { 0, 0, 0, 0 },
                        static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                        false, false
                    };
                    break;
                case 2:
                    out = new buffer_t{ 0,
                        reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                        {
                            static_cast<int32_t>(PyArray_DIM(array, 0)),
                            static_cast<int32_t>(PyArray_DIM(array, 1)),
                            0, 0
                        },
                        {
                            static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                            static_cast<int32_t>(PyArray_STRIDE(array, 1)),
                            0, 0
                        },
                        { 0, 0, 0, 0 },
                        static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                        false, false
                    };
                    break;
                case 3:
                    out = new buffer_t{ 0,
                        reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                        {
                            static_cast<int32_t>(PyArray_DIM(array, 0)),
                            static_cast<int32_t>(PyArray_DIM(array, 1)),
                            static_cast<int32_t>(PyArray_DIM(array, 2)),
                            0
                        },
                        {
                            static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                            static_cast<int32_t>(PyArray_STRIDE(array, 1)),
                            static_cast<int32_t>(PyArray_STRIDE(array, 2)),
                            0
                        },
                        { 0, 0, 0, 0 },
                        static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                        false, false
                    };
                    break;
                case 4:
                    out = new buffer_t{ 0,
                        reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                        {
                            static_cast<int32_t>(PyArray_DIM(array, 0)),
                            static_cast<int32_t>(PyArray_DIM(array, 1)),
                            static_cast<int32_t>(PyArray_DIM(array, 2)),
                            static_cast<int32_t>(PyArray_DIM(array, 3))
                        },
                        {
                            static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                            static_cast<int32_t>(PyArray_STRIDE(array, 1)),
                            static_cast<int32_t>(PyArray_STRIDE(array, 2)),
                            static_cast<int32_t>(PyArray_STRIDE(array, 3))
                        },
                        { 0, 0, 0, 0 },
                        static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                        false, false
                    };
                    break;
                default:
                    out = nullptr;
                    break;
            }
            return out;
        }
        
        void heapdestroy(buffer_t* buffer) {
            // delete buffer->extent;
            // delete buffer->stride;
            // delete buffer->min;
            delete buffer;
        }
        
    }
    
}

