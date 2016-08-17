/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
#include <array>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "buffer.hpp"

namespace im {
    
    namespace buffer {
        
        /// COPYING HEAPCOPY: im::buffer::heapcopy(buffer_t buffer) ...
        /// heap-allocating copy-construction analogue
        
        buffer_t* heapcopy(buffer_t const* buffer) {
            return new buffer_t{
                buffer->dev,
                buffer->host,
                { buffer->extent[0], buffer->extent[1], buffer->extent[2], buffer->extent[3] },
                { buffer->stride[0], buffer->stride[1], buffer->stride[2], buffer->stride[3] },
                { buffer->min[0],    buffer->min[1],    buffer->min[2],    buffer->min[3]    },
                buffer->elem_size,
                buffer->host_dirty,
                buffer->dev_dirty
            };
        }
        
        /// SCALING HEAPCOPY: im::buffer::heapcopy(buffer_t buffer, float scale) ...
        /// assumes planar arrangement, ignores mins (for now) --
        /// NB. is it retarded to just copy over the dev/host pointers here?
        /// ... like I'm just saying dogg I can't really concieve of a case
        /// in which that would actually work without meddling
        
        buffer_t* heapcopy(buffer_t const* buffer, float scale) {
            
            std::array<int32_t, 4> extents{{ static_cast<int32_t>(buffer->extent[0] * scale),
                                             static_cast<int32_t>(buffer->extent[1] * scale),
                                             static_cast<int32_t>(buffer->extent[2]),
                                             static_cast<int32_t>(buffer->extent[3]) }};
            
            std::array<int32_t, 4> stridings{{ 1, extents[0],
                                                  extents[0]*extents[1],
                                                  extents[0]*extents[1]*extents[2] }};
            
            return new buffer_t{
                buffer->dev,
                buffer->host,
                { extents[0],   extents[1],   extents[2],   extents[3] },
                { stridings[0], stridings[1], stridings[2], stridings[3] },
                { 0, 0, 0, 0 },
                buffer->elem_size,
                buffer->host_dirty,
                buffer->dev_dirty };
        }
        
        /// CONVERTING HEAPCOPY: im::buffer::heapcopy(Py_buffer pybuffer) ...
        /// heap-allocating conversion-construction analogue --
        /// NB. in an ideal world -- a less-dirty world -- this would inspect
        /// the pybuffer->format member, which maybe points to a c-string with
        /// a structcode that we could use to, like, calculate and/or validate
        /// some of this shit here
        
        /// ... actually now that I think of it, actually, code like that should
        /// get abstracted (hear me out!) into a traited template thingy -- like this:
        ///
        ///     im::buffer::traits<buffer_t>::pixel_t               -> typename uint8_t
        ///     im::buffer::traits<buffer_t>::value_t               -> typename int32_t (extent, etc)
        ///     im::buffer::traits<buffer_t>::width(buffer)         -> whatevs
        ///     im::buffer::traits<Py_buffer>::height(pybuffer)     -> IDK TBH
        ///     im::buffer::traits<PyArrayObject>::data(array)      -> dogg it's just a data pointer
        ///                                                            that is what the pixels are
        ///
        /// ... RIGHT!??! That would be cool, and not overly architecture-astronaut-y. Yes!
        
        buffer_t* heapcopy(Py_buffer const* pybuffer) {
            return new buffer_t{
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
        }
        
        /// INSPECTING HEAPCOPY: im::buffer::heapcopy(PyArrayObject* array) ...
        /// heap-allocating explicit-inspection-construction analogue
        
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
        
        /// HEAPCOPY DESTRUCTOR: im::buffer::heapdestroy(buffer_t buffer) ...
        /// straightforward heap-allocation destructor analogue --
        /// buffer creation is normalized to the point where there aren't any
        /// variants necessitating per-member deallocation strategies
        /// or suchlike (at least not yet -- fingers crossed doggie)
        
        void heapdestroy(buffer_t* buffer) {
            // delete buffer->extent;
            // delete buffer->stride;
            // delete buffer->min;
            delete buffer;
        }
        
    }
    
}

