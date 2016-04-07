/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include "buffer.hpp"

namespace im {
    
    namespace buffer {
        
        buffer_t* heapcopy(buffer_t* buffer) {
            buffer_t* out = new buffer_t{
                buffer->dev,
                buffer->host,
                { 0, 0, 0, 0 },
                { 0, 0, 0, 0 },
                { 0, 0, 0, 0 },
                buffer->elem_size,
                buffer->host_dirty,
                buffer->dev_dirty
            };
            // out->extent = new int32_t[4];
            // out->stride = new int32_t[4];
            // out->min = new int32_t[4];
            for (int idx = 0; idx < 4; ++idx) {
                out->extent[idx] = buffer->extent[idx];
                out->stride[idx] = buffer->stride[idx];
                out->min[idx]    = buffer->min[idx];
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

