/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include "buffer.hpp"

namespace im {
    
    namespace buffer {
        
        buffer_t* heapcopy(buffer_t* buffer) {
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
        
        void heapdestroy(buffer_t* buffer) {
            // delete buffer->extent;
            // delete buffer->stride;
            // delete buffer->min;
            delete buffer;
        }
        
    }
    
}

