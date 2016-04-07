/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_BUFFER_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_BUFFER_HPP_

#include <string>
#include <type_traits>
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
        
    } /* namespace buffer */
    
} /* namespace im */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_BUFFER_HPP_