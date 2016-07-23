/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_BUFFERVIEW_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_BUFFERVIEW_HPP_

#include <memory>
// #include "private/buffer_t.h"
// #include <Python.h>
// #include <Halide.h>
#include <libimread/image.hh>

/// forward-declare buffer_t and Py_buffer:
extern "C" {
    struct buffer_t;
    struct Py_buffer;
}

/// forward-declare Halide::Type:
namespace Halide {
    class Type;
}

namespace im {
    
    namespace buffer {
        
        /// forward-declare buffer heap deleter template:
        template <typename BufferType> struct deleter;
        
        class View : public Image {
            
            public:
                
                using shared_t = std::shared_ptr<buffer_t,
                                        buffer::deleter_t>;
                
                View();
                View(View const& other);
                View(View&& other) noexcept;
                View(buffer_t const* bt);
                View(Py_buffer const* pybt);
                virtual ~View();
                
                /// Image API
                virtual uint8_t* data() const;
                virtual uint8_t* data(int s) const;
                Halide::Type type() const;
                buffer_t* buffer_ptr() const;
                virtual int nbits() const override;
                virtual int nbytes() const override;
                virtual int ndims() const override;
                virtual int dim(int d) const override;
                virtual int stride(int s) const override;
                virtual int min(int s) const override;
                virtual bool is_signed() const override;
                virtual bool is_floating_point() const override;
                inline off_t rowp_stride() const;
                virtual void* rowp(int r) const override;
                
            private:
                
                shared_t shared;
                Halide::Type htype;
        };
    
    }
    
} /* namespace im */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_BUFFERVIEW_HPP_