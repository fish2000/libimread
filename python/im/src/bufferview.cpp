/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <Python.h>
#include <Halide.h>
#include "buffer.hpp"
#include "bufferview.hpp"

namespace im {
    
    namespace buffer {
        
        View::View() {}
        
        View::View(View const& other)
            :shared(other.shared)
            ,htype(Halide::UInt(8))
            {}
        
        View::View(View&& other) noexcept
            :shared(std::move(other.shared))
            ,htype(Halide::UInt(8))
            {}
        
        View::View(buffer_t const* bt)
            :shared(buffer::heapcopy(bt),
                    buffer::deleter_t())
            ,htype(Halide::UInt(8))
            {}
        
        View::View(Py_buffer const* pybt)
            :shared(buffer::heapcopy(pybt),
                    buffer::deleter_t())
            ,htype(Halide::UInt(8))
            {}
        
        View::~View() {}
        
        uint8_t* View::data() const {
            return shared->host;
        };
        
        uint8_t* View::data(int s) const {
            return shared->host + std::ptrdiff_t(s);
        };
        
        Halide::Type View::type() const {
            return htype;
        };
        
        buffer_t* View::buffer_ptr() const {
            return shared.get();
        };
        
        int View::nbits() const {
            return htype.bits();
        };
        
        int View::nbytes() const {
            const int bits = htype.bits();
            return (bits / 8) + bool(bits % 8);
        };
        
        int View::ndims() const {
            return buffer::ndims(*shared.get());
        };
        
        int View::dim(int d) const {
            return shared->extent[d];
        };
        
        int View::stride(int s) const {
            return shared->stride[s];
        };
        
        int View::min(int s) const {
            return shared->min[s];
        }
        
        bool View::is_signed() const {
            return !htype.is_uint();
        };
        
        bool View::is_floating_point() const {
            return htype.is_float();
        };
        
        off_t View::rowp_stride() const {
            return shared->extent[2] ? shared->extent[2] : 0;
        }
        
        void* View::rowp(int r) const {
            uint8_t* host = data();
            host += off_t(r * rowp_stride());
            return static_cast<void*>(host);
        };
    
    }
}

