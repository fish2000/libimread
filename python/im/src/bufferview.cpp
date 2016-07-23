/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <Python.h>
#include <Halide.h>
#include "buffer.hpp"
#include "bufferview.hpp"

namespace im {
    
    namespace buffer {
        
        View::View() noexcept {}
        
        View::View(halotype_t const& halotype,
                   int width, int height, int planes)
            :htype(halotype)
            ,allocation(new uint8_t[width * height * planes * htype.bytes()])
            {
                shared = shared_t(new buffer_t{
                    0, allocation.get(),
                    { static_cast<int32_t>(width),
                      static_cast<int32_t>(height),
                      static_cast<int32_t>(planes),
                      0 },
                    { static_cast<int32_t>(1),
                      static_cast<int32_t>(width),
                      static_cast<int32_t>(width*height),
                      0 },
                    { 0, 0, 0, 0 },
                    static_cast<int32_t>(htype.bytes()),
                    false, false
                });
            }
        
        View::View(View const& other) noexcept
            :shared(other.shared)
            ,htype(Halide::UInt(shared->elem_size * 8))
            {}
        
        View::View(View&& other) noexcept
            :shared(std::move(other.shared))
            ,htype(Halide::UInt(shared->elem_size * 8))
            ,allocation(std::move(other.allocation))
            {
                shared->host = allocation.get();
            }
        
        View::View(buffer_t const* bt)
            :shared(buffer::heapcopy(bt),
                    buffer::deleter_t<buffer_t>{})
            ,htype(Halide::UInt(shared->elem_size * 8))
            {}
        
        View::View(Py_buffer const* pybt)
            :shared(buffer::heapcopy(pybt),
                    buffer::deleter_t<buffer_t>{})
            ,htype(Halide::UInt(shared->elem_size * 8))
            {}
        
        View::~View() {}
        
        uint8_t* View::data() const noexcept {
            return shared->host;
        };
        
        uint8_t* View::data(int s) const {
            return shared->host + std::ptrdiff_t(s);
        };
        
        View::halotype_t View::type() const {
            return htype;
        };
        
        buffer_t* View::buffer_ptr() const {
            return shared.get();
        };
        
        int View::nbits() const noexcept {
            return htype.bits();
        };
        
        int View::nbytes() const {
            const int bits = htype.bits();
            return (bits / 8) + bool(bits % 8);
        };
        
        int View::ndims() const {
            return buffer::ndims(*shared.get());
        };
        
        int View::dim(int d) const noexcept {
            return shared->extent[d];
        };
        
        int View::stride(int s) const noexcept {
            return shared->stride[s];
        };
        
        int View::min(int s) const noexcept {
            return shared->min[s];
        }
        
        bool View::is_signed() const noexcept {
            return !htype.is_uint();
        };
        
        bool View::is_floating_point() const noexcept {
            return htype.is_float();
        };
        
        off_t View::rowp_stride() const noexcept {
            return shared->extent[2] ? shared->extent[2] : 0;
        }
        
        void* View::rowp(int r) const {
            uint8_t* host = data();
            host += off_t(r * rowp_stride());
            return static_cast<void*>(host);
        };
        
        ViewFactory::ViewFactory() noexcept
            :fname("")
            {}
        
        ViewFactory::ViewFactory(std::string const& nm) noexcept
            :fname(nm)
            {}
        
        ViewFactory::~ViewFactory() {}
        
        std::string const& ViewFactory::name() noexcept                      { return fname; }
        std::string const& ViewFactory::name(std::string const& nm) noexcept { fname = nm;
                                                                               return name(); }
        
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
        
        ViewFactory::unique_t ViewFactory::create(int nbits,
                                                  int xHEIGHT, int xWIDTH, int xDEPTH,
                                                  int d3, int d4) {
            Halide::Type htype = Halide::UInt(nbits);
            ViewFactory::image_t out(htype, xWIDTH, xHEIGHT, xDEPTH);
            return ViewFactory::unique_t(
                new ViewFactory::image_t(std::move(out)));
        }
        
#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH
        
    }
}

