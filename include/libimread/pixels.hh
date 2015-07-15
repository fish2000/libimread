/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PIXELS_HH_
#define LIBIMREAD_PIXELS_HH_

#include <cstdint>
#include <cstddef>
#include <libimread/libimread.hpp>

namespace im {

    /// Pixel number converters -- cribbed from Halide's image_io.h
    namespace pix {
        
        /// Convert to u8
        inline void convert(uint8_t in, uint8_t &out) { out = in; }
        inline void convert(uint16_t in, uint8_t &out) { out = in >> 8; }
        inline void convert(uint32_t in, uint8_t &out) { out = in >> 24; }
        inline void convert(int8_t in, uint8_t &out) { out = in; }
        inline void convert(int16_t in, uint8_t &out) { out = in >> 8; }
        inline void convert(int32_t in, uint8_t &out) { out = in >> 24; }
        inline void convert(float in, uint8_t &out) { out = static_cast<uint8_t>(in * 255.0f); }
        inline void convert(double in, uint8_t &out) { out = static_cast<uint8_t>(in * 255.0f); }
    
        /// Convert to u16
        inline void convert(uint8_t in, uint16_t &out) { out = in << 8; }
        inline void convert(uint16_t in, uint16_t &out) { out = in; }
        inline void convert(uint32_t in, uint16_t &out) { out = in >> 16; }
        inline void convert(int8_t in, uint16_t &out) { out = in << 8; }
        inline void convert(int16_t in, uint16_t &out) { out = in; }
        inline void convert(int32_t in, uint16_t &out) { out = in >> 16; }
        inline void convert(float in, uint16_t &out) { out = static_cast<uint16_t>(in * 65535.0f); }
        inline void convert(double in, uint16_t &out) { out = static_cast<uint16_t>(in * 65535.0f); }
    
        /// Convert from u8
        inline void convert(uint8_t in, uint32_t &out) { out = in << 24; }
        inline void convert(uint8_t in, int8_t &out) { out = in; }
        inline void convert(uint8_t in, int16_t &out) { out = in << 8; }
        inline void convert(uint8_t in, int32_t &out) { out = in << 24; }
        inline void convert(uint8_t in, float &out) { out = in / 255.0f; }
        inline void convert(uint8_t in, double &out) { out = in / 255.0f; }
    
        /// Convert from u16
        inline void convert(uint16_t in, uint32_t &out) { out = in << 16; }
        inline void convert(uint16_t in, int8_t &out) { out = in >> 8; }
        inline void convert(uint16_t in, int16_t &out) { out = in; }
        inline void convert(uint16_t in, int32_t &out) { out = in << 16; }
        inline void convert(uint16_t in, float &out) { out = in / 65535.0f; }
        inline void convert(uint16_t in, double &out) { out = in / 65535.0f; }
        
        template <typename P = byte, typename O = std::ptrdiff_t>
        struct alignas(alignof(P) * 16) accessor {
            /// Pointer difference type
            typedef O pd_t;
            
            /// Stride values
            const pd_t stride0, stride1, stride2;
            
            /// Pointer to data buffer
            P *data_ptr;
            
            /// One-way constructor
            explicit accessor(P *dp, pd_t s0, pd_t s1, pd_t s2)
                :data_ptr(dp), stride0(s0), stride1(s1), stride2(s2)
                {}
            
            /// Inline operator accessor function
            inline P *operator()(pd_t x, pd_t y, pd_t z) {
                return &data_ptr[x*stride0 +
                                 y*stride1 +
                                 z*stride2];
            }
            
        };
        
        
    }


}

#endif /// LIBIMREAD_PIXELS_HH_
