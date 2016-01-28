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
        inline void convert(uint8_t in,     uint8_t& out)       { out = in; }
        inline void convert(uint16_t in,    uint8_t& out)       { out = in >> 8; }
        inline void convert(uint32_t in,    uint8_t& out)       { out = in >> 24; }
        inline void convert(int8_t in,      uint8_t& out)       { out = in; }
        inline void convert(int16_t in,     uint8_t& out)       { out = in >> 8; }
        inline void convert(int32_t in,     uint8_t& out)       { out = in >> 24; }
        inline void convert(float in,       uint8_t& out)       { out = static_cast<uint8_t>(in * 255.0f); }
        inline void convert(double in,      uint8_t& out)       { out = static_cast<uint8_t>(in * 255.0f); }
        
        /// Convert to u16
        inline void convert(uint8_t in,     uint16_t& out)      { out = in << 8; }
        inline void convert(uint16_t in,    uint16_t& out)      { out = in; }
        inline void convert(uint32_t in,    uint16_t& out)      { out = in >> 16; }
        inline void convert(int8_t in,      uint16_t& out)      { out = in << 8; }
        inline void convert(int16_t in,     uint16_t& out)      { out = in; }
        inline void convert(int32_t in,     uint16_t& out)      { out = in >> 16; }
        inline void convert(float in,       uint16_t& out)      { out = static_cast<uint16_t>(in * 65535.0f); }
        inline void convert(double in,      uint16_t& out)      { out = static_cast<uint16_t>(in * 65535.0f); }
        
        /// Convert to u32
        inline void convert(uint32_t in,    uint32_t& out)      { out = in; }
        inline void convert(int8_t in,      uint32_t& out)      { out = in << 24; }
        inline void convert(int16_t in,     uint32_t& out)      { out = in << 16; }
        inline void convert(int32_t in,     uint32_t& out)      { out = in; }
        inline void convert(float in,       uint32_t& out)      { out = static_cast<uint32_t>(in * 16711425.0f); }
        inline void convert(double in,      uint32_t& out)      { out = static_cast<uint32_t>(in * 16711425.0f); }
        
        /// Convert from u8
        inline void convert(uint8_t in,     uint32_t& out)      { out = in << 24; }
        inline void convert(uint8_t in,     int8_t& out)        { out = in; }
        inline void convert(uint8_t in,     int16_t& out)       { out = in << 8; }
        inline void convert(uint8_t in,     int32_t& out)       { out = in << 24; }
        inline void convert(uint8_t in,     float& out)         { out = in / 255.0f; }
        inline void convert(uint8_t in,     double& out)        { out = in / 255.0f; }
        
        /// Convert from u16
        inline void convert(uint16_t in,    uint32_t& out)      { out = in << 16; }
        inline void convert(uint16_t in,    int8_t& out)        { out = in >> 8; }
        inline void convert(uint16_t in,    int16_t& out)       { out = in; }
        inline void convert(uint16_t in,    int32_t& out)       { out = in << 16; }
        inline void convert(uint16_t in,    float& out)         { out = in / 65535.0f; }
        inline void convert(uint16_t in,    double& out)        { out = in / 65535.0f; }
        
        /// Convert from u32
        inline void convert(uint32_t in,    int8_t& out)        { out = in >> 24; }
        inline void convert(uint32_t in,    int16_t& out)       { out = in >> 16; }
        inline void convert(uint32_t in,    int32_t& out)       { out = in; }
        inline void convert(uint32_t in,    float& out)         { out = in / 16711425.0f; }
        inline void convert(uint32_t in,    double& out)        { out = in / 16711425.0f; }
        
        template <typename PixelType  = byte,
                  typename OffsetType = std::ptrdiff_t>
        struct alignas(alignof(PixelType) * 16) accessor {
            
            /// Pointer difference and pixel types
            using pdiff_t = OffsetType;
            using pdata_t = PixelType;
            
            /// Restricted pointer to data buffer
            pdata_t* __restrict__ data_pointer;
            
            /// Stride values
            const pdiff_t stride0, stride1, stride2;
            
            /// One-way constructor
            explicit accessor(pdata_t* __restrict__ dp,
                              pdiff_t s0, pdiff_t s1=0, pdiff_t s2=0)
                :data_pointer(dp)
                ,stride0(s0)
                ,stride1(s1)
                ,stride2(s2)
                {}
            
            /// Inline operator accessor function
            inline pdata_t* operator()(pdiff_t x, pdiff_t y, pdiff_t z) {
                return &data_pointer[x * stride0 +
                                     y * stride1 +
                                     z * stride2];
            }
            
        };
        
        
    }


}

#endif /// LIBIMREAD_PIXELS_HH_
