
#ifndef LIBIMREAD_PIXELS_HH_
#define LIBIMREAD_PIXELS_HH_

#include <cstdint>

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
    
    }


}

#endif /// LIBIMREAD_PIXELS_HH_
