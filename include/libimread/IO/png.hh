// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <cstring>
#include <vector>
#include <sstream>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/tools.hh>
#include <libimread/ext/io_png.hh>

#ifdef __APPLE__
    #include <libpng16/png.h>   /* this is the Homebrew path */
#else
    #include <png.h>            /* this is the standard location */
#endif

namespace im {
    
    /// Number converters -- cribbed from Halide's image_io.h
    namespace png {
        
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
    
    using namespace symbols::s;
    
    /*
    auto options =
    D(
        _compression_level     = -1,
        _backend               = "io_png"
    );
    */
    
    class PNGFormat : public ImageFormat {
        public:
            typedef std::true_type can_read;
            typedef std::true_type can_write;
            
            /// NOT AN OVERRIDE:
            static bool match_format(byte_source *src) {
                return match_magic(src, "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A", 8);
            }

            std::unique_ptr<Image> read(byte_source *src,
                                        ImageFactory *factory,
                                        const options_map &opts);
            void write(Image *input,
                       byte_sink *output,
                       const options_map &opts);
    };
    
    namespace format {
        using PNG = PNGFormat;
    }
    
}


#endif // LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
