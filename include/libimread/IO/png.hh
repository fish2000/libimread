// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <cstring>
#include <csetjmp>
#include <vector>
#include <sstream>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/tools.hh>
#include <libimread/pixels.hh>
//#include <libimread/ext/io_png.hh>

#ifdef __APPLE__
    #include <libpng16/png.h>   /* this is the Homebrew path */
#else
    #include <png.h>            /* this is the standard location */
#endif

namespace im {
    
    /*
    using namespace symbols::s;
    
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
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) override;
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
    };
    
    namespace format {
        using PNG = PNGFormat;
    }
    
}


#endif // LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
