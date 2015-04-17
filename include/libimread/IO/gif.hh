// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef FISH2K_GIF_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define FISH2K_GIF_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/pixels.hh>
#include <libimread/ext/fmemopen.hh>
#include <libimread/ext/WriteGIF.h>

namespace im {
    
    class GIFFormat : public ImageFormat {
        public:
            typedef std::true_type can_write;
            typedef std::true_type can_write_multi;
            
            static bool match_format(byte_source *src) {
                /// 47 49 46 38 ("GIF8" in ASCII);
                /// ... from “File Magic Numbers” [0]
                return match_magic(src, "\x47\x49\x46\x38", 4);
            }
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
    };
    
    namespace format {
        using GIF = GIFFormat;
    }
    
}

/*
 *   [0]  http://www.astro.keele.ac.uk/oldusers/rno/Computing/File_magic.html
 */ 
#endif // FISH2K_GIF_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
