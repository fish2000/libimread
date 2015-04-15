// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_JPEG_H_INCLUDE_GUARD_THU_FEB__2_18_14_07_WET_2012
#define LPC_JPEG_H_INCLUDE_GUARD_THU_FEB__2_18_14_07_WET_2012

#include <cstdio>
#include <csetjmp>
#include <sstream>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/pixels.hh>

namespace im {

    class JPEGFormat : public ImageFormat {
        public:
            typedef std::true_type can_read;
            typedef std::true_type can_write;
            
            /// NOT AN OVERRIDE:
            static bool match_format(byte_source *src) {
                return match_magic(src, "\xff\xd8\xff", 3);
            }
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) override;
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
    };
    
    namespace format {
        using JPG = JPEGFormat;
        using JPEG = JPEGFormat;
    }
    
}


#endif // LPC_JPEG_H_INCLUDE_GUARD_THU_FEB__2_18_14_07_WET_2012
