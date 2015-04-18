// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_BMP_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012
#define LPC_BMP_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012

#include <sstream>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/tools.hh>

namespace im {

    class BMPFormat : public ImageFormat {
        public:
            typedef std::true_type can_read;
            
            static bool match_format(byte_source *src) {
                return match_magic(src, "\x42\x4d", 2);
            }
            
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) override;
    };
    
    namespace format {
        using BMP = BMPFormat;
    }
    
}

#endif // LPC_BMP_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012

