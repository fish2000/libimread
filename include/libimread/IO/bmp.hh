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
            
            std::unique_ptr<Image> read(byte_source *src,
                                        ImageFactory *factory,
                                        const options_map &opts);
    };
    
    namespace format {
        using BMP = BMPFormat;
    }
    
}

#endif // LPC_BMP_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012

