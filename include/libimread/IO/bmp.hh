// Copyright 2014 Alexander Böhn <fish2000@gmail.com>.org>
// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_BMP_HH_
#define LIBIMREAD_IO_BMP_HH_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {

    class BMPFormat : public ImageFormat {
        public:
            using can_read = std::true_type;
            
            static const options_t OPTS() {
                const options_t O(
                    "\x42\x4d",
                    "png",
                    "image/png"
                );
                return O;
            }
            static const options_t options;
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) override;
    };
    
    namespace format {
        using BMP = BMPFormat;
    }
    
}

#endif /// LIBIMREAD_IO_BMP_HH_
