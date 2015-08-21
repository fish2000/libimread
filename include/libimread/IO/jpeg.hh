/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_JPEG_HH_
#define LIBIMREAD_IO_JPEG_HH_

#include <cstdio>
#include <csetjmp>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {

    class JPEGFormat : public ImageFormat {
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            static const options_t OPTS() {
                const options_t O(
                    "\xff\xd8\xff",
                    "jpg",
                    "image/jpeg"
                );
                return O;
            }
            static const options_t options;
            
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


#endif /// LIBIMREAD_IO_JPEG_HH_
