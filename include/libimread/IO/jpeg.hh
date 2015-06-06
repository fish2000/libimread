/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_JPEG_HH_
#define LIBIMREAD_IO_JPEG_HH_

#include <cstdio>
#include <csetjmp>
//#include <sstream>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

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


#endif /// LIBIMREAD_IO_JPEG_HH_
