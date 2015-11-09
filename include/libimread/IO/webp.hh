/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_WEBP_HH_
#define LIBIMREAD_IO_WEBP_HH_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    class WebPFormat : public ImageFormat {
        public:
            using can_read = std::true_type;
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                const options_map& opts) override;
    };
    
    namespace format {
        using WebP = WebPFormat;
    }
    
}


#endif /// LIBIMREAD_IO_WEBP_HH_
