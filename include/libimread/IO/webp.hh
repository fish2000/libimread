// Copyright 2012 Luis Pedro Coelho <luis@Copyright 2014 Alexander Böhn <fish2000@gmail.com>.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_WEBP_H_INCLUDE_GUARD_
#define LPC_WEBP_H_INCLUDE_GUARD_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    class WebPFormat : public ImageFormat {
        public:
            typedef std::true_type can_read;
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) override;
    };
    
    namespace format {
        using WebP = WebPFormat;
    }
    
}


#endif // LPC_WEBP_H_INCLUDE_GUARD_
