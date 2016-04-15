/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_WEBP_HH_
#define LIBIMREAD_IO_WEBP_HH_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    class WebPFormat : public ImageFormatBase<WebPFormat> {
        public:
            using can_read = std::true_type;
            
            // DECLARE_OPTIONS(
            //     base64::encode("RIFF", 4), 4, /// see https://en.wikipedia.org/wiki/WebP
            //     "webp",
            //     "image/webp");
            
            DECLARE_OPTIONS(
                _signature = base64::encode("RIFF", 4),
                _siglength = 4,
                _suffix = "webp",
                _mimetype = "image/webp"
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                const options_map& opts) override;
    };
    
    namespace format {
        using WebP = WebPFormat;
    }
    
}


#endif /// LIBIMREAD_IO_WEBP_HH_
