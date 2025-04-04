/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_WEBP_HH_
#define LIBIMREAD_IO_WEBP_HH_

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

namespace im {
    
    class WebPFormat : public ImageFormatBase<WebPFormat> {
        public:
            using can_read = std::true_type;
            
            /// see https://en.wikipedia.org/wiki/WebP
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("RIFF", 4)
                },
                _suffixes = { "webp", "wbp" },
                _mimetype = "image/webp"
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                Options const& opts) override;
    };
    
    namespace format {
        using WebP = WebPFormat;
    }
    
}


#endif /// LIBIMREAD_IO_WEBP_HH_
