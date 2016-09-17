/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <iod/json.hh>
#include <libimread/IO/webp.hh>
#include <webp/decode.h>

namespace im {
    
    DECLARE_FORMAT_OPTIONS(WebPFormat);
    
    std::unique_ptr<Image> WebPFormat::read(byte_source* src,
                                            ImageFactory* factory,
                                            options_map const& opts) {
        std::vector<byte> data = src->full_data();
        int w, h;
        int ok = WebPGetInfo(&data[0], data.size(), &w, &h);
        
        if (!ok) {
            imread_raise(CannotReadError,
                "File does not validate as WebP");
        }
        
        std::unique_ptr<Image> output(factory->create(8, h, w, 4));
        const int stride = w * 4;
        const byte* p = WebPDecodeRGBAInto(
                &data[0], data.size(),
                output->rowp_as<byte>(0), h * stride, stride);
        
        if (p != output->rowp_as<byte>(0)) {
            imread_raise(CannotReadError, "Error in decoding WebP file");
        }
        
        return output;
    }

}
