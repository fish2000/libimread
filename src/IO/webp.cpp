// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <libimread/IO/webp.hh>

namespace im {

    Image WebPFormat::read(byte_source *src,
                           ImageFactory *factory,
                           const options_map &opts) {
        std::vector<byte> data = src->full_data();
        int w, h;
        int ok = WebPGetInfo(&data[0], data.size(), &w, &h);
        
        if (!ok) {
            throw CannotReadError("im::WebPFormat::read(): File does not validate as WebP");
        }
        
        Image output(factory->create(8, h, w, 4));
        const int stride = w*4;
        const byte *p = WebPDecodeRGBAInto(
                &data[0], data.size(),
                output.rowp_as<byte>(0), h*stride, stride);
        
        if (p != output.rowp_as<byte>(0)) {
            throw CannotReadError("im::WebPFormat::read(): Error in decoding file");
        }
        
        return output;
    }

}
