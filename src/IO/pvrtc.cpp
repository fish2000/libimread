/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <cstring>
#include <libimread/IO/pvrtc.hh>
#include <libimread/errors.hh>
#include <libimread/ext/pvr.hh>

namespace im {
    
    std::unique_ptr<Image> PVRTCFormat::read(byte_source *src,
                                             ImageFactory *factory,
                                             const options_map &opts)  {
        std::vector<byte> data = src->full_data();
        PVRTexture pvr;
        
        ePVRLoadResult res = pvr.load(&data[0], data.size());
        if (res) {
            imread_raise(CannotReadError, "File isn't a valid PVRTC texture.");
        }
    
        std::unique_ptr<Image> output(factory->create(8, pvr.height, pvr.width, 4));
    
        if (pvr.data) {
            byte *rowp = output->rowp_as<byte>(0);
            std::memcpy(rowp, &pvr.data[0], pvr.width*pvr.height*4);
        } else {
            imread_raise(CannotReadError, "Error copying PVRTC post-decompress data");
        }
    
        return output;
    }
    
}