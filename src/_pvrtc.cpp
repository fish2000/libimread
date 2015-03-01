// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <stdio.h>
#include <string.h>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/_pvrtc.hh>
#include <libimread/tools.hh>

#include <libimread/pvr.hh>

namespace im {
    
    std::unique_ptr<Image> PVRTCFormat::read(byte_source* src, ImageFactory* factory, const options_map& opts) {
        std::vector<byte> data = full_data(*src);
        PVRTexture pvr;
        
        ePVRLoadResult res = pvr.load(&data[0], data.size());
        if (res) {
            throw CannotReadError("im::PVRTCFormat::read(): File isn't a valid PVRTC texture.");
        }
    
        std::unique_ptr<Image> output(factory->create(8, pvr.height, pvr.width, 4));
    
        if (pvr.data) {
            byte* rowp = output->rowp_as<byte>(0);
            memcpy(rowp, &pvr.data[0], pvr.width*pvr.height*4);
        } else {
            throw CannotReadError("im::PVRTCFormat::read(): Error reading PVRTC file.");
        }
    
        return output;
    }
    
}