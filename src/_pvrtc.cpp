// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifdef PVRTC_DEBUG
#include <stdio.h>
#endif

#include <string.h>
#include "base.h"
#include "_pvrtc.h"
#include "tools.h"

#include "pvr.h"

namespace im {
    
    std::unique_ptr<Image> PVRTCFormat::read(byte_source* src, ImageFactory* factory, const options_map& opts) {
        std::vector<byte> data = full_data(*src);
        PVRTexture pvr;
    
        ePVRLoadResult res = pvr.load(&data[0], data.size());
    
        if (res) {
            throw CannotReadError("imread.imread._pvrtc: File isn't a valid PVRTC texture.");
        }
    
        std::unique_ptr<Image> output(factory->create(8, pvr.height, pvr.width, 4));
    
        if (pvr.data) {
            byte* rowp = output->rowp_as<byte>(0);
            memcpy(rowp, &pvr.data[0], pvr.width*pvr.height*4);
        } else {
            throw CannotReadError("imread.imread._pvrtc: Error reading PVRTC file.");
        }
    
        return output;
    }
    
}