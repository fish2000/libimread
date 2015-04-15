// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_PVRTC_H_INCLUDE_GUARD_
#define LPC_PVRTC_H_INCLUDE_GUARD_

#include <cstdio>
#include <cstring>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/ext/pvr.h>

namespace im {
    
    class PVRTCFormat : public ImageFormat {
        public:
            typedef std::true_type can_read;
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) override;
    };
    
    namespace format {
        using PVR = PVRTCFormat;
        using PVRTC = PVRTCFormat;
    }
    

}


#endif // LPC_PVRTC_H_INCLUDE_GUARD_
