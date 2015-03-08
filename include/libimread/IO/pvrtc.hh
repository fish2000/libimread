// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_PVRTC_H_INCLUDE_GUARD_
#define LPC_PVRTC_H_INCLUDE_GUARD_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    class PVRTCFormat : public ImageFormat {
        public:
            typedef std::true_type can_read;
            
            std::unique_ptr<Image> read(byte_source *src,
                                        ImageFactory *factory,
                                        const options_map &opts);
    };
    
    namespace format {
        using PVR = PVRTCFormat;
    }
    

}


#endif // LPC_PVRTC_H_INCLUDE_GUARD_
