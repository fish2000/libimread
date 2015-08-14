/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LPC_PVRTC_H_INCLUDE_GUARD_
#define LPC_PVRTC_H_INCLUDE_GUARD_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    class PVRTCFormat : public ImageFormat {
        public:
            using can_read = std::true_type;
            
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
