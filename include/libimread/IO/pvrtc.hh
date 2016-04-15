/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LPC_PVRTC_H_INCLUDE_GUARD_
#define LPC_PVRTC_H_INCLUDE_GUARD_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    class PVRTCFormat : public ImageFormatBase<PVRTCFormat> {
        public:
            using can_read = std::true_type;
            
            // DECLARE_OPTIONS(
            //     base64::encode("4\x00\x00\x00", 4), 4, /// PVRTC "magic" tag is internal
            //     "pvr",
            //     "image/x-pvr");
            
            DECLARE_OPTIONS(
                _signature = base64::encode("4\x00\x00\x00", 4),
                _siglength = 4,
                _suffix = "pvr",
                _mimetype = "image/x-pvr"
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                const options_map& opts) override;
    };
    
    namespace format {
        using PVR = PVRTCFormat;
        using PVRTC = PVRTCFormat;
    }
    

}


#endif // LPC_PVRTC_H_INCLUDE_GUARD_
