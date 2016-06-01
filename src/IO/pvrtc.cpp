/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <cstring>
#include <iod/json.hh>
#include <libimread/IO/pvrtc.hh>
#include <libimread/errors.hh>
#include <libimread/ext/pvr.hh>

namespace im {
    
    DECLARE_FORMAT_OPTIONS(PVRTCFormat);
    
    bool PVRTCFormat::match_format(byte_source* src) {
        /// Adapted from the standard match_format() case, using techniques found in
        /// PVRTexture::loadApplePVRTC() and PVRTexture::load(), from ext/pvr.cpp
        std::vector<byte> headerbuf(sizeof(PVRHeader));
        const int bytesread = static_cast<int>(src->read(&headerbuf.front(), sizeof(PVRHeader)));
        src->seek_relative(-bytesread);
        PVRHeader* header = (PVRHeader*)&headerbuf.front();
        return (bytesread == sizeof(PVRHeader) && header->magic == 0x21525650) ||
               (countBits(src->size()) == 1);
    }
    
    std::unique_ptr<Image> PVRTCFormat::read(byte_source* src,
                                             ImageFactory* factory,
                                             options_map const& opts)  {
        PVRTexture pvr;
        std::vector<byte> data = src->full_data();
        ePVRLoadResult result = pvr.load(&data[0], data.size());
        
        if (result) {
            imread_raise(CannotReadError,
                "File isn't a valid PVRTC texture.");
        }
        
        std::unique_ptr<Image> output(factory->create(8, pvr.height,
                                                         pvr.width, 4));
        
        if (pvr.data) {
            int siz = pvr.width * pvr.height * 4;
            byte* rowp = output->rowp_as<byte>(0);
            std::memcpy(rowp, &pvr.data[0], siz);
        } else {
            imread_raise(CannotReadError,
                "Error copying PVRTC post-decompress data");
        }
        
        return output;
    }
    
}