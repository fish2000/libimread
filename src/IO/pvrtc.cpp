/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <cstring>
#include <iod/json.hh>
#include <libimread/IO/pvrtc.hh>
#include <libimread/seekable.hh>
#include <libimread/options.hh>
#include <libimread/errors.hh>

/// support for PVRTCFormat::read():
#include <libimread/ext/pvr.hh>

/// support for PVRTCFormat::write():
#include "image_compression/public/compressed_image.h"
#include "image_compression/public/pvrtc_compressor.h"


namespace im {
    
    DECLARE_FORMAT_OPTIONS(PVRTCFormat);
    
    namespace detail {
        constexpr static const std::size_t kHeaderSize = sizeof(PVRHeader);
    }
    
    bool PVRTCFormat::match_format(byte_source* src) {
        /// Adapted from the standard match_format() case, using techniques found in
        /// PVRTexture::loadApplePVRTC() and PVRTexture::load(), from ext/pvr.cpp
        bytevec_t headerbuf(detail::kHeaderSize);
        const int bytesread = static_cast<int>(src->read(&headerbuf.front(), detail::kHeaderSize));
        src->seek_relative(-bytesread);
        PVRHeader* header = (PVRHeader*)&headerbuf.front();
        return (bytesread == detail::kHeaderSize &&
                header->magic == options.signatures[0]) ||
               (countBits(src->size()) == 1);
    }
    
    std::unique_ptr<Image> PVRTCFormat::read(byte_source* src,
                                             ImageFactory* factory,
                                             Options const& opts)  {
        PVRTexture pvr;
        ePVRLoadResult result = pvr.load(src->data(), src->size());
        
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
    
    void PVRTCFormat::write(Image& input,
                            byte_sink* output,
                            Options const& opts) {
        // imread_raise_default(NotImplementedError);
        
    }
    
}