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
#include <libimread/endian.hh>
#include "image_compression/internal/color_types.h"
#include "image_compression/internal/color_util.h"
#include "image_compression/public/compressed_image.h"
#include "image_compression/public/pvrtc_compressor.h"
namespace aux = ::imagecompression;

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
            imread_raise(PVRTCIOError,
                "File isn't a valid PVRTC texture.");
        }
        
        std::unique_ptr<Image> output(factory->create(8, pvr.height,
                                                         pvr.width, 4));
        
        if (pvr.data) {
            int siz = pvr.width * pvr.height * 4;
            byte* rowp = output->rowp_as<byte>(0);
            std::memcpy(rowp, &pvr.data[0], siz);
        } else {
            imread_raise(PVRTCIOError,
                "Error copying PVRTC post-decompress data");
        }
        
        return output;
    }
    
    namespace detail {
        using bytebuffer_t  = std::unique_ptr<byte[]>;
        using pixelbuffer_t = std::unique_ptr<aux::Rgba8888[]>;
    }
    
    void PVRTCFormat::write(Image& input,
                            byte_sink* output,
                            Options const& opts) {
        /// inspect input image dimensions:
        int x = 0,
            y = 0,
            c = 0;
        const int width = input.width(),
                  height = input.height(),
                  channels = input.planes();
        
        /// check that channel count isn’t BU SAN BU SI:
        if (channels != 3 && channels != 4) {
            imread_raise(PVRTCIOError,
                "im::PVRTCFormat::write() says:   \"IMAGE COLOR MODE is BU SAN BU SI\"",
             FF("im::PVRTCFormat::write() got:    `channels` = (int){ %d }", channels),
                "im::PVRTCFormat::write() needs:  `channels` = (int){ 3 | 4 }");
        }
        
        /// allocate aux::Rgb8888 pixel buffer block,
        /// per input image dimensions:
        std::size_t pixelbuffer_size = width * height;
        detail::pixelbuffer_t pixelbuffer = std::make_unique<aux::Rgba8888[]>(pixelbuffer_size);
        
        /// prepare image view and subview:
        av::strided_array_view<byte, 3> view = input.view();
        av::strided_array_view<byte, 1> subview;
        byte* __restrict__ data = reinterpret_cast<byte*>(pixelbuffer.get());
        byte* __restrict__ rgb;
        
        /// fill pixelbuffer with image data from input:
        if (channels == 3) {
            /// set alpha value to 255 -- fully opaque:
            for (; y < height; ++y) {
                for (; x < width; ++x) {
                    rgb = data + (width * y + x);
                    subview = view[x][y];
                    for (; c < 3; ++c) {
                        rgb[c] = subview[c];
                    }
                    rgb[3] = 255;
                }
            }
        } else if (channels == 4) {
            /// use the alpha value provided by the image:
            for (; y < height; ++y) {
                for (; x < width; ++x) {
                    rgb = data + (width * y + x);
                    subview = view[x][y];
                    for (; c < 4; ++c) {
                        rgb[c] = subview[c];
                    }
                }
            }
        }
        
        /// create intermediate compressed image:
        aux::PvrtcCompressor compressor;
        std::size_t intermediate_size = compressor.ComputeCompressedDataSize(aux::CompressedImage::kRGBA,
                                                                             static_cast<uint32_t>(height),
                                                                             static_cast<uint32_t>(width));
        
        detail::bytebuffer_t bytebuffer = std::make_unique<byte[]>(intermediate_size);
        aux::CompressedImage intermediate(intermediate_size, bytebuffer.get());
        
        /// perform the actual act of compression:
        bool success = compressor.Compress(aux::CompressedImage::kRGBA,
                                           static_cast<uint32_t>(height),
                                           static_cast<uint32_t>(width),
                                           (const uint8_t*)pixelbuffer.get(),
                                          &intermediate);
        
        /// verify compression results:
        if (!success) {
            imread_raise(PVRTCIOError,
                "im::PVRTCFormat::write() says:   \"PVRTC COMPRESSOR FAILED TO COMPRESS\"");
        }
        if (!compressor.IsValidCompressedImage(intermediate)) {
            imread_raise(PVRTCIOError,
                "im::PVRTCFormat::write() says:   \"PVRTC COMPRESSOR YIELDED INVALID COMPRESSED IMAGE\"");
        }
        if (!intermediate.GetDataSize()) {
            imread_raise(PVRTCIOError,
                "im::PVRTCFormat::write() says:   \"PVRTC COMPRESSOR YIELDED ZERO-LENGTH COMPRESSED DATA\"");
        }
        
        /// write the compressed images’ data out to the sink:
        output->write((const void*)intermediate.GetData(),
                                   intermediate.GetDataSize());
        output->flush();
        
    }
    
}