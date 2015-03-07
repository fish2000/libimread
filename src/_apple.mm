// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <cstring>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/_apple.hh>

namespace im {

    namespace ns {
        std::unique_ptr<Image> IMDecodeDataVector(std::vector<byte> data,
                                                  ImageFactory *factory) {
            @autoreleasepool {
                NSData *nsdata = [NSData dataWithBytes:(const void *)&data[0]
                                              length:(NSInteger)data.size()];
                NSBitmapImageRep *rep = [NSBitmapImageRep imageRepWithData:nsdata];
                NSBitmapFormat format = [rep bitmapFormat];
                
                NSInteger height = [rep pixelsHigh];
                NSInteger width = [rep pixelsWide];
                NSInteger channels = [rep samplesPerPixel];
                
                std::unique_ptr<Image> output(factory->create(
                    (int)[rep bitsPerSample],
                    height, width, channels));
                
                //if (bitmapFormat & NSAlphaFirstBitmapFormat) {}
                //if (bitmapFormat & NSAlphaNonpremultipliedBitmapFormat) {}
                //if (bitmapFormat & NSFloatingPointSamplesBitmapFormat) {}
                
                if (format & NSFloatingPointSamplesBitmapFormat) {
                    float *frowp = output->rowp_as<float>(0);
                    std::memcpy(frowp, (float *)[rep bitmapData], height*width*channels);
                } else {
                    byte *rowp = output->rowp_as<byte>(0);
                    std::memcpy(rowp, [rep bitmapData], height*width*channels);
                }
                return output;
            };
        }
    }
    
    std::unique_ptr<Image> NSImageFormat::read(byte_source* src,
                                               ImageFactory* factory,
                                               const options_map& opts) {
        std::vector<byte> data = src->full_data();
        return ns::IMDecodeDataVector(data, factory);
    }
}