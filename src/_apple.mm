// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <cstring>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/_apple.hh>
#include <libimread/tools.hh>

namespace ns {
    std::unique_ptr<Image> IMDecodeDataVector(std::vector<byte> data,
                                              ImageFactory *factory) {
        @autoreleasepool {
            NSData *data = [NSData dataWithBytes:(const void *)&data[0]
                                          length:(NSInteger)data.size()];
            NSBitmapImageRep *rep = [NSBitmapImageRep imageRepWithData:data];
            NSBitmapFormat format = [rep bitmapFormat];
            
            NSInteger height = [rep pixelsHigh];
            NSInteger width = [rep pixelsWide];
            NSInteger channels = [rep samplesPerPixel];
            
            std::unique_ptr<Image> output(factory->create(
                [rep bitsPerSample],
                height, width, channels);
            
            //if (bitmapFormat & NSAlphaFirstBitmapFormat) {}
            //if (bitmapFormat & NSAlphaNonpremultipliedBitmapFormat) {}
            //if (bitmapFormat & NSFloatingPointSamplesBitmapFormat) {}
            
            byte* rowp = output->rowp_as<byte>(0);
            std::memcpy(rowp, [rep bitmapData], height*width*channels);
            return output;
        };
    }
}

namespace im {
    std::unique_ptr<Image> NSImageFormat::read(byte_source* src,
                                               ImageFactory* factory,
                                               const options_map& opts) {
        std::vector<byte> data = full_data(*src);
        return ns::IMDecodeDataVector(data, factory);
    }
}