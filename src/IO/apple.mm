// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <libimread/IO/apple.hh>
#include <libimread/file.hh>

namespace im {

    namespace ns {
        std::unique_ptr<Image> IMDecodeDataVector(std::vector<byte> data,
                                                  ImageFactory *factory) {
            @autoreleasepool {
                NSData *datum;
                NSBitmapImageRep *rep;
                NSBitmapFormat format;
                
                @autoreleasepool {
                    datum = [[NSData alloc] initWithBytes:(const void *)&data[0]
                                                   length:(NSInteger)data.size()];
                    rep = [[NSBitmapImageRep alloc] initWithData:datum];
                    format = [rep bitmapFormat];
                };
                [datum release];
                
                NSInteger height = [rep pixelsHigh];
                NSInteger width = [rep pixelsWide];
                NSInteger channels = [rep samplesPerPixel];
                int bpp = (int)[rep bitsPerSample];
                int siz = (bpp / 8) + bool(bpp % 8);
                
                std::unique_ptr<Image> output(factory->create(
                    bpp, height, width, channels));
                
                if (format & NSFloatingPointSamplesBitmapFormat) {
                    float *frowp = output->rowp_as<float>(0);
                    std::memcpy(frowp, (float *)[rep bitmapData], siz*height*width*channels);
                } else {
                    byte *rowp = output->rowp_as<byte>(0);
                    std::memcpy(rowp, (byte *)[rep bitmapData], siz*height*width*channels);
                }
                // NSLog(@"ABOUT TO RELEASE THEM.");
                [rep release];
                return output;
            };
        }
    }
    
    std::unique_ptr<Image> NSImageFormat::read(byte_source *src,
                                               ImageFactory *factory,
                                               const options_map &opts)  {
        std::vector<byte> data = src->full_data();
        @autoreleasepool {
            // NSLog(@"About To Start This Shit: %@",
            //     [[NSString alloc] initWithUTF8String:dynamic_cast<FileSource*>(src)->path()]);
            return ns::IMDecodeDataVector(data, factory);
        };
    }
    
}