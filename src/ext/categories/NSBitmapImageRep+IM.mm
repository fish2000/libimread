/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSBitmapImageRep+IM.hh>

using im::byte;
using im::Image;
using im::ImageFactory;

@implementation NSBitmapImageRep (AXBitmapImageRepAdditions)

+ (NSBitmapImageRep *) imageRepWithByteVector:(const std::vector<byte>&)byteVector {
    NSBitmapImageRep* rep;
    @autoreleasepool {
        NSData* datum;
        datum = [[NSData alloc] initWithBytes:(const void*)&byteVector[0]
                                       length:(NSInteger)byteVector.size()];
        rep = [[NSBitmapImageRep alloc] initWithData:datum];
        #if !__has_feature(objc_arc)
            [datum release];
        #endif
    };
    return rep;
}

- initWithByteVector:(const std::vector<byte>&)byteVector {
    NSData* datum;
    datum = [[NSData alloc] initWithBytes:(const void*)&byteVector[0]
                                   length:(NSInteger)byteVector.size()];
    [self initWithData:datum];
    #if !__has_feature(objc_arc)
        [datum release];
    #endif
    return self;
}

- (std::unique_ptr<Image>) imageUsingImageFactory:(ImageFactory*)factory {
    NSInteger height = [self pixelsHigh];
    NSInteger width = [self pixelsWide];
    NSInteger channels = [self samplesPerPixel];
    int bps = (int)[self bitsPerSample];
    int siz = (bps / 8) + bool(bps % 8);
    
    std::unique_ptr<Image> output(factory->create(
        bps, height, width, channels));
    
    if ([self bitmapFormat] & NSFloatingPointSamplesBitmapFormat) {
        float* frowp = output->rowp_as<float>(0);
        std::memcpy(frowp, reinterpret_cast<float*>([self bitmapData]),
                           siz*height*width*channels);
    } else {
        byte* irowp = output->rowp_as<byte>(0);
        std::memcpy(irowp, static_cast<byte*>([self bitmapData]),
                           siz*height*width*channels);
    }
    
    return output;
}

@end

