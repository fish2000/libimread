/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSBitmapImageRep+IM.hh>

using namespace im;

@implementation NSBitmapImageRep (IMBitmapImageRepAdditions)
+ (instancetype) imageRepWithByteVector:(const std::vector<byte>&)byteVector {
    NSBitmapImageRep *rep;
    @autoreleasepool {
        NSData *datum;
        datum = [[NSData alloc] initWithBytes:(const void *)&byteVector[0]
                                       length:(NSInteger)byteVector.size()];
        NSBitmapImageRep *rep = [NSBitmapImageRep imageRepWithData:datum];
        #if !__has_feature(objc_arc)
            [datum release];
        #endif
    };
    return rep;
}
- initWithByteVector:(const std::vector<byte>&)byteVector {
    NSData *datum;
    datum = [[NSData alloc] initWithBytes:(const void *)&byteVector[0]
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
        float *frowp = output->rowp_as<float>(0);
        std::memcpy(frowp, (float *)[self bitmapData], siz*height*width*channels);
    } else {
        byte *rowp = output->rowp_as<byte>(0);
        std::memcpy(rowp, static_cast<byte*>([self bitmapData]), siz*height*width*channels);
    }
    
    return output;
}
@end

