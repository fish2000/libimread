/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <memory>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSData+IM.hh>
#include <libimread/ext/categories/NSString+STL.hh>
// #include <libimread/objc-rt/objc-rt.hh>

using im::byte;
using im::byte_source;
using im::byte_sink;
using im::NSDataSource;
using im::NSDataSink;


@implementation NSData (AXDataAdditions)

+ (NSData *) dataWithByteVector:(const std::vector<byte>&)byteVector {
    NSData* datum;
    datum = [[NSData alloc] initWithBytes:(const void*)&byteVector[0]
                                   length:(NSInteger)byteVector.size()];
    return datum;
}

+ (NSData *) dataWithByteSource:(byte_source*)byteSource {
    return [NSData dataWithByteVector:byteSource->full_data()];
}

+ (NSData *) dataWithByteSource:(byte_source*)byteSource
                         length:(NSUInteger)bytes {
    NSData* datum;
    std::unique_ptr<byte[]> buffer = std::make_unique<byte[]>(bytes);
    int idx = byteSource->read(buffer.get(),
                               static_cast<std::size_t>(bytes));
    datum = [[NSData alloc] initWithBytes:(const void*)buffer.get()
                                   length:(NSInteger)bytes];
    return datum;
}

- (NSDataSource) dataSource {
    return NSDataSource(self);
}

- (NSUInteger) writeUsingByteSink:(byte_sink*)byteSink {
    return static_cast<NSUInteger>(byteSink->write((byte*)self.bytes,
                                                   (std::size_t)self.length));
}

- (NSUInteger) writeUsingByteSink:(byte_sink*)byteSink
                           length:(NSUInteger)bytes {
    return static_cast<NSUInteger>(byteSink->write((byte*)self.bytes,
                                                   (std::size_t)bytes));
}

// + (NSData *) dataWithCFData:(const CF::Data&)cfdata {
//     CF::Data out(cfdata);
//     return objc::bridge<NSData*>(out.GetCFObject());
// }

// - (CF::Data) cf {
//     return CF::Data(objc::bridge<CFDataRef>(self));
// }


@end

@implementation NSMutableData (AXMutableDataAdditions)

- (NSDataSink) dataSink {
    return NSDataSink(self);
}

@end
