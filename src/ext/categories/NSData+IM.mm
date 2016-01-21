/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <algorithm>
#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSData+IM.hh>
#include <libimread/ext/categories/NSString+STL.hh>

using im::byte;
using im::byte_source;
using im::byte_sink;
using im::NSDataSource;
using im::NSDataSink;


@implementation NSData (AXDataAdditions)

+ (instancetype) dataWithByteVector:(std::vector<byte> const&)byteVector {
    return [[NSData alloc] initWithBytes:(const void*)&byteVector[0]
                                  length:(NSInteger)byteVector.size()];
}

+ (instancetype) dataWithSTLString:(std::string const&)string {
    return [[NSData alloc] initWithBytes:(const void*)string.data()
                                  length:(NSInteger)string.size()];
}

+ (instancetype) dataWithByteSource:(byte_source*)byteSource {
    return [NSData dataWithByteVector:byteSource->full_data()];
}

+ (instancetype) dataWithByteSource:(byte_source*)byteSource
                             length:(NSUInteger)bytes {
    std::unique_ptr<byte[]> buffer = std::make_unique<byte[]>(bytes);
    __attribute__((unused))
    int idx = byteSource->read(buffer.get(),
                               static_cast<std::size_t>(bytes));
    return [[NSData alloc] initWithBytes:(const void*)buffer.get()
                                  length:(NSInteger)bytes];
}
- initWithByteVector:(std::vector<byte> const&)byteVector {
    return [self initWithBytes:(const void*)&byteVector[0]
                        length:(NSInteger)byteVector.size()];
}

- initWithSTLString:(std::string const&)string {
    return [self initWithBytes:(const void*)string.data()
                        length:(NSInteger)string.size()];
}

- initWithByteSource:(byte_source*)byteSource {
    return [self initWithByteVector:byteSource->full_data()];
}

- initWithByteSource:(byte_source*)byteSource
              length:(NSUInteger)bytes {
    std::unique_ptr<byte[]> buffer = std::make_unique<byte[]>(bytes);
    __attribute__((unused))
    int idx = byteSource->read(buffer.get(),
                               static_cast<std::size_t>(bytes));
    return [self initWithBytes:(const void*)buffer.get()
                        length:(NSInteger)bytes];
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

- (std::unique_ptr<NSDataSource>) dataSource {
    return std::make_unique<NSDataSource>(self);
}

- (std::string) STLString {
    return std::string(static_cast<const char*>(self.bytes),
                       static_cast<std::size_t>(self.length));
}

@end

@implementation NSMutableData (AXMutableDataAdditions)

- (std::unique_ptr<NSDataSink>) dataSink {
    return std::make_unique<NSDataSink>(self);
}

@end
