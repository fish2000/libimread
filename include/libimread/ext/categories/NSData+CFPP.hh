/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSDATA_PLUS_CFPP_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSDATA_PLUS_CFPP_HH_

#include <vector>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#endif /// __OBJC__

#include <CF++.hpp>
#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>


namespace im {
    
    class NSDataSource : public byte_source {
        public:
            NSDataSource(NSData *d)
                :data(d), pos(0)
                {
                    [data retain];
                }
            
            virtual ~NSDataSource() { [data release]; }
            
            virtual std::size_t read(byte *buffer, std::size_t n) {
                if (pos + n > data.length) { n = data.length-pos; }
                std::memmove(buffer, (byte *)data.bytes + pos, n);
                pos += n;
                return n;
            }
            
            virtual bool can_seek() const noexcept { return true; }
            virtual std::size_t seek_absolute(std::size_t p) { return pos = p; }
            virtual std::size_t seek_relative(int delta) { return pos += delta; }
            virtual std::size_t seek_end(int delta) { return pos = (data.length-delta-1); }
        
        private:
            NSData *data;
            std::size_t pos;
    };
    
    class NSDataSink : public byte_sink {
        public:
            NSDataSink(NSMutableData *d)
                :data(d), pos(0)
                {
                    [data retain];
                }
            
            virtual ~NSDataSink() { [data release]; }
            
            virtual bool can_seek() const noexcept { return true; }
            virtual std::size_t seek_absolute(std::size_t p) { return pos = p; }
            virtual std::size_t seek_relative(int delta) { return pos += delta; }
            virtual std::size_t seek_end(int delta) { return pos = (data.length-delta-1); }
            
            virtual std::size_t write(const void *buffer, std::size_t n) {
                seek_end(0);
                [data appendBytes:buffer length:n];
                return n;
            }
            
            virtual std::vector<byte> contents() {
                std::vector<byte> out(data.length);
                std::memcpy(&out[0], (byte *)data.bytes, out.size());
                return out;
            }
            
        private:
            NSMutableData *data;
            std::size_t pos;
    };

}

using namespace im;

#ifdef __OBJC__

@interface NSData (IMDataAdditions)
+ (NSData *)            dataWithByteVector:(const std::vector<byte>&)byteVector;
+ (NSData *)            dataWithByteSource:(byte_source*)byteSource;
+ (NSData *)            dataWithByteSource:(byte_source*)byteSource
                                    length:(NSUInteger)bytes;
+ (NSData *)            dataWithCFData:(const CF::Data&)cfdata;
- (CF::Data)            cf;
- (NSDataSource)        dataSource;
- (NSUInteger)          writeUsingByteSink:(byte_sink*)byteSink;
- (NSUInteger)          writeUsingByteSink:(byte_sink*)byteSink
                                    length:(NSUInteger)bytes;
@end

@interface NSMutableData (IMMutableDataAdditions)
- (NSDataSink)          dataSink;
@end

#endif /// __OBJC__
#endif /// LIBIMREAD_EXT_CATEGORIES_NSDATA_PLUS_CFPP_HH_