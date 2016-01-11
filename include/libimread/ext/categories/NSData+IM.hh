/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSDATA_PLUS_CFPP_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSDATA_PLUS_CFPP_HH_

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#endif /// __OBJC__

#include <cstdlib>
#include <memory>
#include <vector>
#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>
#include <libimread/ext/categories/NSString+STL.hh>

namespace im {
    
    class NSDataSource : public byte_source {
        public:
            NSDataSource(NSData* d)
                :data(d), pos(0)
                {
                    #if !__has_feature(objc_arc)
                        [data retain];
                    #endif
                }
            NSDataSource(NSMutableData* d)
                :data((NSData*)d), pos(0)
                {
                    #if !__has_feature(objc_arc)
                        [data retain];
                    #endif
                }
            
            virtual ~NSDataSource() {
                #if !__has_feature(objc_arc)
                    [data release];
                #endif
            }
            
            virtual std::size_t read(byte* buffer, std::size_t n) {
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
            NSData* data;
            std::size_t pos;
    };
    
    class NSDataSink : public byte_sink {
        public:
            NSDataSink(NSMutableData* d)
                :data(d), pos(0)
                {
                    #if !__has_feature(objc_arc)
                        [data retain];
                    #endif
                }
            
            virtual ~NSDataSink() {
                #if !__has_feature(objc_arc)
                    [data release];
                #endif
            }
            
            virtual bool can_seek() const noexcept { return true; }
            virtual std::size_t seek_absolute(std::size_t p) { return pos = p; }
            virtual std::size_t seek_relative(int delta) { return pos += delta; }
            virtual std::size_t seek_end(int delta) { return pos = (data.length-delta-1); }
            
            virtual std::size_t write(const void* buffer, std::size_t n) {
                seek_end(0);
                [data appendBytes:buffer length:n];
                return n;
            }
            
            virtual std::vector<byte> contents() {
                std::vector<byte> out(data.length);
                std::memcpy(&out[0], (byte*)data.bytes, out.size());
                return out;
            }
            
        private:
            NSMutableData* data;
            std::size_t pos;
    };

}

using im::byte;
using im::byte_source;
using im::byte_sink;
using im::NSDataSource;
using im::NSDataSink;

#ifdef __OBJC__

@interface NSData (AXDataAdditions)
+ (instancetype)        dataWithByteVector:(std::vector<byte> const&)byteVector;
+ (instancetype)        dataWithByteSource:(byte_source*)byteSource;
+ (instancetype)        dataWithByteSource:(byte_source*)byteSource
                                    length:(NSUInteger)bytes;
- (NSUInteger)          writeUsingByteSink:(byte_sink*)byteSink;
- (NSUInteger)          writeUsingByteSink:(byte_sink*)byteSink
                                    length:(NSUInteger)bytes;
- (std::unique_ptr<NSDataSource>) dataSource;
@end

@interface NSMutableData (AXMutableDataAdditions)
- (std::unique_ptr<NSDataSink>) dataSink;
@end

#endif /// __OBJC__
#endif /// LIBIMREAD_EXT_CATEGORIES_NSDATA_PLUS_CFPP_HH_