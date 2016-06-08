/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#include <iod/json.hh>

#include <libimread/libimread.hpp>
#include <libimread/IO/apple.hh>
#include <libimread/ext/categories/NSBitmapImageRep+IM.hh>
#include <libimread/ext/categories/NSData+IM.hh>
#include <libimread/ext/categories/NSURL+IM.hh>

namespace im {
    
    DECLARE_FORMAT_OPTIONS(NSImageFormat);
    
    namespace detail {
        
        __attribute__((ns_returns_retained))
        NSDictionary* translate_options(options_map const& opts) {
            NSMutableDictionary* out = [NSMutableDictionary dictionaryWithDictionary:@{
                NSImageColorSyncProfileData     : [NSNull null],    /// NSData containing ICC profile data
                NSImageEXIFData                 : @{},              /// JPEG EXIF NSDictionary
                NSImageProgressive              : @1,               /// JPEG progressive-format boolean
                NSImageCompressionFactor        : @0.75,            /// JPEG compression
                NSImageCompressionMethod        : @5,               /// TIFF compression (5 = LZW)
                NSImageRGBColorTable            : [NSNull null],    /// NSData containing GIF packed-RGB color table
                NSImageDitherTransparency       : @0,               /// GIF dither boolean (0 = off, 1 = on)
                NSImageInterlaced               : @1,               /// PNG interlacing boolean
                NSImageGamma                    : @1.0,             /// PNG “gamma” value
                NSImageFallbackBackgroundColor  : [NSNull null],    /// NSColor transparency value for non-alpha formats
                NSImageFrameCount               : @0,               /// Animated GIF frame count
                NSImageCurrentFrame             : @0,               /// Animated GIF current frame
                NSImageCurrentFrameDuration     : @0,               /// Animated GIF current frame duration in seconds
                NSImageLoopCount                : @0,               /// Animated GIF loop count
            }];
            
            [out setObject:[NSNumber numberWithInt:opts.cast<bool>("jpeg:progressive", true) ? 1 : 0]
                    forKey:NSImageProgressive];
            [out setObject:[NSNumber numberWithFloat:opts.cast<float>("jpeg:quality", 0.75)]
                    forKey:NSImageCompressionFactor];
            [out setObject:[NSNumber numberWithInt:opts.cast<bool>("gif:dither", false) ? 1 : 0]
                    forKey:NSImageDitherTransparency];
            [out setObject:[NSNumber numberWithInt:opts.cast<bool>("png:interlace", true) ? 1 : 0]
                    forKey:NSImageInterlaced];
            [out setObject:[NSNumber numberWithFloat:opts.cast<float>("png:gamma", 1.0)]
                    forKey:NSImageGamma];
            
            return [NSDictionary dictionaryWithDictionary:out];
        }
        
    }
    
    std::unique_ptr<Image> NSImageFormat::read(byte_source* src,
                                               ImageFactory* factory,
                                               options_map const& opts)  {
        @autoreleasepool {
            NSBitmapImageRep* rep = [NSBitmapImageRep imageRepWithByteVector:src->full_data()];
            return [rep imageUsingImageFactory:factory];
        }
    }
    
    void NSImageFormat::write(Image& input, byte_sink* output,
                                            options_map const& opts) {
        NSInteger filetype = objc::image::filetype(opts.get("extension"));
        imread_assert(filetype != -1,
                      "[apple-io] Can't write image of unknown type:",
                      opts.get("filename"));
        
        @autoreleasepool {
            NSBitmapImageRep* rep = [[NSBitmapImageRep alloc] initWithImage:input];
            NSDictionary* props = detail::translate_options(opts);
            NSData* datum = [rep representationUsingType:static_cast<NSBitmapImageFileType>(filetype)
                                              properties:props];
            [datum writeUsingByteSink:output];
        }
    }
    
}