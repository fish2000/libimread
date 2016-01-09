/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CLASSES_AXINTERLEAVEDIMAGEREP_HH_
#define LIBIMREAD_EXT_CLASSES_AXINTERLEAVEDIMAGEREP_HH_

#include <libimread/color.hh>
#include <libimread/interleaved.hh>

#import <libimread/ext/classes/AXCoreGraphicsImageRep.h>
#import <Foundation/Foundation.h>
#import <AppKit/NSGraphics.h>
#import <AppKit/NSGraphicsContext.h>

using im::color::RGB;
using im::color::RGBA;
using im::color::Monochrome;

using Meta = im::Meta<RGB>;
using Interleaved = im::InterleavedImage<RGB>;

template <typename Color = RGB>
__attribute__((cf_returns_not_retained))
CGImageRef CGImageFromInterleaved(
    const im::InterleavedImage<Color>& interleaved,
    __attribute__((cf_consumed))
        CGColorSpaceRef colorspace);

@interface AXInterleavedImageRep : AXCoreGraphicsImageRep {
    Interleaved interleavedImage;
}

+ (instancetype)        imageRepWithInterleaved:(const Interleaved&)interleaved;
+ (instancetype)        imageRepWithInterleaved:(const Interleaved&)interleaved
                                 colorSpaceName:(NSString*)space;
-                       initWithInterleaved:(const Interleaved&)interleaved;
-                       initWithInterleaved:(const Interleaved&)interleaved
                             colorSpaceName:(NSString*)space;
- (void)                setInterleaved:(const Interleaved&)interleaved;
- (Interleaved const&)  interleaved;
- (Meta const&)         imageMeta;

@end

#endif /// LIBIMREAD_EXT_CLASSES_AXINTERLEAVEDIMAGEREP_HH_