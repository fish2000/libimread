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
using Meta = im::Meta<RGB, 3>;
using InterleavedImage = im::InterleavedImage<RGB, 3>;

@interface AXInterleavedImageRep : AXCoreGraphicsImageRep {
    InterleavedImage interleavedImage;
}

+ (instancetype)        imageRepWithInterleaved:(const InterleavedImage&)interleaved;
+ (instancetype)        imageRepWithInterleaved:(const InterleavedImage&)interleaved colorSpaceName:(NSString *)space;
-                       initWithInterleaved:(const InterleavedImage&)interleaved;
-                       initWithInterleaved:(const InterleavedImage&)interleaved colorSpaceName:(NSString *)space;
- (void)                setInterleaved:(const InterleavedImage&)interleaved;
- (InterleavedImage)    interleaved;
- (const Meta&)         imageMeta;

@end

#endif /// LIBIMREAD_EXT_CLASSES_AXINTERLEAVEDIMAGEREP_HH_