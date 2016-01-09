/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/ext/classes/AXInterleavedImageRep.hh>
#include <libimread/objc-rt/types.hh>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <AppKit/NSGraphics.h>
#import <AppKit/NSGraphicsContext.h>


template <>
__attribute__((cf_returns_not_retained))
CGImageRef CGImageFromInterleaved<RGB>(
    const im::InterleavedImage<RGB>& interleaved,
    __attribute__((cf_consumed))
        CGColorSpaceRef colorspace) {
    CGContextRef context = CGBitmapContextCreate(
        interleaved.data(),
        interleaved.width(),
        interleaved.height(),
        interleaved.nbits(), /// bits per component
        interleaved.width(), /// bytes per row
        colorspace, kCGImageAlphaNone);
    
    CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
    CGContextSetShouldAntialias(context, NO);
    CGImageRef imageref = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    CGColorSpaceRelease(colorspace);
    
    return imageref;
}

template <>
__attribute__((cf_returns_not_retained))
CGImageRef CGImageFromInterleaved<RGBA>(
    const im::InterleavedImage<RGBA>& interleaved,
    __attribute__((cf_consumed))
        CGColorSpaceRef colorspace) {
    CGContextRef context = CGBitmapContextCreate(
        interleaved.data(),
        interleaved.width(),
        interleaved.height(),
        interleaved.nbits(), /// bits per component
        interleaved.width(), /// bytes per row
        colorspace, kCGImageAlphaLast);
    
    CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
    CGContextSetShouldAntialias(context, NO);
    CGImageRef imageref = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    CGColorSpaceRelease(colorspace);
    
    return imageref;
}

template <>
__attribute__((cf_returns_not_retained))
CGImageRef CGImageFromInterleaved<Monochrome>(
    const im::InterleavedImage<Monochrome>& interleaved,
    __attribute__((cf_consumed))
        CGColorSpaceRef colorspace) {
    CGContextRef context = CGBitmapContextCreate(
        interleaved.data(),
        interleaved.width(),
        interleaved.height(),
        interleaved.nbits(), /// bits per component
        interleaved.width(), /// bytes per row
        colorspace, kCGImageAlphaNone);
    
    CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
    CGContextSetShouldAntialias(context, NO);
    CGImageRef imageref = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    CGColorSpaceRelease(colorspace);
    
    return imageref;
}

@implementation AXInterleavedImageRep : AXCoreGraphicsImageRep

+ (void) initialize {
    OBJC_INITIALIZE;
    [self registerImageRepClass:self];
}

+ (instancetype) imageRepWithInterleaved:(const Interleaved&)interleaved {
    return [[AXInterleavedImageRep alloc] initWithInterleaved:interleaved];
}

+ (instancetype) imageRepWithInterleaved:(const Interleaved&)interleaved
                          colorSpaceName:(NSString*)space {
    return [[AXInterleavedImageRep alloc] initWithInterleaved:interleaved
                                               colorSpaceName:space];
}

- initWithInterleaved:(const Interleaved&)interleaved {
    
    if (!(self = [super init])) { return nil; }
    interleavedImage = interleaved;
    
    CGImageRef imageref = CGImageFromInterleaved(
        interleaved,
        CGColorSpaceCreateDeviceRGB());
    
    NSString* space = objc::bridge<NSString*>(
        kCGColorSpaceGenericRGB);
    
    return [self initWithImageRef:imageref
                   colorSpaceName:space];
}

- initWithInterleaved:(const Interleaved&)interleaved
       colorSpaceName:(NSString*)space {
    
    if (!(self = [super init])) { return nil; }
    interleavedImage = interleaved;
    
    CGImageRef imageref = CGImageFromInterleaved(
        interleaved,
        CGColorSpaceCreateDeviceRGB());
    
    return [self initWithImageRef:imageref
                   colorSpaceName:space];
}

- (void) setInterleaved:(const Interleaved&)interleaved {
    interleavedImage = interleaved;
    [self setImage:CGImageFromInterleaved(interleaved, CGColorSpaceCreateDeviceRGB())];
    [self setColorSpaceName:objc::bridge<NSString*>(kCGColorSpaceGenericRGB)];
}

- (Interleaved const&) interleaved {
    return interleavedImage;
}

- (Meta const&) imageMeta {
    return interleavedImage.getMeta();
}

@end

#endif /// __OBJC__

// /// Init and dealloc
// - initWithImageRef:(CGImageRef)myImage colorSpaceName:(NSString *)space {
//     if (!(self = [super init])) { return nil; }
//
//     cgImage = myImage;
//     CGImageRetain(cgImage);
//     colorSpaceName = [space copy];
//
//     return self;
// }
//
// - (void)dealloc {
//     if (cgImage != NULL) { CGImageRelease(cgImage); }
//
//     [colorSpaceName release];
//     [heldObject release];
//     [super dealloc];
// }
//
// /// API
// - (void)setImage:(CGImageRef)newImage {
//     if (cgImage != newImage) {
//         if (cgImage != NULL) { CGImageRelease(cgImage); }
//         cgImage = CGImageRetain(newImage);
//     }
// }
//
// - (void)setColorSpaceHolder:(id<NSObject>)anObject {
//     /* The reason for this is a little obscure. We never actually use the color space object
//        (an OIICCProfile instance). It's mainly just a wrapper around a CGColorSpaceRef, and
//        the CGImage holds on to that by itself. However, if we keep the OIICCProfile from
//        being deallocated, it will maintain a map table entry which allows image processors
//        to use the same CGColorSpace for identical color profiles read from different images.
//        Is this actually a performance gain? I have no idea. It seems like it ought to be, though.
//     */
//     [heldObject autorelease];
//     heldObject = [anObject retain];
// }
//
// /// NSImageRep attributes
//
// - (NSInteger)bitsPerSample {
//     if (cgImage) {
//         return CGImageGetBitsPerComponent(cgImage);
//     }
//     return 0;
// }
//
// - (NSString *)colorSpaceName {
//     return colorSpaceName;
// }
//
