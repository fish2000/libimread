/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#import <libimread/ext/classes/AXInterleavedImageRep.hh>
#import <libimread/objc-rt.hh>

#import <Foundation/Foundation.h>
#import <AppKit/NSGraphics.h>
#import <AppKit/NSGraphicsContext.h>

@implementation AXInterleavedImageRep

+ (void)initialize {
    OBJC_INITIALIZE;
    [self registerImageRepClass:self];
}

+ (instancetype) imageRepWithInterleaved:(const InterleavedImage&)interleaved {
    return [[AXInterleavedImageRep alloc] initWithInterleaved:interleaved];
}

+ (instancetype) imageRepWithInterleaved:(const InterleavedImage&)interleaved
                          colorSpaceName:(NSString *)space {
    return [[AXInterleavedImageRep alloc] initWithInterleaved:interleaved
                                               colorSpaceName:space];
}

CGImageRef CGImageFromInterleaved(const InterleavedImage& interleaved,
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

- initWithInterleaved:(const InterleavedImage&)interleaved {
    
    if (!(self = [super init])) { return nil; }
    interleavedImage = interleaved;
    
    CGImageRef imageref = CGImageFromInterleaved(
        interleaved,
        CGColorSpaceCreateDeviceRGB());
    
    NSString *space = objc::bridge<NSString*>(
        kCGColorSpaceGenericRGB);
    
    return [self initWithImageRef:imageref
                   colorSpaceName:space];
}

- initWithInterleaved:(const InterleavedImage&)interleaved
       colorSpaceName:(NSString *)space {
    
    if (!(self = [super init])) { return nil; }
    interleavedImage = interleaved;
    
    CGImageRef imageref = CGImageFromInterleaved(
        interleaved,
        CGColorSpaceCreateDeviceRGB());
    
    return [self initWithImageRef:imageref
                   colorSpaceName:space];
}

- (void) setInterleaved:(const InterleavedImage&)interleaved {
    interleavedImage = interleaved;
}

- (InterleavedImage) interleaved {
    return interleavedImage;
}

- (const Meta&) imageMeta {
    return const_cast<Meta&>(interleavedImage.getMeta());
}

@end

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
// - (BOOL)draw {
//     if (cgImage == NULL) { return NO; }
//     CGRect where = CGRectWithPointAndSize({0, 0},
//                                           CGImageGetWidth(cgImage),
//                                           CGImageGetHeight(cgImage));
//
//     CGContextDrawImage((CGContextRef)[[NSGraphicsContext currentContext] graphicsPort],
//                        where, cgImage);
//     return YES;
// }
//
// - (BOOL)drawAtPoint:(NSPoint)point {
//     if (cgImage == NULL) { return NO; }
//     CGRect where = CGRectWithPointAndSize(point,
//                                           CGImageGetWidth(cgImage),
//                                           CGImageGetHeight(cgImage));
//
//     CGContextDrawImage((CGContextRef)[[NSGraphicsContext currentContext] graphicsPort],
//                        where, cgImage);
//     return YES;
// }
//
// - (BOOL)drawInRect:(NSRect)rect {
//     if (cgImage == NULL) { return NO; }
//     CGRect where = CGRectWithRect(rect);
//
//     CGContextDrawImage((CGContextRef)[[NSGraphicsContext currentContext] graphicsPort],
//                        where, cgImage);
//     return YES;
// }
//
// - (BOOL)hasAlpha {
//     if (cgImage == NULL) { return NO; }
//     switch(CGImageGetAlphaInfo(cgImage)) {
//         case kCGImageAlphaNone:
//         case kCGImageAlphaNoneSkipLast:
//         case kCGImageAlphaNoneSkipFirst:
//             return NO;
//
//         case kCGImageAlphaPremultipliedLast:
//         case kCGImageAlphaPremultipliedFirst:
//         case kCGImageAlphaLast:
//         case kCGImageAlphaFirst:
//         case kCGImageAlphaOnly:
//         default:
//             return YES;
//     }
//     return NO;
// }
//
// - (NSInteger)pixelsHigh {
//     if (cgImage) { return CGImageGetHeight(cgImage); }
//     return 0;
// }
//
// - (NSInteger)pixelsWide {
//     if (cgImage) { return CGImageGetWidth(cgImage); }
//     return 0;
// }
