/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREGRAPHICS_H_
#define LIBIMREAD_COREGRAPHICS_H_

#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import <libimread/ext/UTI.h>
#endif

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>
#include <libimread/IO/apple.hh>


namespace im {
    
    /*
    CGImageRef CreateScaledImageAtFactor(CGImageRef sourceImage, CGFloat scaleFactor) {
        size_t sourceWidth = CGImageGetWidth(sourceImage);
        size_t sourceHeight = CGImageGetHeight(sourceImage);
        size_t scaledWidth = floorf(sourceWidth * scaleFactor);
        size_t scaledHeight = floorf(sourceHeight * scaleFactor);

        size_t bytesPerRow = scaledWidth * 4;
        if (bytesPerRow % 16) {
            bytesPerRow = ((bytesPerRow / 16) + 1) * 16;
        }

        CGContextRef bitmapContext = CGBitmapContextCreate(NULL, scaledWidth, scaledHeight, 8, bytesPerRow, CGImageGetColorSpace(sourceImage), kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Host);
        if (!bitmapContext) {
            NSLog(@"ERROR - failed to create bitmap context to rescale image");
            return NULL;
        }

        CGContextScaleCTM(bitmapContext, scaleFactor, scaleFactor);

        CGRect bounds = CGRectMake(0.0, 0.0, sourceWidth, sourceHeight);
        CGContextClearRect(bitmapContext, bounds);
        CGContextDrawImage(bitmapContext, bounds, sourceImage);

        CGImageRef scaledImage = CGBitmapContextCreateImage(bitmapContext);
        CGContextRelease(bitmapContext);

        return scaledImage;
    }
    
    CFDataRef CreateCompressedJPEGDataFromImage(CGImageRef image, CGFloat compressionFactor) {
        CFMutableDataRef imageData = CFDataCreateMutable(kCFAllocatorDefault, 0);
        CGImageDestinationRef destination = CGImageDestinationCreateWithData(imageData, kUTTypeJPEG, 1, NULL);
        if (!destination) {
            NSLog(@"ERROR - failed to create in-memory image destination");
            CFRelease(imageData);
            return NULL;
        }
        // set JPEG compression
        NSDictionary* properties = @{(id)kCGImageDestinationLossyCompressionQuality: @(compressionFactor)};
        CGImageDestinationAddImage(destination, image, (__bridge CFDictionaryRef)properties);
        BOOL status = CGImageDestinationFinalize(destination);
        if (!status) {
            NSLog(@"ERROR - failed to write scaled image to in-memory buffer");
            CFRelease(imageData);
            CFRelease(destination);
            return NULL;
        }
        CFRelease(destination);

        return (CFDataRef)imageData;
    }
    
    
    void saveJPG(std::vector<byte> data, ImageFactory *factory) {
        @autoreleasepool {
            @autoreleasepool {
                NSData *datum;
                
                @autoreleasepool {
                    datum = [[NSData alloc] initWithBytes:(const void *)&data[0]
                                                   length:(NSInteger)data.size()];
                    cf::REF<CGImageSourceRef> source(
                        CGImageSourceCreateWithData((__bridge CFDataRef)imageData, NULL));
                };
                [datum release];
            
            cf::REF<CGImageRef> image(GImageSourceCreateImageAtIndex(source, 0, NULL));
            if (!image.get()) {} /// throw something
            
            CGFloat scaleFactor = MAX(
                (CGFloat)outputWidth/CGImageGetWidth(image),
                (CGFloat)outputHeight/CGImageGetHeight(image));
            cf::REF<CGImageRef> scaled(CreateScaledImageAtFactor(image, scaleFactor));
            cf::REF<CFDataRef> compressed(CreateCompressedJPEGDataFromImage(scaled, compressionQuality));
            
            datum = (__bridge_transfer NSData*)compressed;
            //BOOL status = [datum writeToURL:outputFileURL atomically:YES];
        };
    }
    */
    
    namespace apple {
        
        static const options_map opts; /// NOT IN USE
        
        template <typename T>
        using ImageType = HybridImage<typename std::decay<T>::type>;
        template <typename T>
        using image_ptr = std::unique_ptr<ImageType<T>>;
        
        template <typename T = byte>
        image_ptr<T> read_unique(const std::string &filename) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            image_ptr<T> output = dynamic_cast_unique<ImageType<T>>(
                    format->read(input.get(), &factory, opts));
            return output;
        }
        
        template <typename T = byte>
        ImageType<T> read(const std::string &filename) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            ImageType<T> image(dynamic_cast<ImageType<T>&>(*output));
            return image;
        }
        
    }
    
}

#endif // LIBIMREAD_COREGRAPHICS_H_