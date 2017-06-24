//
//  main.mm
//  depixelate
//
//  Created by Karl Stenerud on 1/2/14.
//  Copyright (c) 2014 Karl Stenerud. All rights reserved.
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <type_traits>

#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

#import "xbrz/xbrz.h"
#import "hqx/hqx.h"
#import "scale2x/scalebit.h"

using scale_f = std::add_pointer_t<bool(const void* srcData,
                                              void* dstData,
                                              int width, int height, int scale)>;

typedef struct {
    const char*   name;
    const int     minScale;
    const int     maxScale;
    scale_f       scaler;
} Algorithm;

static bool scale_xbrz(const void* srcData, void* dstData, int width, int height, int scale) {
    xbrz::scale(scale, static_cast<uint32_t*>(const_cast<void*>(srcData)),
                       static_cast<uint32_t*>(dstData),
                       width, height);
    return true;
}

static bool scale_hqx(const void* srcData, void* dstData, int width, int height, int scale) {
    hqxInit();
    switch (scale) {
        case 2: hq2x_32(static_cast<uint32_t*>(const_cast<void*>(srcData)),
                        static_cast<uint32_t*>(dstData),
                        width, height);
        case 3: hq3x_32(static_cast<uint32_t*>(const_cast<void*>(srcData)),
                        static_cast<uint32_t*>(dstData),
                        width, height);
        case 4: hq4x_32(static_cast<uint32_t*>(const_cast<void*>(srcData)),
                        static_cast<uint32_t*>(dstData),
                        width, height);
    }
    return true;
}

static bool scale_scale2x(const void* srcData, void* dstData, int width, int height, int scaleVal) {
    scale2x_scale(scaleVal,
                  dstData, 4 * width * scaleVal,
                  srcData, 4 * width,
                  4, width, height);
    return true;
}

static const std::array<Algorithm, 3> algorithms{{
    { "xbrz",    2, 5,  scale_xbrz    },
    { "hqx",     2, 4,  scale_hqx     },
    { "scale2x", 2, 4,  scale_scale2x }
}};

Algorithm const* getAlgorithm(const char* name) {
    for (int i = 0; i < algorithms.size(); ++i) {
        Algorithm const* algorithm = &algorithms[i];
        if (std::strcmp(name, algorithm->name) == 0) { return algorithm; }
    }
    return nullptr;
}

int depixelate_main(int argc, const char* argv[]) {
    
    @autoreleasepool {
        
        if (argc != 5) {
            std::printf("Usage: %s <algorithm> <scale> <srcfile> <dstfile>\n", argv[0]);
            std::printf("Algorithms:\n");
            for (int i = 0; i < algorithms.size(); ++i) {
                Algorithm const* algorithm = &algorithms[i];
                std::printf("- %s\n", algorithm->name);
            }
            return EXIT_FAILURE;
        }
        
        const char* algoName = argv[1];
        int         scale    = std::atoi(argv[2]);
        NSString*   srcFile  = [NSString stringWithUTF8String:argv[3]];
        NSString*   dstFile  = [NSString stringWithUTF8String:argv[4]];
        
        Algorithm const* algorithm = getAlgorithm(algoName);
        
        if (algorithm == nullptr) {
            std::printf("%s: Unknown algorithm\n", algoName);
            return EXIT_FAILURE;
        }
        
        if (scale < algorithm->minScale || scale > algorithm->maxScale) {
            std::printf("Scale must be from %d to %d\n", algorithm->minScale, algorithm->maxScale);
            return EXIT_FAILURE;
        }
        
        NSImage* image = [[NSImage alloc] initWithContentsOfFile:srcFile];
        if (image == nil) {
            std::printf("%s: Unable to open file\n", srcFile.UTF8String);
            return EXIT_FAILURE;
        }
        
        NSBitmapImageRep* rep    = static_cast<NSBitmapImageRep*>([[image representations] objectAtIndex:0]);
        const void* srcData      = static_cast<void*>(rep.bitmapData);
        std::size_t dstByteCount = rep.pixelsWide * rep.pixelsHigh * rep.bitsPerPixel * scale;
        void* dstData            = std::malloc(dstByteCount);
        
        if (!algorithm->scaler(srcData, dstData,
                               static_cast<int>(rep.pixelsWide),
                               static_cast<int>(rep.pixelsHigh),
                               scale)) {
            std::printf("Scaling failed!\n");
            std::free(dstData);
            return EXIT_FAILURE;
        }
        
        NSBitmapImageRep* dstRep = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:(unsigned char**)&dstData
                                                                           pixelsWide:rep.pixelsWide * scale
                                                                           pixelsHigh:rep.pixelsHigh * scale
                                                                        bitsPerSample:rep.bitsPerSample
                                                                      samplesPerPixel:rep.samplesPerPixel
                                                                             hasAlpha:rep.hasAlpha
                                                                             isPlanar:rep.isPlanar
                                                                       colorSpaceName:rep.colorSpaceName
                                                                         bitmapFormat:rep.bitmapFormat
                                                                          bytesPerRow:rep.bytesPerRow * scale
                                                                         bitsPerPixel:rep.bitsPerPixel];
        
        NSData* finalData = [dstRep representationUsingType:NSPNGFileType
                                                 properties:@{}];
        
        [finalData writeToFile:dstFile
                    atomically:YES];
        std::free(dstData);
    
    } /* @autoreleasepool */
    
    return EXIT_SUCCESS;
}
