/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_

#include <string>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <Cocoa/Cocoa.h>
#endif /// __OBJC__

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/categories/NSString+STL.hh>
#include <libimread/objc-rt/objc-rt.hh>

namespace objc {
    
    namespace image {
        
        template <NSBitmapImageFileType t>
        struct suffix;
        
        #define DEFINE_SUFFIX(endstring, nstype)                                \
        template <>                                                             \
        struct suffix<nstype> {                                                 \
            using NSImageType = NSBitmapImageFileType;                          \
            static constexpr char const* str = endstring;                       \
            static constexpr NSImageType type = nstype;                         \
        };
        
        DEFINE_SUFFIX("tiff", NSTIFFFileType);
        DEFINE_SUFFIX("bmp",  NSBMPFileType);
        DEFINE_SUFFIX("gif",  NSGIFFileType);
        DEFINE_SUFFIX("jpg",  NSJPEGFileType);
        DEFINE_SUFFIX("png",  NSPNGFileType);
        DEFINE_SUFFIX("jp2",  NSJPEG2000FileType);
        
        inline NSInteger filetype(std::string const& suffix) {
            if (suffix == "tiff" || suffix == ".tiff" ||
                suffix == "tif"  || suffix == ".tif") {
                return static_cast<NSInteger>(NSTIFFFileType);
            } else if (suffix == "bmp" || suffix == ".bmp") {
                return static_cast<NSInteger>(NSBMPFileType);
            } else if (suffix == "gif" || suffix == ".gif") {
                return static_cast<NSInteger>(NSGIFFileType);
            } else if (suffix == "jpg"  || suffix == ".jpg" ||
                       suffix == "jpeg" || suffix == ".jpeg") {
                return static_cast<NSInteger>(NSJPEGFileType);
            } else if (suffix == "png" || suffix == ".png") {
                return static_cast<NSInteger>(NSPNGFileType);
            } else if (suffix == "jp2" || suffix == ".jp2") {
                return static_cast<NSInteger>(NSJPEG2000FileType);
            } else {
                /// NO MATCH
                return -1;
            }
        }
        
    };
    
};

#ifdef __OBJC__

@interface NSURL (IMURLAdditions)
+ (instancetype)            fileURLWithFilesystemPath:(const filesystem::path&)path;
-                           initFileURLWithFilesystemPath:(const filesystem::path&)path;
- (instancetype)            URLByAppendingSTLPathComponent:(const std::string&)component;
- (instancetype)            URLByAppendingFilesystemPath:(const filesystem::path&)path;
- (BOOL)                    openWithApplication:(NSString *)application;
- (BOOL)                    preview;
- (BOOL)                    isImage;
- (NSBitmapImageFileType)   imageFileType;
- (filesystem::path)        filesystemPath;
@end

#endif /// __OBJC__
#endif /// LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_