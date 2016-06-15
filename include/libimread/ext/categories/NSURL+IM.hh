/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <Cocoa/Cocoa.h>
#endif /// __OBJC__

#include <string>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>

static constexpr NSBitmapImageFileType AXPVRFileType = static_cast<NSBitmapImageFileType>(444);

namespace objc {
    
    namespace image {
        
        template <NSBitmapImageFileType t>
        struct suffix_t;
        
        #define DEFINE_SUFFIX(endstring, nstype)                                        \
        template <>                                                                     \
        struct suffix_t<nstype> {                                                       \
            using NSImageType = NSBitmapImageFileType;                                  \
            static constexpr std::size_t N = im::static_strlen(endstring);              \
            static constexpr char const str[N] = endstring;                             \
            static constexpr NSImageType type = nstype;                                 \
        };
        
        DEFINE_SUFFIX("tiff", NSTIFFFileType);
        DEFINE_SUFFIX("bmp",  NSBMPFileType);
        DEFINE_SUFFIX("gif",  NSGIFFileType);
        DEFINE_SUFFIX("jpg",  NSJPEGFileType);
        DEFINE_SUFFIX("png",  NSPNGFileType);
        DEFINE_SUFFIX("jp2",  NSJPEG2000FileType);
        DEFINE_SUFFIX("pvr",  AXPVRFileType);
        
        template <NSBitmapImageFileType nstype>
        char const* suffix_value = suffix_t<nstype>::str;
        
        std::string suffix(NSBitmapImageFileType nstype);
        NSInteger filetype(std::string const& suffix);
        
    };
    
};

#ifdef __OBJC__

@interface NSURL (AXURLAdditions)
+ (instancetype)            fileURLWithFilesystemPath:(filesystem::path const&)path;
-                           initFileURLWithFilesystemPath:(filesystem::path const&)path;
- (instancetype)            URLByAppendingSTLPathComponent:(std::string const&)component;
- (instancetype)            URLByAppendingFilesystemPath:(filesystem::path const&)path;
- (BOOL)                    openWithApplication:(NSString *)application;
- (BOOL)                    preview;
- (BOOL)                    isImage;
- (NSBitmapImageFileType)   imageFileType;
- (filesystem::path)        filesystemPath;
- (std::string)             STLString;
@end

#endif /// __OBJC__
#endif /// LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_