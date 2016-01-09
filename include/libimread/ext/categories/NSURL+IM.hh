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
#include <libimread/ext/categories/NSString+STL.hh>

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
        
        template <NSBitmapImageFileType nstype>
        char const* suffix_value = suffix_t<nstype>::str;
        
        inline std::string suffix(NSBitmapImageFileType nstype) {
            if (nstype == NSTIFFFileType)     { return suffix_t<NSTIFFFileType>::str;     }
            if (nstype == NSBMPFileType)      { return suffix_t<NSBMPFileType>::str;      }
            if (nstype == NSGIFFileType)      { return suffix_t<NSGIFFileType>::str;      }
            if (nstype == NSJPEGFileType)     { return suffix_t<NSJPEGFileType>::str;     }
            if (nstype == NSPNGFileType)      { return suffix_t<NSPNGFileType>::str;      }
            if (nstype == NSJPEG2000FileType) { return suffix_t<NSJPEG2000FileType>::str; }
            return "";
        }
        
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
+ (instancetype)            fileURLWithFilesystemPath:(filesystem::path const&)path;
-                           initFileURLWithFilesystemPath:(filesystem::path const&)path;
- (instancetype)            URLByAppendingSTLPathComponent:(std::string const&)component;
- (instancetype)            URLByAppendingFilesystemPath:(filesystem::path const&)path;
- (BOOL)                    openWithApplication:(NSString *)application;
- (BOOL)                    preview;
- (BOOL)                    isImage;
- (NSBitmapImageFileType)   imageFileType;
- (filesystem::path)        filesystemPath;
@end

#endif /// __OBJC__
#endif /// LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_