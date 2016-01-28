/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSURL+IM.hh>
#include <libimread/objc-rt/types.hh>

namespace objc {
    
    namespace image {
        
        DECLARE_CONSTEXPR_CHAR(suffix_t<NSTIFFFileType>::str,       "tiff");
        DECLARE_CONSTEXPR_CHAR(suffix_t<NSBMPFileType>::str,        "bmp");
        DECLARE_CONSTEXPR_CHAR(suffix_t<NSGIFFileType>::str,        "gif");
        DECLARE_CONSTEXPR_CHAR(suffix_t<NSJPEGFileType>::str,       "jpg");
        DECLARE_CONSTEXPR_CHAR(suffix_t<NSPNGFileType>::str,        "png");
        DECLARE_CONSTEXPR_CHAR(suffix_t<NSJPEG2000FileType>::str,   "jp2");
        
        std::string suffix(NSBitmapImageFileType nstype) {
            switch (nstype) {
                case NSTIFFFileType:        { return suffix_t<NSTIFFFileType>::str;     }
                case NSJPEGFileType:        { return suffix_t<NSJPEGFileType>::str;     }
                case NSPNGFileType:         { return suffix_t<NSPNGFileType>::str;      }
                case NSGIFFileType:         { return suffix_t<NSGIFFileType>::str;      }
                case NSBMPFileType:         { return suffix_t<NSBMPFileType>::str;      }
                case NSJPEG2000FileType:    { return suffix_t<NSJPEG2000FileType>::str; }
                default:                    { return "";                                }
            }
        }
        
        NSInteger filetype(std::string const& suffix) {
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
        
    }
}

#ifdef __OBJC__

@implementation NSURL (AXURLAdditions)

+ (instancetype) fileURLWithFilesystemPath:(filesystem::path const&)path {
    return [NSURL fileURLWithPath:[NSString
              stringWithSTLString:path.make_absolute().str()]
                      isDirectory:objc::boolean(path.is_directory())];
}

- initFileURLWithFilesystemPath:(filesystem::path const&)path {
    return [self initFileURLWithPath:[NSString
                 stringWithSTLString:path.make_absolute().str()]
                         isDirectory:objc::boolean(path.is_directory())];
}

- (instancetype) URLByAppendingSTLPathComponent:(std::string const&)component {
    return [self URLByAppendingPathComponent:[NSString stringWithSTLString:component]];
}

- (instancetype) URLByAppendingFilesystemPath:(filesystem::path const&)path {
    NSURL* url = [self copy];
    for (auto const& component : path.components()) {
        url = [url URLByAppendingSTLPathComponent:component];
    }
    return url;
}

- (BOOL) openWithApplication:(NSString *)application {
    NSString* filePath = [[NSString alloc] initWithUTF8String:[self fileSystemRepresentation]];
    return [[NSWorkspace sharedWorkspace] openFile:filePath
                                   withApplication:application];
}

- (BOOL) preview {
    return [self openWithApplication:@"Preview.app"];
}

- (BOOL) isImage {
    return objc::boolean(
        objc::image::filetype(
            [[self.pathExtension lowercaseString] STLString]) != -1);
}

- (NSBitmapImageFileType) imageFileType {
    return static_cast<NSBitmapImageFileType>(
        objc::image::filetype(
            [[self.pathExtension lowercaseString] STLString]));
}

- (filesystem::path) filesystemPath {
    filesystem::path out([self fileSystemRepresentation]);
    return out;
}

- (std::string) STLString {
    return [self fileSystemRepresentation];
}

@end

#endif /// __OBJC__
