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
    }
}

#ifdef __OBJC__

@implementation NSURL (IMURLAdditions)

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

@end

#endif /// __OBJC__
