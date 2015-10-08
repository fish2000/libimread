/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSURL+IM.hh>
#include <libimread/ext/categories/NSString+STL.hh>
#include <libimread/objc-rt.hh>

@implementation NSURL (IMURLAdditions)

+ (instancetype) fileURLWithFilesystemPath:(const filesystem::path&)path {
    return [NSURL fileURLWithPath:[NSString
              stringWithSTLString:path.str()]
                      isDirectory:objc::boolean(path.is_directory())];
}

- initFileURLWithFilesystemPath:(const filesystem::path&)path {
    return [self initFileURLWithPath:[NSString
                 stringWithSTLString:path.str()]
                         isDirectory:objc::boolean(path.is_directory())];
}

- (instancetype) URLByAppendingFilesystemPath:(const filesystem::path&)path {
    NSURL *url = [self copy];
    for (auto component : path.components()) {
        url = [url URLByAppendingPathComponent:[NSString
                           stringWithSTLString:component]];
    }
    return url;
}

- (filesystem::path) filesystemPath {
    filesystem::path out([self fileSystemRepresentation]);
    return std::move(out);
}

@end
