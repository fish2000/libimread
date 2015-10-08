/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSURL+IM.hh>
#include <libimread/ext/categories/NSString+STL.hh>
#include <libimread/objc-rt.hh>

@implementation NSURL (IMURLAdditions)

+ (instancetype) fileURLWithFilesystemPath:(const filesystem::path&)path {
    return [NSURL fileURLWithPath:[NSString
              stringWithSTLString:path.make_absolute().str()]
                      isDirectory:objc::boolean(path.is_directory())];
}

- initFileURLWithFilesystemPath:(const filesystem::path&)path {
    return [self initFileURLWithPath:[NSString
                 stringWithSTLString:path.make_absolute().str()]
                         isDirectory:objc::boolean(path.is_directory())];
}

- (instancetype) URLByAppendingSTLPathComponent:(const std::string&)component {
    return [self URLByAppendingPathComponent:[NSString stringWithSTLString:component]];
}

- (instancetype) URLByAppendingFilesystemPath:(const filesystem::path&)path {
    NSURL *url = [self copy];
    for (auto component : path.components()) {
        url = [url URLByAppendingSTLPathComponent:component];
    }
    return url;
}

- (BOOL) openWithApplication:(NSString *)application {
    NSString *filePath = [[NSString alloc] initWithUTF8String:[self fileSystemRepresentation]];
    return [[NSWorkspace sharedWorkspace] openFile:filePath
                                   withApplication:application];
}

- (BOOL) preview {
    return [self openWithApplication:@"Preview.app"];
}

- (filesystem::path) filesystemPath {
    filesystem::path out([self fileSystemRepresentation]);
    return std::move(out);
}

@end
