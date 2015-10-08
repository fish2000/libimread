/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#endif /// __OBJC__

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/categories/NSString+STL.hh>
#include <libimread/objc-rt.hh>

#ifdef __OBJC__

// using namespace filesystem;

@interface NSURL (IMURLAdditions)
+ (instancetype)        fileURLWithFilesystemPath:(const filesystem::path&)path;
-                       initFileURLWithFilesystemPath:(const filesystem::path&)path;
- (instancetype)        URLByAppendingFilesystemPath:(const filesystem::path&)path;
- (filesystem::path)    filesystemPath;
@end

#endif /// __OBJC__
#endif /// LIBIMREAD_EXT_CATEGORIES_NSURL_PLUS_IM_HH_