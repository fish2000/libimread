/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSBITMAPIMAGEREP_PLUS_IM_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSBITMAPIMAGEREP_PLUS_IM_HH_

#include <vector>
#include <memory>
#include <cstdio>
#include <cstring>
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

#include <libimread/libimread.hpp>
#include <libimread/image.hh>

using namespace im;

@interface NSBitmapImageRep (IMBitmapImageRepAdditions)
+ (instancetype)           imageRepWithByteVector:(const std::vector<byte>&)byteVector;
-                          initWithByteVector:(const std::vector<byte>&)byteVector;
- (std::unique_ptr<Image>) imageUsingImageFactory:(ImageFactory*)factory;
@end


#endif /// LIBIMREAD_EXT_CATEGORIES_NSBITMAPIMAGEREP_PLUS_IM_HH_