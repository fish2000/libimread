/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSBITMAPIMAGEREP_PLUS_IM_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSBITMAPIMAGEREP_PLUS_IM_HH_

#include <vector>
#include <memory>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#endif /// __OBJC__

#include <libimread/libimread.hpp>
#include <libimread/image.hh>

using im::byte;
using im::Image;
using im::ImageFactory;

#ifdef __OBJC__

@interface NSBitmapImageRep (AXBitmapImageRepAdditions)
+ (NSBitmapImageRep *)     imageRepWithByteVector:(std::vector<byte> const&)byteVector;
-                          initWithByteVector:(std::vector<byte> const&)byteVector;
- (std::unique_ptr<Image>) imageUsingImageFactory:(ImageFactory*)factory;
@end

#endif /// __OBJC__
#endif /// LIBIMREAD_EXT_CATEGORIES_NSBITMAPIMAGEREP_PLUS_IM_HH_