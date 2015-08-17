/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSSTRING_PLUS_STL_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSSTRING_PLUS_STL_HH_

#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

#include <libimread/color.hh>

@interface NSColor (IMColorAdditions)
+ (NSColor *)               colorWithUniformRGBA:(const im::color::RGBA&)rgba;
+ (NSColor *)               colorWithUniformRGB:(const im::color::RGB&)rgb;
+ (NSColor *)               colorWithUniformMonochrome:(const im::color::Monochrome&)mono;
- (im::color::RGBA)         uniformRGBA;
- (im::color::RGB)          uniformRGB;
- (im::color::Monochrome)   uniformMonochrome;
@end


#endif /// LIBIMREAD_EXT_CATEGORIES_NSSTRING_PLUS_STL_HH_