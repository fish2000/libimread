/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSCOLOR_PLUS_IM_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSCOLOR_PLUS_IM_HH_

#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

#include <libimread/libimread.hpp>
#include <libimread/color.hh>

using namespace im;

@interface NSColor (IMColorAdditions)
+ (NSColor *)           colorWithUniformRGBA:(const color::RGBA&)rgba;
+ (NSColor *)           colorWithUniformRGB:(const color::RGB&)rgb;
+ (NSColor *)           colorWithUniformMonochrome:(const color::Monochrome&)mono;
- (color::RGBA)         uniformRGBA;
- (color::RGB)          uniformRGB;
- (color::Monochrome)   uniformMonochrome;
@end


#endif /// LIBIMREAD_EXT_CATEGORIES_NSCOLOR_PLUS_IM_HH_