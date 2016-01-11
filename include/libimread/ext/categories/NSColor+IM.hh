/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSCOLOR_PLUS_IM_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSCOLOR_PLUS_IM_HH_

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#endif /// __OBJC__

#include <libimread/libimread.hpp>
#include <libimread/color.hh>

using namespace im;

@interface NSColor (AXColorAdditions)
+ (instancetype)        colorWithUniformRGBA:(color::RGBA const&)rgba;
+ (instancetype)        colorWithUniformRGB:(color::RGB const&)rgb;
+ (instancetype)        colorWithUniformMonochrome:(color::Monochrome const&)mono;
- (color::RGBA)         uniformRGBA;
- (color::RGB)          uniformRGB;
- (color::Monochrome)   uniformMonochrome;
@end


#endif /// LIBIMREAD_EXT_CATEGORIES_NSCOLOR_PLUS_IM_HH_