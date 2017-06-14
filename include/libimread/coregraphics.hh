/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREGRAPHICS_HH_
#define LIBIMREAD_COREGRAPHICS_HH_

#include <memory>
#include <cstdlib>

#import  <CoreFoundation/CoreFoundation.h>
#import  <ImageIO/ImageIO.h>

/// These macros, like much other stuff in this implementation TU,
/// are adapted from the always-handy-dandy Ray Wenderlich -- specifically, q.v.:
/// https://www.raywenderlich.com/69855/image-processing-in-ios-part-1-raw-bitmap-modification
/// http://www.modejong.com/blog/post3_pixel_binary_layout_w_premultiplied_alpha/index.html

#define UNCOMPAND(x)    ((x) & 0xFF)
#define R(x)            (UNCOMPAND(x))
#define G(x)            (UNCOMPAND(x >> 8 ))
#define B(x)            (UNCOMPAND(x >> 16))
#define A(x)            (UNCOMPAND(x >> 24))

#define CG_FLOAT(x)     static_cast<CGFloat>(x)

#endif /// LIBIMREAD_COREGRAPHICS_HH_