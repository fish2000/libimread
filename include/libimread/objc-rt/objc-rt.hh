/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_OBJC_RT_HH
#define LIBIMREAD_OBJC_RT_OBJC_RT_HH

#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include <array>
#include <utility>
#include <functional>
#include <type_traits>

#ifdef __APPLE__
// #import <libimread/ext/categories/NSString+STL.hh>
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <objc/message.h>
#import <objc/runtime.h>
#endif /// __APPLE__

#include "types.hh"
#include "selector.hh"
#include "message-args.hh"
#include "traits.hh"
#include "object.hh"
#include "message.hh"
#include "namespace-std.hh"
#include "namespace-im.hh"


#endif /// LIBIMREAD_OBJC_RT_OBJC_RT_HH