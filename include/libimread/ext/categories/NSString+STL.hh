/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_CATEGORIES_NSSTRING_PLUS_STL_HH_
#define LIBIMREAD_EXT_CATEGORIES_NSSTRING_PLUS_STL_HH_

#include <string>

#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

#ifndef FUNC_NAME_WTF
#define FUNC_NAME_WTF(stuff) CFStringConvertEncodingToNSStringEncoding(stuff)
#endif /// FUNC_NAME_WTF

#if TARGET_RT_BIG_ENDIAN
    const NSStringEncoding kSTLWideStringEncoding = FUNC_NAME_WTF(kCFStringEncodingUTF32BE);
#else
    const NSStringEncoding kSTLWideStringEncoding = FUNC_NAME_WTF(kCFStringEncodingUTF32LE);
#endif /// TARGET_RT_BIG_ENDIAN

@interface NSString (IMStringAdditions)
+ (NSString *)   stringWithSTLString:(const std::string&)str;
+ (NSString *)   stringWithSTLWideString:(const std::wstring&)wstr;
- (NSString *)   initWithSTLString:(const std::string&)str;
- (NSString *)   initWithSTLWideString:(const std::wstring&)wstr;
- (std::string)  STLString;
- (std::string)  STLStringUsingEncoding:(NSStringEncoding)encoding;
- (std::wstring) STLWideString;
@end

#ifdef FUNC_NAME_WTF
#undef FUNC_NAME_WTF
#endif /// FUNC_NAME_WTF

#endif /// LIBIMREAD_EXT_CATEGORIES_NSSTRING_PLUS_STL_HH_