/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdlib>
#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSString+STL.hh>

#ifdef __OBJC__

@implementation NSString (IMStringAdditions)

+ (instancetype) stringWithSTLString:(const std::string&)str {
    return [[NSString alloc] initWithUTF8String:str.c_str()];
}

+ (instancetype) stringWithSTLWideString:(const std::wstring&)wstr {
    unsigned siz = wstr.size() * sizeof(wchar_t);
    const char* bytes = reinterpret_cast<const char*>(wstr.data());
    return [[NSString alloc] initWithBytes:bytes
                                    length:siz
                                  encoding:kSTLWideStringEncoding];
}

- initWithSTLString:(const std::string&)str {
    return [self initWithUTF8String:str.c_str()];
}

- initWithSTLWideString:(const std::wstring&)wstr {
    unsigned siz = wstr.size() * sizeof(wchar_t);
    const char* bytes = reinterpret_cast<const char*>(wstr.data());
    return [self initWithBytes:bytes
                        length:siz
                      encoding:kSTLWideStringEncoding];
}

- (std::string) STLString {
    return [self cStringUsingEncoding:NSUTF8StringEncoding];
}

- (std::string) STLStringUsingEncoding:(NSStringEncoding)encoding {
    return [self cStringUsingEncoding:encoding];
}

- (std::wstring) STLWideString {
    NSData* enc = [self dataUsingEncoding:kSTLWideStringEncoding];
    return std::wstring(
        static_cast<const wchar_t*>([enc bytes]),
        static_cast<unsigned>([enc length]) / sizeof(wchar_t));
}

@end

#endif /// __OBJC__