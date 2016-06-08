/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdlib>
#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSString+STL.hh>

@implementation NSString (AXStringAdditions)

+ (instancetype) stringWithSTLString:(std::string const&)str {
    return [[NSString alloc] initWithUTF8String:str.c_str()];
}

+ (instancetype) stringWithSTLWideString:(std::wstring const&)wstr {
    unsigned siz = wstr.size() * sizeof(wchar_t);
    char const* bytes = reinterpret_cast<char const*>(wstr.data());
    return [[NSString alloc] initWithBytes:bytes
                                    length:siz
                                  encoding:kSTLWideStringEncoding];
}

- initWithSTLString:(std::string const&)str {
    return [self initWithUTF8String:str.c_str()];
}

- initWithSTLWideString:(std::wstring const&)wstr {
    unsigned siz = wstr.size() * sizeof(wchar_t);
    char const* bytes = reinterpret_cast<char const*>(wstr.data());
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
        static_cast<wchar_t const*>([enc bytes]),
        static_cast<unsigned>([enc length]) / sizeof(wchar_t));
}

@end

