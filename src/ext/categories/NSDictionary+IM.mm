/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <string>
#include <vector>
#include <algorithm>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/categories/NSString+STL.hh>
#include <libimread/ext/categories/NSDictionary+IM.hh>

#ifdef __OBJC__

@implementation NSDictionary (AXDictionaryAdditions)

+ (instancetype) dictionaryWithOptionsMap:(options_map const&)optionsMap {
    NSDictionary* optionsDict;
    std::string optionsJSONString = optionsMap.format();
    NSError* error;
    NSData* optionsJSON = [[NSData alloc] initWithBytes:(const void*)optionsJSONString.data()
                                                 length:(NSInteger)optionsJSONString.size()];
    optionsDict = (NSDictionary*)[NSJSONSerialization JSONObjectWithData:optionsJSON
                                                                 options:static_cast<NSJSONReadingOptions>(0)
                                                                   error:&error];
    imread_assert(optionsDict != nil,
                "NSDictionary error in dictionaryWithOptionsMap:",
                [error.localizedDescription STLString]);
    return optionsDict;
}

- initWithOptionsMap:(options_map const&)optionsMap {
    NSDictionary* optionsDict;
    std::string optionsJSONString = optionsMap.format();
    NSError* error;
    NSData* optionsJSON = [[NSData alloc] initWithBytes:(const void*)optionsJSONString.data()
                                                 length:(NSInteger)optionsJSONString.size()];
    optionsDict = (NSDictionary*)[NSJSONSerialization JSONObjectWithData:optionsJSON
                                                                 options:static_cast<NSJSONReadingOptions>(0)
                                                                   error:&error];
    imread_assert(optionsDict != nil,
                "NSDictionary error in initWithOptionsMap:",
                [error.localizedDescription STLString]);
    return [self initWithDictionary:optionsDict];
}

- (options_map) asOptionsMap {
    /// NSJSONWritingPrettyPrinted
    NSError* error;
    NSData* datum = [NSJSONSerialization dataWithJSONObject:self
                                                    options:static_cast<NSJSONWritingOptions>(0)
                                                      error:&error];
    NSString* json = [[NSString alloc] initWithData:datum
                                           encoding:NSUTF8StringEncoding];
    options_map out = options_map::parse([json STLString]);
    return out;
}

@end

#endif /// __OBJC__