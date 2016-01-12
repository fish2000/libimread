/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <string>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/categories/NSData+IM.hh>
#include <libimread/ext/categories/NSString+STL.hh>
#include <libimread/ext/categories/NSDictionary+IM.hh>

#ifdef __OBJC__

using im::options_map;
static const NSJSONReadingOptions readingOptions = static_cast<NSJSONReadingOptions>(0);
static const NSJSONWritingOptions writingOptions = static_cast<NSJSONWritingOptions>(0);

@implementation NSDictionary (AXDictionaryAdditions)

+ (instancetype) dictionaryWithOptionsMap:(options_map const&)optionsMap {
    NSDictionary* optionsDict;
    NSError* error;
    NSData* optionsJSON = [[NSData alloc] initWithSTLString:optionsMap.format()];
    optionsDict = (NSDictionary*)[NSJSONSerialization JSONObjectWithData:optionsJSON
                                                                 options:readingOptions
                                                                   error:&error];
    imread_assert(optionsDict != nil,
                  "NSDictionary error in dictionaryWithOptionsMap:",
                  [error.localizedDescription STLString]);
    return optionsDict;
}

- initWithOptionsMap:(options_map const&)optionsMap {
    NSDictionary* optionsDict;
    NSError* error;
    NSData* optionsJSON = [[NSData alloc] initWithSTLString:optionsMap.format()];
    optionsDict = (NSDictionary*)[NSJSONSerialization JSONObjectWithData:optionsJSON
                                                                 options:readingOptions
                                                                   error:&error];
    imread_assert(optionsDict != nil,
                  "NSDictionary error in initWithOptionsMap:",
                  [error.localizedDescription STLString]);
    return [self initWithDictionary:optionsDict];
}

- (options_map) asOptionsMap {
    NSError* error;
    NSData* datum = [NSJSONSerialization dataWithJSONObject:self
                                                    options:writingOptions
                                                      error:&error];
    imread_assert(datum != nil,
                  "NSDictionary error in asOptionsMap",
                  [error.localizedDescription STLString]);
    return options_map::parse([datum STLString]);
}

@end

#endif /// __OBJC__