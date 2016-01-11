/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <array>
#include <algorithm>
#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSColor+IM.hh>

#ifdef __OBJC__

using namespace im;

@implementation NSColor (AXColorAdditions)

+ (instancetype) colorWithUniformRGBA:(color::RGBA const&)rgba {
    using Color = color::RGBA;
    using Channel = Color::channel_t;
    Color::array_t array = rgba.to_array();
    std::array<CGFloat, Color::N> farray;
    std::transform(array.begin(), array.end(),
                   farray.begin(), [](Channel channel) {
                       return (CGFloat)((float)channel / (float)Color::max());
                   });
    return [NSColor colorWithRed:farray[0]
                           green:farray[1]
                            blue:farray[2]
                           alpha:farray[3]];
}

+ (instancetype) colorWithUniformRGB:(color::RGB const&)rgb {
    using Color = color::RGB;
    using Channel = Color::channel_t;
    Color::array_t array = rgb.to_array();
    std::array<CGFloat, Color::N> farray;
    std::transform(array.begin(), array.end(),
                   farray.begin(), [](Channel channel) {
                       return (CGFloat)((float)channel / (float)Color::max());
                   });
    return [NSColor colorWithRed:farray[0]
                           green:farray[1]
                            blue:farray[2]
                           alpha:(CGFloat)0.0f];
}

+ (instancetype) colorWithUniformMonochrome:(color::Monochrome const&)mono {
    using Color = color::Monochrome;
    CGFloat white = (CGFloat)((float)mono.components[0] / (float)Color::max());
    return [NSColor colorWithWhite:white
                             alpha:(CGFloat)0.0f];
}
- (color::RGBA) uniformRGBA {
    using Color = color::RGBA;
    using Channel = Color::channel_t;
    Color out{
        static_cast<Channel>([self redComponent]    * Color::max()),
        static_cast<Channel>([self greenComponent]  * Color::max()),
        static_cast<Channel>([self blueComponent]   * Color::max()),
        static_cast<Channel>([self alphaComponent]  * Color::max()) };
    return out;
}

- (color::RGB) uniformRGB {
    using Color = color::RGB;
    using Channel = Color::channel_t;
    Color out{
        static_cast<Channel>([self redComponent]    * Color::max()),
        static_cast<Channel>([self greenComponent]  * Color::max()),
        static_cast<Channel>([self blueComponent]   * Color::max()) };
    return out;
}

- (color::Monochrome) uniformMonochrome {
    using Color = color::Monochrome;
    using Channel = Color::channel_t;
    Color out{
        static_cast<Channel>([self brightnessComponent] * Color::max()) };
    return out;
}

@end
#endif /// __OBJC__