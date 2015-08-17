/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <array>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSColor+IM.hh>

#ifdef __OBJC__

@implementation NSColor (IMColorAdditions)
+ (NSColor *) colorWithUniformRGBA:(const im::color::RGBA&)rgba {
    using Color = im::color::RGBA;
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
+ (NSColor *) colorWithUniformRGB:(const im::color::RGB&)rgb {
    using Color = im::color::RGB;
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
+ (NSColor *) colorWithUniformMonochrome:(const im::color::Monochrome&)mono {
    using Color = im::color::Monochrome;
    CGFloat white = (CGFloat)((float)mono.components[0] / (float)Color::max());
    return [NSColor colorWithWhite:white
                             alpha:(CGFloat)0.0f];
}
- (im::color::RGBA) uniformRGBA {
    using Color = im::color::RGBA;
    using Channel = Color::channel_t;
    Color out{
        static_cast<Channel>([self redComponent]    * Color::max()),
        static_cast<Channel>([self greenComponent]  * Color::max()),
        static_cast<Channel>([self blueComponent]   * Color::max()),
        static_cast<Channel>([self alphaComponent]  * Color::max()) };
    return out;
}
- (im::color::RGB) uniformRGB {
    using Color = im::color::RGB;
    using Channel = Color::channel_t;
    Color out{
        static_cast<Channel>([self redComponent]    * Color::max()),
        static_cast<Channel>([self greenComponent]  * Color::max()),
        static_cast<Channel>([self blueComponent]   * Color::max()) };
    return out;
}
- (im::color::Monochrome) uniformMonochrome {
    using Color = im::color::Monochrome;
    using Channel = Color::channel_t;
    Color out{
        static_cast<Channel>([self brightnessComponent] * Color::max()) };
    return out;
}
@end

#endif /// __OBJC__