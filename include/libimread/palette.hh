/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PALETTE_HH_
#define LIBIMREAD_PALETTE_HH_

#include <cstdint>
#include <limits>
#include <bitset>
#include <array>
#include <utility>
#include <iostream>

namespace im {
    
    template <typename Channel>
    struct ChannelBase {
        using Limits = std::numeric_limits<Channel>;
        static constexpr Channel min() { return Limits::min(); }
        static constexpr Channel max() { return Limits::max(); }
    };
    
    template <typename Channel>
    struct Mono : ChannelBase<Channel> {
        enum channels : Channel { Y, None };
        static constexpr uint8_t channel_count = 1;
    };
    
    template <typename Channel>
    struct RGB : ChannelBase<Channel> {
        enum channels : Channel { R, G, B, None };
        static constexpr uint8_t channel_count = 3;
    };
    
    template <typename Channel>
    struct BGR : ChannelBase<Channel> {
        enum channels : Channel { B, G, R, None };
        static constexpr uint8_t channel_count = 3;
    };
    
    template <typename Channel>
    struct RGBA : ChannelBase<Channel> {
        enum channels : Channel { R, G, B, A, None };
        static constexpr uint8_t channel_count = 4;
    };
    
    template <template <typename> typename ChannelMeta = RGBA,
              typename Composite = uint32_t,
              typename Channel = uint8_t>
    struct UniformColor : public ChannelMeta<Channel> {
        
        static_assert(sizeof(Composite) > sizeof(Channel),
                      "UniformColor needs a composite type larger than its channel type");
        
        static constexpr uint8_t N = ChannelMeta<Channel>::channel_count;
        using Meta = ChannelMeta<Channel>;
        using Components = alignas(Composite) Channel[N];
        
        using bitset_t = std::bitset<sizeof(Composite) * 8>;
        using array_t = std::array<Channel, N>;
        using index_t = std::make_index_sequence<N>;
        using component_t = Components;
        using composite_t = Composite;
        using channel_t = Channel;
        
        union alignas(Composite) {
            Components components;
            Composite composite{ 0 };
        };
        
        constexpr UniformColor() noexcept = default;
        
        explicit constexpr UniformColor(Composite c) noexcept
            :composite(c)
            {}
        
        explicit constexpr UniformColor(std::initializer_list<Channel> initlist) noexcept {
            int idx = 0;
            for (auto it = initlist.begin();
                 it != initlist.end() && idx < N;
                 ++it) { components[0] = *it; ++idx; }
        }
        
        constexpr operator Composite() noexcept { return composite; }
        constexpr operator array_t() noexcept { return array_impl(index_t()); }
        constexpr operator std::string() { return string_impl(index_t()); }
        
        constexpr Channel &operator[](std::size_t c) { return components[c]; }
        constexpr bool operator<(const UniformColor& rhs) noexcept { return composite < rhs.composite; }
        constexpr bool operator>(const UniformColor& rhs) noexcept { return composite > rhs.composite; }
        constexpr bool operator==(const UniformColor& rhs) noexcept { return composite == rhs.composite; }
        constexpr bool operator!=(const UniformColor& rhs) noexcept { return composite != rhs.composite; }
        
        private:
            template <std::size_t ...I> inline
            array_t array_impl(std::index_sequence<I...>) {
                return array_t{ components[I]... };
            }
            template <std::size_t ...I> inline
            std::string string_impl(std::index_sequence<I...>) {
                std::string out("{ ");
                int unpack[] __attribute__((unused)) { 0, 
                    (out += std::to_string(components[I]) + (I == N-1 ? "" : ", "), 0)...
                };
                out += " }";
                return out;
            }
    };
    
    using Monochrome = UniformColor<Mono, uint16_t, uint8_t>;
    using RGBAColor = UniformColor<>;
    using RGBColor = UniformColor<RGB>
    using HDRColor = UniformColor<RGBA, int64_t, int16_t>;
    
    template <typename Color = RGBAColor, std::size_t Nelems = 256>
    struct Palette {
        static constexpr std::size_t N = Nelems;
        using color_t = Color;
        using component_t = Color::component_t;
        using channel_t = Color::channel_t;
        using composite_t = Color::composite_t;
        
        
    };
    
}

#endif /// LIBIMREAD_PALETTE_HH_