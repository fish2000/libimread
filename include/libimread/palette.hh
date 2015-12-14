/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PALETTE_HH_
#define LIBIMREAD_PALETTE_HH_

#include <cstdint>
#include <iomanip>
#include <utility>
#include <limits>
#include <array>
#include <set>
#include <bitset>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <initializer_list>

#include <libimread/libimread.hpp>
#include <libimread/process/neuquant.hh>
#include <libimread/color.hh>
#include <libimread/errors.hh>

#ifndef ALIGN_AS
#define ALIGN_AS(type) __attribute__((aligned (alignof(type))))
#endif

namespace im {
    
    template <typename Color = color::RGBA, std::size_t Nelems = 256>
    struct Palette {
        static constexpr std::size_t N = Nelems;
        static constexpr std::size_t C = Color::Meta::channel_count;
        using color_t = Color;
        using nonvalue_t = typename Color::NonValue;
        using component_t = typename Color::component_t;
        using channel_t = typename Color::channel_t;
        using composite_t = typename Color::composite_t;
        using string_array_t = std::array<std::string, N>;
        using array_t = std::array<component_t, N>;
        using sequence_t = std::make_index_sequence<N>;
        
        using channel_list_t = typename Color::channel_list_t;
        using channel_listlist_t = std::initializer_list<channel_list_t>;
        using composite_list_t = std::initializer_list<composite_t>;
        using composite_listlist_t = std::initializer_list<composite_list_t>;
        
        std::set<Color> items;
        
        constexpr Palette() noexcept = default;
        constexpr Palette(const Palette& other)
            :items(other.items)
            {}
        constexpr Palette(Palette&& other) {
            items.swap(other.items);
        }
        
        explicit constexpr Palette(composite_list_t initlist)     { add_impl(initlist); }
        explicit constexpr Palette(channel_listlist_t initlist)   { add_impl(initlist); }
        Palette &operator=(const Palette& other)        { items = other.items;                      return *this; }
        Palette &operator=(Palette&& other)             { items.clear(); items.swap(other.items);   return *this; }
        Palette &operator=(composite_list_t initlist)   { items.clear(); add_impl(initlist);        return *this; }
        Palette &operator=(channel_listlist_t initlist) { items.clear(); add_impl(initlist);        return *this; }
        
        constexpr bool add(const nonvalue_t& nonvalue)            { return false; }
        constexpr bool add(const Color& color)                    { return items.insert(color).second; }
        constexpr bool add(composite_t composite)                 { return items.emplace(composite).second; }
        constexpr bool add(channel_list_t channel_list)           { return items.emplace(channel_list).second; }
        constexpr bool bulk_add(composite_list_t composite_list)  { return add_impl(composite_list); }
        constexpr bool bulk_add(channel_listlist_t channel_list)  { return add_impl(channel_list); }
        
        template <typename pT>
        void rawcopy(pT* rawptr) const { /// neuquant::u8
            array_t array = to_component_array();
            const std::size_t siz = size();
            for (unsigned int color = 0; color < siz; color++) {
                for (unsigned int channel = 0; channel < C; channel++) {
                    *rawptr = static_cast<pT>(array[color][channel]);    /// copy element
                    rawptr++;                                            /// move forward
                }
            }
        }
        
        constexpr std::size_t max_size() const noexcept { return N; }
        inline const std::size_t size() const { return items.size(); }
        
        inline const bool contains(const nonvalue_t& nonvalue) const { return false; }
        inline const bool contains(const Color& color) const {
            return static_cast<bool>(items.count(color));
        }
        inline const bool contains(composite_t composite) const {
            return static_cast<bool>(items.count(Color(composite)));
        }
        
        inline const bool remove(const nonvalue_t& nonvalue) { return false; }
        inline const bool remove(const Color& color) { return static_cast<bool>(items.erase(color)); }
        inline const bool remove(composite_t color) { return static_cast<bool>(items.erase(Color(color))); }
        inline const bool remove(channel_list_t color) { return static_cast<bool>(items.erase(Color(color))); }
        
        Color operator[](composite_t composite) const noexcept {
            try {
                auto search = items.find(Color(composite));
                if (search != items.end()) { return *search; }
            } catch (std::exception& e) {
                return nonvalue_t();
            }
            return nonvalue_t();
        }
        Color operator[](const Color& color) const noexcept {
            try {
                return items.at(color);
            } catch (std::out_of_range& e) {
                return nonvalue_t();
            }
        }
        
        constexpr bool operator<(const Palette& rhs) const noexcept { return bool(items < rhs.items); }
        constexpr bool operator>(const Palette& rhs) const noexcept { return bool(items > rhs.items); }
        constexpr bool operator<=(const Palette& rhs) const noexcept { return bool(items <= rhs.items); }
        constexpr bool operator>=(const Palette& rhs) const noexcept { return bool(items >= rhs.items); }
        constexpr bool operator==(const Palette& rhs) const noexcept { return bool(items == rhs.items); }
        constexpr bool operator!=(const Palette& rhs) const noexcept { return bool(items != rhs.items); }
        
        const std::string to_string() const {
            static const std::string prefix("* ");
            std::string out("");
            string_array_t array = to_string_array();
            std::for_each(array.begin(), array.end(),
                      [&](const std::string& s) {
                out += prefix + s + "\n";
            });
            return out;
        }
        
        private:
            array_t to_component_array() const {
                array_t array;
                array.fill(Color(0).components);
                std::transform(items.begin(), items.end(),
                               array.begin(), [](const Color& color) {
                    return color.components;
                });
                return array;
            }
            
            string_array_t to_string_array() const {
                static const std::string filler("");
                string_array_t array;
                array.fill(filler);
                std::transform(items.begin(), items.end(),
                               array.begin(), [](const Color& color) {
                    return color.to_string();
                });
                return array;
            }
            
            template <typename List> inline
            bool add_impl(List list) {
                int idx;
                const int siz = idx = size();
                auto seterator = items.begin();
                for (auto it = list.begin();
                     it != list.end() && idx < N;
                     ++it) { seterator = items.emplace_hint(seterator, *it);
                             ++idx; }
                return static_cast<bool>(siz < size());
            }
    };
    
}

#endif /// LIBIMREAD_PALETTE_HH_