/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_BOUNDARIES_HH_
#define LIBIMREAD_EXT_BOUNDARIES_HH_

#include <algorithm>
#include <functional>

namespace im {
    
    namespace detail {
        
        struct ARG {} arg;
        
    }
    
}

namespace std {
    
    template <>
    struct is_placeholder<im::detail::ARG> : public std::integral_constant<int, 1> {};
    
}

namespace im {
    
    template <typename T>
    using clamper_f = std::function<T&(T&...)>;
    
    template <typename T,
              typename = std::enable_if_t<std::is_arithmetic<
                                          std::remove_cv_t<T>>::value>>
    clamper_f<T> clamper(T& lower_bound, T& upper_bound) {
        return std::bind(std::clamp<T>, detail::arg, lower_bound, upper_bound);
    }
    
    template <typename T,
              typename = std::enable_if_t<std::is_arithmetic<
                                          std::remove_cv_t<T>>::value>>
    clamper_f<T> clamper(T& upper_bound) {
        return std::bind(std::clamp<T>, detail::arg, 0, upper_bound);
    }
    
    template <typename T,
              typename = std::enable_if_t<std::is_arithmetic<
                                          std::remove_cv_t<T>>::value>>
    struct boundaries_t {
        
        T min_x{ 0 }; T min_y{ 0 };
        T max_x{ 0 }; T max_y{ 0 };
        
        constexpr boundaries_t(T maxX, T maxY)
            :max_x{ maxX }
            ,max_y{ maxY }
            {}
        
        constexpr boundaries_t(T minX, T minY,
                               T maxX, T maxY)
            :min_x{ minX }
            ,min_y{ minY }
            ,max_x{ maxX }
            ,max_y{ maxY }
            {}
        
        constexpr boundaries_t() noexcept = delete;
        constexpr boundaries_t(boundaries_t const&) = default;
        constexpr boundaries_t(boundaries_t&&) noexcept = default;
        
        constexpr T&       width(T arg)       & { return std::clamp(arg, min_x, max_x); }
        constexpr T&       height(T arg)      & { return std::clamp(arg, min_y, max_y); }
        constexpr T const& width(T arg)  const& { return std::clamp(arg, min_x, max_x); }
        constexpr T const& height(T arg) const& { return std::clamp(arg, min_y, max_y); }
        
    };
    
} /// namespace im

#endif /// LIBIMREAD_EXT_BOUNDARIES_HH_