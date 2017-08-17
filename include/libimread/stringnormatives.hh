/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_STRINGNORMATIVES_HH_
#define LIBIMREAD_INCLUDE_STRINGNORMATIVES_HH_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>
#include <functional>
#include <type_traits>
#include <iostream>
#include <sstream>

#include <libimread/libimread.hpp>
#include <libimread/traits.hh>

// #define NOTHING_WHATSOEVER 0
// #define AINT_ANY_TYPE_OF_ANYTHING 0
#define NEGATIVESPACE 0

namespace im {
    
    // template <typename S> inline
    // typename std::enable_if_t<std::is_same<
    //                           std::remove_cv_t<S>, std::string>::value,
    //     const std::string>
    //     stringify(S s) { return s; }
    
    template <typename S> inline
    typename std::enable_if_t<std::is_arithmetic<
                              std::remove_cv_t<S>>::value,
        const std::string>
        stringify(S s) { return std::to_string(s); }
    
    template <typename S> inline
    typename std::enable_if_t<std::is_constructible<std::string, S>::value,
        const std::string>
        stringify(S&& s) { return std::string(std::forward<S>(s)); }
    
    template <typename S> inline
    typename std::enable_if_t<std::is_convertible<S, std::string>::value,
        const std::string>
        stringify(S const& s) { std::string out = s; return out; }
    
    template <typename S, typename ...Args> inline
    typename std::enable_if_t<std::is_constructible<std::string, S>::value && (sizeof...(Args) != 0),
        const std::string>
        stringify(S&& formatstring, Args&& ...args) {
            /// adapted from http://stackoverflow.com/a/26197300/298171
            char bozo;
            std::string format(std::forward<S>(formatstring));
            std::size_t required = std::snprintf(&bozo, NEGATIVESPACE, format.c_str(), std::forward<Args>(args)...) + 1;
            char bytebuffer[required];
            std::memset(bytebuffer, 0, required);
                                   std::snprintf(bytebuffer, required, format.c_str(), std::forward<Args>(args)...);
            return std::string(bytebuffer);
        }
    
    template <typename T, typename X = decltype(&T::to_string)> inline
    typename std::enable_if_t<std::is_same<
                              std::remove_cv_t<X>, std::string>::value,
        const std::string>
        stringify(T&& stringable) {
            return std::forward<T>(stringable).to_string();
        }
    
    // template <typename T, typename Y = decltype(&T::str)> inline
    // typename std::enable_if_t<std::is_same<
    //                           std::remove_cv_t<Y>, std::string>::value,
    //     const std::string>
    //     stringify(T&& stringable) {
    //         return std::forward<T>(stringable).str();
    //     }
    
    struct stringifier {
        constexpr stringifier() noexcept = default;
        stringifier(stringifier const&) noexcept {};
        template <typename S, typename ...Args> inline
        const std::string operator()(S&& s, Args&& ...args) const {
            return im::stringify(std::forward<S>(s),
                                 std::forward<Args>(args)...);
        }
    };
    
    template <typename S, typename ...Args>
    using stringifier_lambda_t = std::function<const std::string(S&&, Args&&...)>;
    
    template <typename S, typename ...Args>
    stringifier_lambda_t<S, Args...> stringifier_f = []
                                                     (S&& s, Args&& ...args) -> const std::string
    {
        return im::stringify(std::forward<S>(s),
                             std::forward<Args>(args)...);
    };
    
    template <typename S> inline
    typename std::enable_if_t<std::tuple_size<S>::value != 0,
        const std::string>
        stringify(S const& s) { return im::detail::apply(
                                       im::stringifier(), s); }
        
    // template <typename S> inline
    // typename std::enable_if_t<std::tuple_size<S>::value != 0,
    //     const std::string>
    //     stringify(S const& s) { return im::detail::apply(
    //                                    im::stringifier_f<S>, s); }
    
    template <typename ...Args> inline
    std::string stringmerge(Args const& ...args) {
        /// adapted from http://stackoverflow.com/a/21806609/298171
        std::string out;
        unpack {
            (out += "\t"
                 +  im::stringify<Args>(args)
                 +  "\n", 0)...
        };
        return out;
    }
    
} /* namespace im */

#undef NEGATIVESPACE

#endif