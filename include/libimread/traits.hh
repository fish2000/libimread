/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_TRAITS_HH_
#define LIBIMREAD_TRAITS_HH_

#include <type_traits>
#include <libimread/libimread.hpp>

//#define TEST_ARGS byte_source*, ImageFactory*, const Options&
#define TEST_ARGS T

namespace im {
    
    namespace detail {
        
        template <bool B, typename T = void>
        using disable_if = std::enable_if<!B, T>;
        
        template <bool B, typename T = void>
        using disable_if_t = std::enable_if_t<!B, T>;
        
        /// apply_impl() and apply() courtesy of http://stackoverflow.com/a/19060157/298171
        template <typename F, typename Tuple, std::size_t ...I> inline
        auto apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
            return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
        }
        
        template <typename F, typename Tuple> inline
        auto apply(F&& f, Tuple&& t) {
            using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
            return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices());
        }
        
    } /* namespace detail */
    
    namespace detail {
        
        /// “all_of” bit based on http://stackoverflow.com/q/13562823/298171
        template <bool ...booleans>
        struct static_all_of;
        
        /// derive recursively, if the first argument is true:
        template <bool ...tail>
        struct static_all_of<true, tail...> : static_all_of<tail...>
        {};
        
        /// end recursion: false, if first argument is false:
        template <bool ...tail>
        struct static_all_of<false, tail...> : std::false_type
        {};
        
        /// end recursion: true, if all arguments have been exhausted:
        template <>
        struct static_all_of<> : std::true_type
        {};
        
        template <bool ...booleans>
        using static_all_of_t = typename static_all_of<booleans...>::type;
        
        template <bool ...booleans>
        constexpr bool static_all_of_v = static_all_of<booleans...>::value;
        
        template <typename Type, typename ...Requirements>
        struct are_bases_of : static_all_of_t<std::is_base_of<Type,
                                              std::decay_t<Requirements>>::value...>
        {};
        
        template <typename ...Requirements>
        struct are_arithmetic : static_all_of_t<std::is_arithmetic<
                                                std::decay_t<Requirements>>::value...>
        {};
        
        template <typename ...Requirements>
        struct are_string_ctorable : static_all_of_t<std::is_constructible<std::string,
                                                     std::decay_t<Requirements>>::value...>
        {};
        
        template <typename ...Requirements>
        struct are_string_convertible : static_all_of_t<std::is_convertible<std::decay_t<Requirements>,
                                                        std::string>::value...>
        {};
        
    } /* namespace detail */
    
    namespace traits {
        
        template <typename>
        struct sfinae_true : std::true_type{};
        
        namespace detail {
            template <typename T, typename ...Args>
            static auto test_read(int) -> typename T::can_read;
            template <typename, typename ...Args>
            static auto test_read(long) -> std::false_type;
            
            template <typename T, typename ...Args>
            static auto test_read_multi(int) -> typename T::can_read_multi;
            template <typename, typename ...Args>
            static auto test_read_multi(long) -> std::false_type;
            
            template <typename T, typename ...Args>
            static auto test_read_metadata(int) -> typename T::can_write_metadata;
            template <typename, typename ...Args>
            static auto test_read_metadata(long) -> std::false_type;
            
            template <typename T, typename ...Args>
            static auto test_write(int) -> typename T::can_write;
            template <typename, typename ...Args>
            static auto test_write(long) -> std::false_type;
            
            template <typename T, typename ...Args>
            static auto test_write_multi(int) -> typename T::can_write_multi;
            template <typename, typename ...Args>
            static auto test_write_multi(long) -> std::false_type;
            
            template <typename T, typename ...Args>
            static auto test_write_metadata(int) -> typename T::can_write_metadata;
            template <typename, typename ...Args>
            static auto test_write_metadata(long) -> std::false_type;
            
        }
        
        template <typename T>
        struct has_read : decltype(detail::test_read<T, TEST_ARGS>(0)) {
            template <typename X = std::enable_if<decltype(detail::test_read<T, TEST_ARGS>(0))::value>>
            static constexpr bool value() { return true; }
            static constexpr bool value() { return detail::test_read<T, TEST_ARGS>(0); }
        };
        
        template <typename T>
        struct has_read_multi : decltype(detail::test_read_multi<T, TEST_ARGS>(0)){
            template <typename X = std::enable_if<decltype(detail::test_read_multi<T, TEST_ARGS>(0))::value>>
            static constexpr bool value() { return true; }
            static constexpr bool value() { return detail::test_read_multi<T, TEST_ARGS>(0); }
        };
        
        template <typename T>
        struct has_read_metadata : decltype(detail::test_read_metadata<T, TEST_ARGS>(0)){
            template <typename X = std::enable_if<decltype(detail::test_read_metadata<T, TEST_ARGS>(0))::value>>
            static constexpr bool value() { return true; }
            static constexpr bool value() { return detail::test_read_metadata<T, TEST_ARGS>(0); }
        };
        
        template <typename T>
        struct has_write : decltype(detail::test_write<T, TEST_ARGS>(0)){
            template <typename X = std::enable_if<decltype(detail::test_write<T, TEST_ARGS>(0))::value>>
            static constexpr bool value() { return true; }
            static constexpr bool value() { return detail::test_write<T, TEST_ARGS>(0); }
        };
        
        template <typename T>
        struct has_write_multi : decltype(detail::test_write_multi<T, TEST_ARGS>(0)){
            template <typename X = std::enable_if<decltype(detail::test_write_multi<T, TEST_ARGS>(0))::value>>
            static constexpr bool value() { return true; }
            static constexpr bool value() { return detail::test_write_multi<T, TEST_ARGS>(0); }
        };
        
        template <typename T>
        struct has_write_metadata : decltype(detail::test_write_metadata<T, TEST_ARGS>(0)){
            template <typename X = std::enable_if<decltype(detail::test_write_metadata<T, TEST_ARGS>(0))::value>>
            static constexpr bool value() { return true; }
            static constexpr bool value() { return detail::test_write_metadata<T, TEST_ARGS>(0); }
        };
    } /* namespace traits */

} /* namespace im */

#undef TEST_ARGS

#endif /// LIBIMREAD_TRAITS_HH_