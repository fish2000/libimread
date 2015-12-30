/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_MESSAGE_ARGS_HH
#define LIBIMREAD_OBJC_RT_MESSAGE_ARGS_HH

#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include <array>
#include <utility>
#include <functional>
#include <type_traits>

#include "types.hh"

namespace objc {
    
    /// variadic tuple-unpacking argument wrapper structure
    /// for referencing objective-c message arguments passed
    /// to objc_msgSend and its multitudinous variants.
    /// ... users really shouldn't need to invoke this directly;
    /// use  `objc::msg` instead, see its definition and notes below.
    
    template <typename Return, typename ...Args>
    struct arguments {
        static constexpr std::size_t argc = sizeof...(Args);
        using is_argument_list_t = std::true_type;
        using sequence_t = std::make_index_sequence<argc>;
        using return_t = Return;
        using tuple_t = std::tuple<Args...>;
        using prebound_t = std::function<return_t(types::ID, types::selector, Args...)>;
        using sender_t = typename std::conditional<
                                  std::is_void<Return>::value,
                                      void_sender_t<Args...>,
                                      return_sender_t<return_t, Args...>>::type;
        
        /// I like my members like I like my args: tupled
        tuple_t args;
        
        /// You would think that one of these function pointers -- probably the one
        /// corresponding to std::is_class<T> -- would be objc_msgSend_stret, right?
        /// WRONG. As it turns out, that is only what you want if you like segfaults;
        /// the _stret-related functionality is actually somehow included in 
        /// plain ol' objc_msgSend() these days. WHO KNEW.
        sender_t dispatcher = (std::is_floating_point<Return>::value ? (sender_t)objc_msgSend_fpret : 
                              (std::is_class<Return>::value          ? (sender_t)objc_msgSend : 
                                                                       (sender_t)objc_msgSend));
        
        template <typename Tuple,
                  typename X = std::enable_if_t<
                               std::is_same<Tuple, tuple_t>::value &&
                               std::tuple_size<Tuple>::value == argc>>
        explicit arguments(Tuple t)
            :args(t)
            {}
        
        explicit arguments(Args... a)
            :args(std::forward_as_tuple(a...))
            {}
        
        private:
            template <std::size_t ...I> inline
            void void_impl(types::ID self, types::selector op, std::index_sequence<I...>) const {
                static_cast<prebound_t>(dispatcher)(self, op, std::get<I>(args)...);
            }
            
            template <std::size_t ...I> inline
            return_t send_impl(types::ID self, types::selector op, std::index_sequence<I...>) const {
                return static_cast<prebound_t>(dispatcher)(self, op, std::get<I>(args)...);
            }
        
        public:
            inline auto send(types::ID self, types::selector op) const -> return_t {
                if (!std::is_void<Return>::value) {
                    /// dead code elimination collapses this conditional
                    return send_impl(self, op, sequence_t());
                }
                void_impl(self, op, sequence_t());
            }
        
        private:
            arguments(const arguments&);
            arguments(arguments&&);
            arguments &operator=(const arguments&);
            arguments &operator=(arguments&&);
    };
    
    /// objc::arguments<...> subclass used for reimplementing objc_msgSendv()
    /// ... which you may ask, why did I do that? Why would a sane person do that?
    /// Hahaha. I had my reasons, of which I am not, at time of writing,
    /// necessarily proud. Buy me a beer and I will explain it to you.
    
    template <typename Return, typename ...Args>
    struct message : public arguments<Return, Args...> {
        using arguments_t = arguments<Return, Args...>;
        using arguments_t::argc;
        using arguments_t::args;
        
        types::selector op;
        types::ID __unsafe_unretained self;
        
        explicit message(types::ID s, types::selector o, Args&&... a)
            :arguments_t(a...), self(s), op(o)
            {
                #if __has_feature(objc_arc)
                    [self retain];
                #endif
            }
        
        virtual ~message() {
            #if __has_feature(objc_arc)
                [self release];
            #endif
        }
        
        private:
            message(const message&);
            message(message&&);
            message &operator=(const message&);
            message &operator=(message&&);
            
    };
    
} /* namespace objc */


#endif /// LIBIMREAD_OBJC_RT_MESSAGE_ARGS_HH