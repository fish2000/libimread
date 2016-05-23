/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_MESSAGE_HH
#define LIBIMREAD_OBJC_RT_MESSAGE_HH

#include <cstdlib>
#include <utility>
#include <type_traits>
#include "types.hh"
#include "selector.hh"
#include "message-args.hh"
#include "traits.hh"
#include "object.hh"

namespace objc {
    
    struct msg {
        
        objc::id target; /// scoped retain/release
        objc::selector action;
        
        explicit msg(types::ID s, types::selector o)
            :target(objc::id(s))
            ,action(objc::selector(o))
            {}
        
        template <typename ...Args>
        void send(types::boolean dispatch, Args ...args) const {
            arguments<void, Args...> ARGS(args...);
            ARGS.send(target.self, action.sel);
        }
        
        template <typename M,
                  typename X = std::enable_if_t<traits::is_argument_list<M>::value()>>
        types::ID sendv(M&& arg_list) const {
            return arg_list.send(target.self, action.sel);
        }
        
        template <typename Return, typename ...Args>
        static Return get(types::ID s, types::selector op, Args ...args) {
            arguments<Return, Args...> ARGS(args...);
            const objc::id selfie(s); /// for scoped retain/release
            return ARGS.send(selfie.self, op);
        }
        
        template <typename ...Args>
        static types::ID send(types::ID s, types::selector op, Args ...args) {
            arguments<types::ID, Args...> ARGS(args...);
            const objc::id selfie(s);
            return ARGS.send(selfie.self, op);
        }
        
        template <typename M,
                  typename X = std::enable_if_t<traits::is_argument_list<M>::value()>>
        static types::ID sendv(types::rID s, types::rSEL op, M&& arg_list) {
            return arg_list.send(s, op);
        }
        
        private:
            msg(const msg&);
            msg(msg&&);
            msg &operator=(const msg&);
            msg &operator=(msg&&);
        
    };
    
} /* namespace objc */


#endif /// LIBIMREAD_OBJC_RT_MESSAGE_HH