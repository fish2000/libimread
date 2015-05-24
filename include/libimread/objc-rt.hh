/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_HH
#define LIBIMREAD_OBJC_RT_HH

#include <tuple>

#ifdef __OBJC__
#import <objc/runtime.h>
#endif

namespace objc {
    
    struct id {
    
        ::id iid;
        
        explicit id(::id ii)
            :iid(ii)
            {}
        
        operator ::id() { return iid; }
        
        struct cls {
            std::string name() {
                return std::string(::object_getClassName(iid));
            }
            ::Class lookup() {
                return ::objc_lookUpClass(::object_getClassName(iid));
            }
        };
    
    };
    
    struct selector {
        
        ::SEL sel;
        
        explicit selector(::SEL s)
            :sel(s)
            {}
        explicit selector(const std::string &name)
            :sel(::sel_registerName(name.c_str()))
            {}
        explicit selector(const char *name)
            :sel(::sel_registerName(name))
            {}
        
        operator ::SEL() { return sel; }
        
        bool operator==(const objc::selector &s) {
            return ::sel_isEqual(sel, s.sel) == YES;
        }
        bool operator!=(const objc::selector &s) {
            return ::sel_isEqual(sel, s.sel) == NO;
        }
        
        std::string name() {
            return std::string(::sel_getName(sel));
        }
        
        static objc::selector register(const std::string &name) {
            return objc::selector(::sel_registerName(name.c_str()));
        }
        static objc::selector register(const char *name) {
            return objc::selector(::sel_registerName(name));
        }
        
    };
    
    objc::selector operator"" _SEL(const char *name) {
        return objc::selector(name);
    }
    
    template <typename ...Args>
    struct arguments {
        using N = sizeof...(Args);
        using make_index = std::index_sequence_for<Args...>;
        using index_type = std::index_sequence<I...>;
        using tuple_type = std::tuple<Args&&...>;
        
        const std::size_t argc;
        tuple_type args;
        
        explicit arguments(Args&&... a)
            :args(std::forward_as_tuple(a...))
            ,argc(N)
            {}
        
        private:
            template <std::size_t ...I>
            ::id send_impl(::id self, ::SEL op, index_type idx) {
                return ::objc_msgSend(self, op,
                    std::get<I>(std::forward<Args>(args))...);
            }
        
        public:
            ::id send(::id self, ::SEL op) {
                return send_impl(self, op, index_type());
            }
        
        private:
            arguments(const arguments&);
            arguments(arguments&&);
            arguments &operator=(const arguments&);
            arguments &operator=(arguments&&);
    };
    
    struct msg {
        
        ::id self;
        ::SEL op;
        
        explicit msg(::id s, ::SEL o)
            :self(s), op(o)
            {}
        
        template <typename ...Args>
        ::id send(Args&& ...args) {
            objc::arguments ARGS(args...);
            return ARGS.send(self, op);
        }
        
        template <typename ...Args>
        static ::id send(::id self, ::SEL op, Args&& ...args) {
            objc::arguments ARGS(args...);
            return ARGS.send(self, op);
        }
        
        private:
            msg(const msg&);
            msg(msg&&);
            msg &operator=(const msg&);
            msg &operator=(msg&&);
        
    };
    
}

#endif /// LIBIMREAD_OBJC_RT_HH