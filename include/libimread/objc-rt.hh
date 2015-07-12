/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_HH
#define LIBIMREAD_OBJC_RT_HH

#include <string>
#include <tuple>
#include <type_traits>

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
        
        std::string classname() {
            return std::string(::object_getClassName(iid));
        }
        
        ::Class lookup() {
            return ::objc_lookUpClass(::object_getClassName(iid));
        }
        
        static std::string classname(::id ii) {
            return std::string(::object_getClassName(ii));
        }
        
        static ::Class lookup(::id ii) {
            return ::objc_lookUpClass(::object_getClassName(ii));
        }
        static ::Class lookup(const std::string &s) {
            return ::objc_lookUpClass(s.c_str());
        }
        static ::Class lookup(const char *s) {
            return ::objc_lookUpClass(s);
        }
    
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
        
        bool operator==(const objc::selector &s) {
            return ::sel_isEqual(sel, s.sel) == YES;
        }
        bool operator!=(const objc::selector &s) {
            return ::sel_isEqual(sel, s.sel) == NO;
        }
        
        std::string name() {
            return std::string(::sel_getName(sel));
        }
        
        operator ::SEL() { return sel; }
        operator std::string() { return name(); }
        operator const char*() { return ::sel_getName(sel); }
        operator char*() { return const_cast<char*>(::sel_getName(sel)); }
        
        static objc::selector register_name(const std::string &name) {
            return objc::selector(name);
        }
        static objc::selector register_name(const char *name) {
            return objc::selector(name);
        }
        
    };
    
    objc::selector operator"" _SEL(const char *name) {
        return objc::selector(name);
    }
    
    template <typename ...Args>
    struct arguments {
        static constexpr std::size_t N = sizeof...(Args);
        using is_argument_list = std::true_type;
        using index_type = std::make_index_sequence<N>;
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
                return objc_msgSend(self, op,
                    std::get<I>(std::forward<Args>(args))...);
            }
        
        public:
            ::id send(::id self, ::SEL op) const {
                return send_impl(self, op, index_type());
            }
        
        private:
            arguments(const arguments&);
            arguments(arguments&&);
            arguments &operator=(const arguments&);
            arguments &operator=(arguments&&);
    };
    
    template <typename ...Args>
    struct message : public arguments<Args...> {
        using arguments_type = arguments<Args...>;
        using arguments_type::argc;
        using arguments_type::args;
        ::id self;
        ::SEL op;
        
        explicit message(::id s, ::SEL o, Args&&... a)
            :arguments_type(a...)
            ,self(s), op(o)
            {}
        
        ::id send() const {
            return send(self, op);
        }
        
        private:
            message(const message&);
            message(message&&);
            message &operator=(const message&);
            message &operator=(message&&);
            
    };
    
    namespace traits {
    
        namespace detail {
            
            template <typename T, typename ...Args>
            static auto test_is_argument_list(int) -> typename T::is_argument_list;
            template <typename, typename ...Args>
            static auto test_is_argument_list(long) -> std::false_type;
            
        }
        
        #define TEST_ARGS T
        
        template <typename T>
        struct is_argument_list : decltype(detail::test_is_argument_list<T, TEST_ARGS>(0)) {
            template <typename X = std::enable_if<decltype(detail::test_is_argument_list<T, TEST_ARGS>(0))::value>>
            static constexpr bool value() { return true; }
            static constexpr bool value() {
                // static_assert(decltype(detail::test_is_argument_list<T, TEST_ARGS>(0))::value,
                //               "Type does not conform to objc::arguments<Args...>");
                return detail::test_is_argument_list<T, TEST_ARGS>(0);
            }
        };
        
        #undef TEST_ARGS
    
    }
    
    struct msg {
        
        ::id self;
        ::SEL op;
        
        explicit msg(::id s, ::SEL o)
            :self(s), op(o)
            {}
        
        template <typename ...Args>
        ::id send(Args&& ...args) {
            arguments<Args...> ARGS(args...);
            return ARGS.send(self, op);
        }
        
        template <typename M,
                  typename X = std::enable_if_t<traits::is_argument_list<M>::value()>>
        ::id sendv(M arg_list) {
            return arg_list.send(self, op);
        }
        
        template <typename ...Args>
        static ::id send(::id self, ::SEL op, Args&& ...args) {
            arguments<Args...> ARGS(args...);
            return ARGS.send(self, op);
        }
        
        template <typename M,
                  typename X = std::enable_if_t<traits::is_argument_list<M>::value()>>
        static ::id sendv(::id self, ::SEL op, M arg_list) {
            return arg_list.send(self, op);
        }
        
        private:
            msg(const msg&);
            msg(msg&&);
            msg &operator=(const msg&);
            msg &operator=(msg&&);
        
    };
    
}

#endif /// LIBIMREAD_OBJC_RT_HH