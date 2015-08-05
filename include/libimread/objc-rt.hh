/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_HH
#define LIBIMREAD_OBJC_RT_HH

#include <cstdlib>
#include <string>
#include <tuple>
#include <type_traits>

#ifdef __OBJC__
#import <objc/message.h>
#import <objc/runtime.h>
#endif

namespace objc {
    
    struct id {
    
        ::id iid;
        
        explicit id(::id ii)
            :iid(ii)
            {}
        
        operator ::id() { return iid; }
        
        inline const char * __cls_name() const          { return ::object_getClassName(iid); }
        inline const char * __cls_name(::id ii) const   { return ::object_getClassName(ii); }
        
        std::string classname() {
            return std::string(__cls_name());
        }
        
        ::Class lookupclass() {
            return ::objc_lookUpClass(__cls_name());
        }
        ::Class getclass() {
            return ::objc_getClass(__cls_name());
        }
        
        static std::string classname(::id ii) {
            return std::string(__cls_name(ii));
        }
        
        static ::Class lookup(::id ii) {
            return ::objc_lookUpClass(__cls_name(ii));
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
        bool operator==(const ::SEL &s) {
            return ::sel_isEqual(sel, s) == YES;
        }
        bool operator!=(const ::SEL &s) {
            return ::sel_isEqual(sel, s) == NO;
        }
        
        inline const char *c_str() const {
            return ::sel_getName(sel);
        }
        inline std::string str() const {
            return std::string(c_str());
        }
        
        operator ::SEL() { return sel; }
        operator std::string() { return str(); }
        operator const char*() { return c_str(); }
        operator char*() { return const_cast<char*>(c_str()); }
        
        static objc::selector register_name(const std::string &name) {
            return objc::selector(name);
        }
        static objc::selector register_name(const char *name) {
            return objc::selector(name);
        }
        
        private:
            selector(void);
        
    };
    
    template <typename Return, typename ...Args>
    struct arguments {
        static constexpr std::size_t argc = sizeof...(Args);
        using is_argument_list = std::true_type;
        using index_type = std::make_index_sequence<argc>;
        using tuple_type = std::tuple<Args...>;
        using send_function = typename std::add_pointer<Return(::id, ::SEL, Args...)>::type;
        
        tuple_type args;
        send_function dispatcher = (send_function)objc_msgSend;
        
        explicit arguments(Args&&... a)
            :args(std::forward_as_tuple(a...))
            {}
        
        private:
            template <std::size_t ...I>
            Return send_impl(::id self, ::SEL op, std::index_sequence<I...>) {
                return dispatcher(self, op, std::get<I>(std::forward<tuple_type>(args))...);
            }
        
        public:
            Return send(::id self, ::SEL op) {
                return send_impl(self, op, index_type());
            }
        
        private:
            arguments(const arguments&);
            arguments(arguments&&);
            arguments &operator=(const arguments&);
            arguments &operator=(arguments&&);
    };
    
    template <typename Return, typename ...Args>
    struct message : public arguments<Return, Args...> {
        using arguments_type = arguments<Return, Args...>;
        using arguments_type::argc;
        using arguments_type::args;
        ::id self;
        ::SEL op;
        
        explicit message(::id s, ::SEL o, Args&&... a)
            :arguments_type(a...)
            ,self(s), op(o)
            {}
        
        Return send() const {
            return arguments_type::send(self, op);
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
        ::id send(::BOOL dispatch, Args&& ...args) {
            arguments<::id, Args...> ARGS(args...);
            return ARGS.send(self, op);
        }
        
        template <typename M,
                  typename X = std::enable_if_t<traits::is_argument_list<M>::value()>>
        ::id sendv(M arg_list) const {
            return arg_list.send(self, op);
        }
        
        template <typename Return, typename ...Args>
        static Return get(::id self, ::SEL op, Args&& ...args) {
            arguments<Return, Args...> ARGS(std::forward<Args>(args)...);
            return ARGS.send(self, op);
        }
        
        template <typename ...Args>
        static ::id send(::id self, ::SEL op, Args&& ...args) {
            return objc::msg::get<::id, Args...>(self, op, std::forward<Args>(args)...);
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

inline objc::selector operator"" _SEL(const char *name) {
    return objc::selector(name);
}

#endif /// LIBIMREAD_OBJC_RT_HH