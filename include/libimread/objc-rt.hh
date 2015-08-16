/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_HH
#define LIBIMREAD_OBJC_RT_HH

#include <cstdlib>
#include <algorithm>
#include <string>
#include <tuple>
#include <array>
#include <functional>
#include <type_traits>

#include <libimread/ext/pystring.hh>

#ifdef __OBJC__
#import <libimread/ext/categories/NSString+STL.hh>
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <objc/message.h>
#import <objc/runtime.h>
#endif /// __OBJC__

namespace objc {
    
    /// pointer swap
    template <typename T>
    void ptr_swap(T*& oA, T*& oB) {
        T* oT = oA; oA = oB; oB = oT;
    }
    
    /// bridge cast
    template <typename T, typename U>
    __attribute__((__always_inline__))
    T bridge(U castee) { return (__bridge T)castee; }
    
    namespace types {
        
        using baseID = ::id;
        using baseSEL = ::SEL;
        
        using rID = std::add_rvalue_reference_t<::id>;
        using xID = std::add_lvalue_reference_t<::id>;
        using tID = std::remove_reference_t<rID>;
        
        using rSEL = std::add_rvalue_reference_t<::SEL>;
        using xSEL = std::add_lvalue_reference_t<::SEL>;
        using tSEL = std::remove_reference_t<rSEL>;
        
        inline decltype(auto) pass_id(rID r)          { return std::forward<tID>(r); }
        inline decltype(auto) pass_selector(rSEL r)   { return std::forward<tSEL>(r); }
        inline decltype(auto) pass_id(xID r)          { return std::forward<tID>(r); }
        inline decltype(auto) pass_selector(xSEL r)   { return std::forward<tSEL>(r); }
        
    }
    
    /// function-pointer templates for wrapping objc_msgSend() with a bunch of possible sigs
    template <typename Return, typename ...Args>
    using return_sender_t = std::add_pointer_t<Return(types::tID, types::tSEL, Args...)>;
    
    template <typename ...Args>
    using void_sender_t = std::add_pointer_t<void(types::tID, types::tSEL, Args...)>;
    
    template <typename ...Args>
    using object_sender_t = std::add_pointer_t<::id(types::tID, types::tSEL, Args...)>;
    
    struct id {
        
        ::id iid;
        
        explicit id(types::rID ii)
            :iid(std::forward<types::tID>(ii))
            {}
        
        explicit id(types::xID ii)
            :iid(ii)
            {}
        
        operator ::id()     const { return iid; }
        ::id operator*()    const { return iid; }
        ::id operator->()   const { return iid; }
        
        inline const char * __cls_name() const      { return ::object_getClassName(iid); }
        static const char * __cls_name(::id ii)     { return ::object_getClassName(ii); }
        
        std::string classname() const {
            return std::string(__cls_name(iid));
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
        
        static ::Class lookup(types::rID ii) {
            return ::objc_lookUpClass(
                __cls_name(std::forward<types::tID>(ii)));
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
        using index_t = std::make_index_sequence<argc>;
        using tuple_t = std::tuple<Args...>;
        using sender_t = return_sender_t<Return, Args...>;
        using prebound_t = std::function<Return(types::tID, types::tSEL, Args...)>;
        
        tuple_t args;
        prebound_t dispatcher;
        sender_t dispatch_with = (sender_t)objc_msgSend;
        
        explicit arguments(Args... a)
            :args(std::forward_as_tuple(a...))
            {}
        
        private:
            template <std::size_t ...I> inline
            Return send_impl(types::tID self, types::tSEL op, std::index_sequence<I...>) {
                return dispatcher(self, op, std::get<I>(args)...);
            }
        
        public:
            Return send(types::rID self, types::rSEL op) {
                dispatcher = (prebound_t)dispatch_with;
                return send_impl(self, op, index_t());
            }
            Return send(types::tID self, types::tSEL op) {
                dispatcher = (prebound_t)dispatch_with;
                return send_impl(self, op, index_t());
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
            
            template <typename From, typename To>
            using is_convertible_t = std::conditional_t<std::is_convertible<From, To>::value,
                                                        std::true_type, std::false_type>;
            
            template <typename T, typename ...Args>
            static auto test_is_argument_list(int) -> typename T::is_argument_list;
            template <typename, typename ...Args>
            static auto test_is_argument_list(long) -> std::false_type;
            
        }
        
        template <typename T>
        struct is_argument_list : decltype(detail::test_is_argument_list<T>(0)) {
            template <typename X = std::enable_if<decltype(detail::test_is_argument_list<T>(0))::value>>
            static constexpr bool value() { return true; }
            static constexpr bool value() {
                static_assert(decltype(detail::test_is_argument_list<T>(0))::value,
                              "Type does not conform to objc::arguments<Args...>");
                return detail::test_is_argument_list<T>(0);
            }
        };
        
        template <typename T, typename V = bool>
        struct is_class : std::false_type {};
        template <typename T>
        struct is_class<T,
            typename std::enable_if_t<
                 std::is_convertible<T, objc::types::tID>::value,
                 bool>> : std::true_type {};
        
    }
    
    struct msg {
        
        using rID = types::rID;
        using rSEL = types::rSEL;
        
        objc::id myself;
        rID self;
        rSEL op;
        
        explicit msg(rID s, rSEL o)
            :self(types::pass_id(s))
            ,myself(s)
            ,op(types::pass_selector(o))
            {
                [*myself retain];
            }
        
        virtual ~msg() {
            [*myself release];
        }
        
        template <typename ...Args>
        ::id send(::BOOL dispatch, Args ...args) {
            arguments<::id, Args...> ARGS(args...);
            return ARGS.send(self, op);
        }
        
        template <typename M,
                  typename X = std::enable_if_t<traits::is_argument_list<M>::value()>>
        ::id sendv(M&& arg_list) const {
            return arg_list.send(self, op);
        }
        
        template <typename Return, typename ...Args>
        static Return get(types::tID self, types::tSEL op, Args ...args) {
            arguments<Return, Args...> ARGS(args...);
            const objc::id selfie(self);
            [*selfie retain];
            return ARGS.send(self, op);
            [*selfie release];
        }
        
        template <typename ...Args>
        static ::id send(types::tID self, types::tSEL op, Args ...args) {
            return objc::msg::get<types::tID, Args...>(self, op, args...);
        }
        
        template <typename M,
                  typename X = std::enable_if_t<traits::is_argument_list<M>::value()>>
        static ::id sendv(rID self, rSEL op, M&& arg_list) {
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

namespace im {
    
    namespace {
        
        /// these functions are so terrible,
        /// they don't even deserve a name for their namespace.
        /// ... OOH BURN
        static constexpr unsigned int len = 3;
        static const std::array<std::string, len> stringnames{
            "nsstring",
            "nsmutablestring",
            "nsattributedstring"
        };
        
        bool is_stringishly_named(const std::string &name) {
            bool out = false;
            std::string lowername = pystring::lower(name);
            std::for_each(stringnames.begin(), stringnames.end(), [&](std::string nm) {
                out = out || nm == lowername;
            });
            return out;
        }
        
    }
    
    /// q.v. libimread/errors.hh, lines 45-90 (aprox., subject to change) --
    ///      ... The other overload-resolution-phase versions of `stringify()` are
    ///      defined therein. This one gets enable-if'ed when anyone tries to use the 
    ///      debug output funcs and macros from errors.hh to print an NSObject subclass.
    ///      ... the current laughable implementation can just* get extended at any time
    ///      with more dynamic whatever-the-fuck type serialization provisions as needed.
    ///
    ///      *) See also http://bit.ly/1P8d8va for in-depth analysis of this pivotal term
    template <typename S> inline
    typename std::enable_if_t<objc::traits::is_class<S>::value, std::string>
        stringify(S *s) {
            const objc::id self(s);
            if (is_stringishly_named(self.classname())) {
                return [*self STLString];
            }
            return [[*self description] STLString];
        }
    
    
}

#endif /// LIBIMREAD_OBJC_RT_HH