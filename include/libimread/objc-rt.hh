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
    
    /// pointer swapper
    template <typename T> inline
    void swap(T*& oA, T*& oB) {
        T* oT = oA; oA = oB; oB = oT;
    }
    
    /// bridge cast
    template <typename T, typename U>
    __attribute__((__always_inline__))
    T bridge(U castee) { return (__bridge T)castee; }
    
    /// bridge-retain cast --
    /// kind of a NOOp in MRC-mode
    /// (which currently we don't do ARC anyway)
    template <typename T, typename U>
    __attribute__((__always_inline__))
    T bridgeretain(U castee) {
        #if __has_feature(objc_arc)
            return (__bridge_retained T)castee;
        #else
            return (__bridge T)castee;
        #endif
    }
    
    /// block type alias --
    /// because `objc::block<T> thing`
    /// looks better than `__block T` 
    
    template <typename Type>
    using block_t = Type __attribute__((__blocks__(byref)));
    
    template <typename Type>
    using block = __block block_t<typename std::remove_cv<Type>::type>;
    
    /// namespaced references,
    /// for everything we use from the objective-c type system
    
    namespace types {
        
        using ID = ::id __attribute__((NSObject));
        using selector = ::SEL;
        using cls = ::Class;
        using boolean = ::BOOL;
        
        using rID = std::add_rvalue_reference_t<ID>;
        using xID = std::add_lvalue_reference_t<ID>;
        using tID = std::remove_reference_t<rID>;
        
        using rSEL = std::add_rvalue_reference_t<selector>;
        using xSEL = std::add_lvalue_reference_t<selector>;
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
    
    /// Straightforward wrapper around an objective-c selector (the SEL type).
    /// + Constructable from, and convertable to, common string types
    /// + Overloaded for equality testing
    
    struct selector {
        
        types::selector sel;
        
        explicit selector(types::selector s)
            :sel(s)
            {}
        explicit selector(const std::string &name)
            :sel(::sel_registerName(name.c_str()))
            {}
        explicit selector(const char *name)
            :sel(::sel_registerName(name))
            {}
        explicit selector(NSString *name)
            :sel(::NSSelectorFromString(name))
            {}
        
        bool operator==(const objc::selector &s) {
            return ::sel_isEqual(sel, s.sel) == YES;
        }
        bool operator!=(const objc::selector &s) {
            return ::sel_isEqual(sel, s.sel) == NO;
        }
        bool operator==(const types::selector &s) {
            return ::sel_isEqual(sel, s) == YES;
        }
        bool operator!=(const types::selector &s) {
            return ::sel_isEqual(sel, s) == NO;
        }
        
        inline const char *c_str() const {
            return ::sel_getName(sel);
        }
        inline std::string str() const {
            return std::string(c_str());
        }
        
        operator types::selector() { return sel; }
        operator std::string() { return str(); }
        operator const char*() { return c_str(); }
        operator char*() { return const_cast<char*>(c_str()); }
        
        static objc::selector register_name(const std::string &name) {
            return objc::selector(name);
        }
        static objc::selector register_name(const char *name) {
            return objc::selector(name);
        }
        static objc::selector register_name(NSString *name) {
            return objc::selector(name);
        }
        
        private:
            selector(void);
        
    };
    
    /// variadic tuple-unpacking argument wrapper structure
    /// for referencing objective-c message arguments passed
    /// to objc_msgSend and its multitudinous variants.
    /// ... users really shouldn't need to invoke this directly;
    /// use  `objc::msg` instead, see its definition and notes below.
    
    template <typename Return, typename ...Args>
    struct arguments {
        static constexpr std::size_t argc = sizeof...(Args);
        using is_argument_list_t = std::true_type;
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
        types::ID self;
        
        explicit message(types::ID s, types::selector o, Args&&... a)
            :arguments_t(a...)
            ,self(s), op(o)
            {
                [self retain];
            }
        
        virtual ~message() {
            [self release];
        }
        
        Return send() const {
            return arguments_t::send(self, op);
        }
        
        private:
            message(const message&);
            message(message&&);
            message &operator=(const message&);
            message &operator=(message&&);
            
    };
    
    /// wrapper around an objective-c instance
    /// ... FEATURING:
    /// + automatic scoped memory management via RAII through MRC messages
    /// ... plus fine control through inlined retain/release/autorelease methods
    /// + access to wrapped object pointer via operator*()
    /// + boolean selector-response test via operator[](T t) e.g.
    ///
    ///     objc::id yodogg([[NSYoDogg alloc] init]);
    ///     if (yodogg[@"iHeardYouRespondTo:"]) {
    ///         [*yodogg iHeardYouRespondTo:argsInYourArgs];
    ///     }
    ///
    /// + convenience methods e.g. yodogg.classname(), yodogg.description(), yodogg.lookup() ...
    /// + inline bridging template e.g. void* asVoid = yodogg.bridge<void*>();
    /// + E-Z static methods for looking shit up in the runtime heiarchy
    
    struct id {
        
        types::ID self = nil;
        
        explicit id(types::rID ii)
            :self(std::forward<types::tID>(ii))
            {
                retain();
            }
        
        explicit id(types::xID ii)
            :self(ii)
            {
                retain();
            }
        
        id(const objc::id& other)
            :self(other.self)
            {
                retain();
            }
        
        id(objc::id&& other)
            :self(other.self)
            {
                retain();
                //other.release();
            }
        
        ~id() { release(); }
        
        id &operator=(const objc::id& other) {
            types::ID s = other.self;
            if (s != nil)    { [s retain]; }
            if (self != nil) { [self release]; }
            self = s;
            return *this;
        }
        
        id &operator=(objc::id&& other) {
            types::ID s = other.self;
            if (s != nil)    { [s retain]; }
            if (self != nil) { [self release]; }
            self = s;
            //other.release();
            return *this;
        }
        
        operator types::ID()     const { return self; }
        types::ID operator*()    const { return self; }
        types::ID operator->()   const { return self; }
        
        template <typename T> inline
        T bridge() {
            return objc::bridge<T>(self);
        }
        
        inline bool responds_to(types::selector s) const {
            return [self respondsToSelector:s] == YES;
        }
        
        inline void retain() const          { if (self != nil) { [self retain]; } }
        inline void release() const         { if (self != nil) { [self release]; } }
        inline void autorelease() const     { if (self != nil) { [self autorelease]; } }
        
        template <typename ...Args>
        types::ID operator()(types::selector s, Args&&... args) {
            arguments<types::ID, Args...> ARGS(std::forward<Args>(args)...);
            retain();
            types::ID out = ARGS.send(self, s);
            release();
            return out;
        }
        
        bool operator[](types::selector s) const      { return responds_to(s); }
        bool operator[](const char *s) const          { return responds_to(::sel_registerName(s)); }
        bool operator[](const std::string &s) const   { return responds_to(::sel_registerName(s.c_str())); }
        bool operator[](NSString *s) const            { return responds_to(::NSSelectorFromString(s)); }
        
        inline const char * __cls_name() const          { return ::object_getClassName(self); }
        static const char * __cls_name(types::ID ii)    { return ::object_getClassName(ii); }
        
        std::string classname() const {
            return std::string(__cls_name(self));
        }
        
        std::string description() const {
            return [[self description] STLString];
        }
        
        types::cls lookup() const {
            return ::objc_lookUpClass(__cls_name());
        }
        types::cls getclass() const {
            return ::objc_getClass(__cls_name());
        }
        
        /// STATIC METHODS
        static std::string classname(types::ID ii) {
            return std::string(__cls_name(ii));
        }
        
        static std::string description(types::ID ii) {
            return [[ii description] STLString];
        }
        
        static types::cls lookup(types::rID ii) {
            return ::objc_lookUpClass(
                __cls_name(std::forward<types::tID>(ii)));
        }
        static types::cls lookup(const std::string &s) {
            return ::objc_lookUpClass(s.c_str());
        }
        static types::cls lookup(const char *s) {
            return ::objc_lookUpClass(s);
        }
        static types::cls lookup(NSString *s) {
            return ::objc_lookUpClass([s UTF8String]);
        }
        
        private:
            id(void);
    };
    
    
    namespace traits {
    
        namespace detail {
            
            template <typename From, typename To>
            using is_convertible_t = std::conditional_t<std::is_convertible<From, To>::value,
                                                        std::true_type, std::false_type>;
            
            template <typename T, typename ...Args>
            static auto test_is_argument_list(int) -> typename T::is_argument_list_t;
            template <typename, typename ...Args>
            static auto test_is_argument_list(long) -> std::false_type;
            
        }
        
        /// Unnecessarily overwrought compile-time test for objc::message and descendants
        
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
        
        /// compile-time tests for objective-c primitives:
        
        /// test for an object-pointer instance (NSObject* and descendants)
        template <typename T, typename V = bool>
        struct is_object : std::false_type {};
        template <typename T>
        struct is_object<T,
            typename std::enable_if_t<
                 std::is_convertible<T, objc::types::ID>::value,
                 bool>> : std::true_type {};
        
        /// test for a selector struct
        template <typename T, typename V = bool>
        struct is_selector : std::false_type {};
        template <typename T>
        struct is_selector<T,
            typename std::enable_if_t<
                std::is_convertible<T, objc::types::selector>::value,
                bool>> : std::true_type {};
        
        /// test for the objective-c class struct type
        template <typename T, typename V = bool>
        struct is_class : std::false_type {};
        template <typename T>
        struct is_class<T,
            typename std::enable_if_t<
                std::is_convertible<T, objc::types::cls>::value,
                bool>> : std::true_type {};
        
    }
    
    struct msg {
        
        objc::id self; /// scoped retain/release
        types::rID selfref;
        types::rSEL op;
        
        explicit msg(types::rID s, types::rSEL o)
            :self(objc::id(s))
            ,selfref(types::pass_id(s))
            ,op(types::pass_selector(o))
            {}
        
        template <typename ...Args>
        types::ID send(types::boolean dispatch, Args ...args) {
            arguments<types::ID, Args...> ARGS(args...);
            return ARGS.send(selfref, op);
        }
        
        template <typename M,
                  typename X = std::enable_if_t<traits::is_argument_list<M>::value()>>
        types::ID sendv(M&& arg_list) const {
            return arg_list.send(selfref, op);
        }
        
        template <typename Return, typename ...Args>
        static Return get(types::tID s, types::tSEL op, Args ...args) {
            arguments<Return, Args...> ARGS(args...);
            const objc::id selfie(s); /// for scoped retain/release
            return ARGS.send(s, op);
        }
        
        template <typename ...Args>
        static types::ID send(types::tID s, types::tSEL op, Args ...args) {
            return objc::msg::get<types::tID, Args...>(s, op, args...);
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
    
}

/// string suffix for inline declaration of objc::selector objects
/// ... e.g. create an inline wrapper for a `yoDogg:` selector like so:
///     objc::selector yodogg = "yoDogg:"_SEL;

inline objc::selector operator"" _SEL(const char *name) {
    return objc::selector(name);
}

namespace im {
    
    /// q.v. libimread/errors.hh, lines 45-90 (aprox., subject to change) --
    ///      ... The other overload-resolution-phase versions of `stringify()` are
    ///      defined therein. This one gets enable-if'ed when anyone tries to use the 
    ///      debug output funcs and macros from errors.hh to print an NSObject subclass.
    ///      ... the current laughable implementation can just* get extended at any time
    ///      with more dynamic whatever-the-fuck type serialization provisions as needed.
    ///
    ///      *) See also http://bit.ly/1P8d8va for in-depth analysis of this pivotal term
    
    template <typename S> inline
    typename std::enable_if_t<objc::traits::is_object<S>::value, std::string>
        stringify(S *s) {
            const objc::id self(s);
            if (self[@"STLString"]) {
                return [*self STLString];
            }
            return self.description();
        }
    
}

#endif /// LIBIMREAD_OBJC_RT_HH