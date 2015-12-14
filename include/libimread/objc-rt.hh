/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_HH
#define LIBIMREAD_OBJC_RT_HH

#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include <array>
#include <utility>
#include <functional>
#include <type_traits>

#include <libimread/ext/pystring.hh>

#ifdef __APPLE__
#import <libimread/ext/categories/NSString+STL.hh>
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <objc/message.h>
#import <objc/runtime.h>
#endif /// __APPLE__

namespace objc {
    
    /// pointer swap
    template <typename T> inline
    void swap(T*& oA, T*& oB) noexcept {
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
    /// looks better than `__block T thing` IN MY HUMBLE OPINION
    
    template <typename Type>
    using block_t = Type
        __attribute__((__blocks__(byref)));
    
    template <typename Type>
    using block = __block block_t<typename std::remove_cv<Type>::type>
        __attribute__((__always_inline__));
    
    /// namespaced references,
    /// for everything we use from the objective-c type system
    
    namespace types {
        
        using ID = ::id __attribute__((NSObject));
        using object_t = struct ::objc_object;
        using selector = ::SEL;
        using cls = ::Class;
        using method = ::Method;
        using implement = ::IMP;
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
    using return_sender_t = std::add_pointer_t<Return(types::ID, types::selector, Args...)>;
    
    template <typename ...Args>
    using void_sender_t = std::add_pointer_t<void(types::ID, types::selector, Args...)>;
    
    template <typename ...Args>
    using object_sender_t = std::add_pointer_t<types::ID(types::ID, types::selector, Args...)>;
    
    /// objc::boolean(bool_value) -> YES or NO
    /// objc::to_bool(BOOL_value) -> true or false
    
    __attribute__((__always_inline__)) types::boolean boolean(bool value);
    __attribute__((__always_inline__)) bool to_bool(types::boolean value);
    
    /// Straightforward wrapper around an objective-c selector (the SEL type).
    /// + Constructable from, and convertable to, common string types
    /// + Overloaded for equality testing
    
    struct selector {
        
        types::selector sel;
        
        explicit selector(const std::string& name)
            :sel(::sel_registerName(name.c_str()))
            {}
        explicit selector(const char* name)
            :sel(::sel_registerName(name))
            {}
        explicit selector(NSString* name)
            :sel(::NSSelectorFromString(name))
            {}
        
        selector(types::selector s)
            :sel(s)
            {}
        selector(const objc::selector& other)
            :sel(other.sel)
            {}
        selector(objc::selector&& other)
            :sel(other.sel)
            {}
        
        objc::selector& operator=(const objc::selector& other) {
            objc::selector(other).swap(*this);
            return *this;
        }
        objc::selector& operator=(types::selector other) {
            objc::selector(other).swap(*this);
            return *this;
        }
        
        bool operator==(const objc::selector& s) const {
            return objc::to_bool(::sel_isEqual(sel, s.sel));
        }
        bool operator!=(const objc::selector& s) const {
            return !objc::to_bool(::sel_isEqual(sel, s.sel));
        }
        bool operator==(const types::selector& s) const {
            return objc::to_bool(::sel_isEqual(sel, s));
        }
        bool operator!=(const types::selector& s) const {
            return !objc::to_bool(::sel_isEqual(sel, s));
        }
        
        inline const char* c_str() const {
            return ::sel_getName(sel);
        }
        inline std::string str() const {
            return c_str();
        }
        inline NSString* ns_str() const {
            return ::NSStringFromSelector(sel);
        }
        inline CFStringRef cf_str() const {
            return objc::bridge<CFStringRef>(ns_str());
        }
        
        friend std::ostream &operator<<(std::ostream& os, const objc::selector& s) {
            return os << "@selector( " << s.str() << " )";
        }
        
        std::size_t hash() const {
            std::hash<std::string> hasher;
            return hasher(str());
        }
        
        void swap(objc::selector& other) noexcept {
            using std::swap;
            using objc::swap;
            swap(this->sel, other.sel);
        }
        void swap(types::selector& other) noexcept {
            using std::swap;
            using objc::swap;
            swap(this->sel, other);
        }
        
        operator types::selector() const { return sel; }
        operator std::string() const { return str(); }
        operator const char*() const { return c_str(); }
        operator char*() const { return const_cast<char*>(c_str()); }
        operator NSString*() const { return ::NSStringFromSelector(sel); }
        operator CFStringRef() const { return objc::bridge<CFStringRef>(ns_str()); }
        
        static objc::selector register_name(const std::string& name) {
            return objc::selector(name);
        }
        static objc::selector register_name(const char* name) {
            return objc::selector(name);
        }
        static objc::selector register_name(NSString* name) {
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
        types::ID self;
        
        explicit message(types::ID s, types::selector o, Args&&... a)
            :arguments_t(a...), self(s), op(o)
            {
                [self retain];
            }
        
        virtual ~message() { [self release]; }
        
        private:
            message(const message&);
            message(message&&);
            message &operator=(const message&);
            message &operator=(message&&);
            
    };
    
    namespace traits {
    
        namespace detail {
            
            /// convenience type struct that should already freaking exist
            /// to go along with std::is_convertible<From, To>
            
            template <typename From, typename To>
            using is_convertible_t = std::conditional_t<std::is_convertible<From, To>::value,
                                                        std::true_type, std::false_type>;
            
            /// detail test defs for overwrought argument-list check (see below)
            
            template <typename T, typename ...Args>
            static auto test_is_argument_list(int) -> typename T::is_argument_list_t;
            template <typename, typename ...Args>
            static auto test_is_argument_list(long) -> std::false_type;
            
            /// OLDSKOOL SFINAE 4 LYFE. Recipe courtesy WikiBooks:
            /// https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector
            /// ... to test for an Objective-C object, has_isa<PointerType> checks the struct
            /// at which it points for an `isa` member (see http://stackoverflow.com/q/1990695/298171)
            /// Now this is a budget way to SFINAE I know but that is cuz my SFINAE is
            /// STRAIGHT OUTTA COMPTON DOGG this one is FOR ALL MY ISAZ
            /// 
            /// (ahem)
            ///
            /// Also the has_isa class is itself enable_if'd only for classes because
            /// specializing it for fundamental types makes it all FREAK THE GEEK OUT
            
            template <typename U, U> struct check;
            typedef char one[1];
            typedef char two[2];
            
            template <typename Target,
                      typename T = std::remove_pointer_t<
                                   std::decay_t<Target>>>
            class has_isa {
                struct detect { int isa; };
                struct composite : T, detect {};
                template <typename U> static detail::one &test(
                                      detail::check<
                                          int detect::*,
                                          &U::isa>*);
                template <typename U> static detail::two &test(...);
                public:
                    typedef T type;
                    typedef Target pointer_type;
                    enum { value = sizeof(test<composite>(0)) == 2 };
            };
            
            template <typename Target,
                      typename T = std::remove_pointer_t<
                                   std::decay_t<Target>>>
            class has_superclass {
                struct detect { int superclass; };
                struct detect_ { int super_class; };
                struct composite : T, detect, detect_ {};
                template <typename U> static detail::one &test(
                                      detail::check<
                                          int detect::*,
                                          &U::superclass>*);
                template <typename U> static detail::one &test(
                                      detail::check<
                                          int detect_::*,
                                          &U::super_class>*);
                template <typename U> static detail::two &test(...);
                public:
                    typedef T type;
                    typedef Target pointer_type;
                    enum { value = sizeof(test<composite>(0)) == 2 };
            };
            
            /// All of this following hoohah is a SFINAE-compatible reimplementation
            /// of std::common_type<T>, taken right from this document:
            ///     http://open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3843.pdf
            template <typename T, typename U>
            using ct2 = decltype(std::declval<bool>() ? std::declval<T>() : std::declval<U>());
            
            template <typename T>
            using void_t = std::conditional_t<true, void, T>;
            
            template <class, class...>
            struct ct {};
            template <class T>
            struct ct<void, T> { using type = std::decay_t<T>; };
            template <class T, class U, class ...V>
            struct ct<void_t<ct2<T, U>>, T, U, V...> : ct<void, ct2<T, U>, V...> {};
            
            template <typename ...Types>
            struct common_type : ct<void, Types...> {};
            
            template <class, class = void>
            struct has_type_member : std::false_type {};
            template <class T>
            struct has_type_member<T, void_t<typename T::type>> : std::true_type {};
            
            template <typename ...Types>
            using common_type_t = typename common_type<Types...>::type;
            template <typename ...Types>
            using has_common_type = has_type_member<common_type<Types...>>;
            
            /// objc::traits::detail::is_object_pointer<T> checks using
            /// objc::traits::detail::has_common_type<T, objc::types::ID>
            template <typename Target>
            using is_object_pointer = has_common_type<std::decay_t<Target>, types::ID>;
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
        
        }; /* namespace detail */
        
        /// compile-time tests for objective-c primitives:
        
        /// test for an object-pointer instance (NSObject* and descendants)
        /// ... uses std::is_pointer<T> to enable_if itself properly,
        /// ... and objc::traits::detail::is_object_pointer<T> to make the call
        template <typename T, typename V = bool>
        struct is_object : std::false_type {};
        template <typename T>
        struct is_object<T,
            typename std::enable_if_t<
                 std::is_pointer<T>::value,
                 bool>> : detail::is_object_pointer<T> {};
        
        /// test for a selector struct
        template <typename T, typename V = bool>
        struct is_selector : std::false_type {};
        template <typename T>
        struct is_selector<T,
            typename std::enable_if_t<
                std::is_same<T, objc::types::selector>::value,
                bool>> : std::true_type {};
        
        /// test for the objective-c class struct type
        template <typename T, typename V = bool>
        struct is_class : std::false_type {};
        template <typename T>
        struct is_class<T,
            typename std::enable_if_t<
                detail::has_superclass<T>::value,
                bool>> : std::true_type {};
        
    } /* namespace traits */
    
    /// wrapper around an objective-c instance
    /// ... FEATURING:
    /// + automatic scoped memory management via RAII through MRC messages
    /// ... plus fine control through inlined retain/release/autorelease methods
    /// + access to wrapped object pointer via operator*()
    /// + boolean selector-response test via operator[](T t) e.g.
    ///
    ///     objc::object<NSYoDogg*> yodogg([[NSYoDogg alloc] init]);
    ///     if (yodogg[@"iHeardYouRespondTo:"]) {
    ///         [*yodogg iHeardYouRespondTo:argsInYourArgs];
    ///     }
    ///
    /// + convenience methods e.g. yodogg.classname(), yodogg.description(), yodogg.lookup() ...
    /// + inline bridging template e.g. void* asVoid = yodogg.bridge<void*>();
    /// + E-Z static methods for looking shit up in the runtime heiarchy
    
    template <typename OCType>
    struct object {
        using object_t = typename std::remove_pointer_t<std::decay_t<OCType>>;
        using pointer_t = OCType;
        
        static_assert(objc::traits::is_object<OCType>::value,
                      "objc::object<OCType> requires a pointer to objc_object");
        
        pointer_t self;
        
        explicit object(pointer_t ii)
            :self(ii)
            {
                retain();
            }
        
        object(const object& other)
            :self(other.self)
            {
                retain();
            }
        
        ~object() { release(); }
        
        object &operator=(const object& other) {
            if (this != &other) {
                object(other).swap(*this);
            }
            return *this;
        }
        
        object &operator=(pointer_t other) {
            if ([self isEqual:other] == NO) {
                object(other).swap(*this);
            }
            return *this;
        }
        
        operator pointer_t()   const { return self; }
        pointer_t operator*()  const { return self; }
        pointer_t operator->() const { return self; }
        
        bool operator==(const object& other) const {
            return objc::to_bool([self isEqual:other.self]);
        }
        bool operator!=(const object& other) const {
            return !objc::to_bool([self isEqual:other.self]);
        }
        bool operator==(const pointer_t& other) const {
            return objc::to_bool([self isEqual:other]);
        }
        bool operator!=(const pointer_t& other) const {
            return !objc::to_bool([self isEqual:other]);
        }
        
        template <typename T> inline
        T bridge() {
            return objc::bridge<T>(self);
        }
        
        inline bool responds_to(types::selector s) const {
            return objc::to_bool([self respondsToSelector:s]);
        }
        
        inline void retain() const      { if (self != nil) { [self retain]; } }
        inline void release() const     { if (self != nil) { [self release]; } }
        inline void autorelease() const { if (self != nil) { [self autorelease]; } }
        
        template <typename ...Args>
        types::ID operator()(types::selector s, Args... args) {
            arguments<types::ID, Args...> ARGS(args...);
            retain();
            types::ID out = ARGS.send(self, s);
            release();
            return out;
        }
        
        bool operator[](types::selector s) const       { return responds_to(s); }
        bool operator[](const objc::selector& s) const { return responds_to(s.sel); }
        bool operator[](const char* s) const           { return responds_to(::sel_registerName(s)); }
        bool operator[](const std::string& s) const    { return responds_to(::sel_registerName(s.c_str())); }
        bool operator[](NSString* s) const             { return responds_to(::NSSelectorFromString(s)); }
        bool operator[](CFStringRef s) const           { return responds_to(::NSSelectorFromString(
                                                                        objc::bridge<NSString*>(s))); }
        
        std::string classname() const {
            return [::NSStringFromClass([self class]) STLString];
        }
        
        std::string description() const {
            return [[self description] STLString];
        }
        
        friend std::ostream &operator<<(std::ostream &os, const object& friendly) {
            return os << "<" << friendly.classname()   << "> "
                      << "(" << friendly.description() << ") "
                      << "[" << std::hex << "0x"
                             << friendly.hash()
                             << std::dec << "]";
        }
        
        types::cls lookup() const {
            return ::objc_lookUpClass(::object_getClassName(self));
        }
        types::cls getclass() const {
            return ::objc_getClass(::object_getClassName(self));
        }
        
        std::size_t hash() const {
            return static_cast<std::size_t>([self hash]);
        }
        
        void swap(object& other) noexcept {
            using std::swap;
            using objc::swap;
            swap(this->self, other.self);
        }
        void swap(pointer_t& other) noexcept {
            using std::swap;
            using objc::swap;
            swap(this->self, other);
        }
        
        /// STATIC METHODS
        static std::string classname(pointer_t ii) {
            return [::NSStringFromClass([ii class]) STLString];
        }
        
        static std::string description(pointer_t ii) {
            return [[ii description] STLString];
        }
        
        static types::cls lookup(pointer_t&& ii) {
            return ::objc_lookUpClass(
                ::object_getClassName(std::forward<pointer_t>(ii)));
        }
        static types::cls lookup(const std::string& s) {
            return ::objc_lookUpClass(s.c_str());
        }
        static types::cls lookup(const char* s) {
            return ::objc_lookUpClass(s);
        }
        // static types::cls lookup(NSString* s) {
        //     return ::NSClassFromString(s);
        // }
        // static types::cls lookup(CFStringRef s) {
        //     return ::NSClassFromString(objc::bridge<NSString*>(s));
        // }
        
        private:
            object(void);
    };
    
    using id = objc::object<types::ID>;
    
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

/// string suffix for inline declaration of objc::selector objects
/// ... e.g. create an inline wrapper for a `yoDogg:` selector like so:
///     objc::selector yodogg = "yoDogg:"_SEL;

inline objc::selector operator"" _SEL(const char* name) {
    return objc::selector(name);
}

namespace std {
    
    /// std::swap() specialization for objc::selector and objc::id
    
    template <>
    void swap(objc::selector& s0, objc::selector& s1);
    
    template <>
    void swap(objc::id& s0, objc::id& s1);
    
    /// std::hash specializations for objc::selector and objc::id
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<objc::selector> {
        
        typedef objc::selector argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& selector) const {
            return static_cast<result_type>(selector.hash());
        }
        
    };
    
    template <>
    struct hash<objc::id> {
        
        typedef objc::id argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& instance) const {
            return static_cast<result_type>(instance.hash());
        }
        
    };
    
    template <>
    struct hash<objc::types::selector> {
        
        typedef objc::types::selector argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& selector) const {
            objc::selector s(selector);
            return static_cast<result_type>(s.hash());
        }
        
    };
    
    template <>
    struct hash<objc::types::ID> {
        
        typedef objc::types::ID argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& instance) const {
            return static_cast<result_type>([instance hash]);
        }
        
    };
    
}; /* namespace std */

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
    typename std::enable_if_t<objc::traits::is_object<S>::value,
        const std::string>
        stringify(S s) {
            const objc::id self(s);
            if (self[@"STLString"]) {
                return [*self STLString];
            } else if (self[@"UTF8String"]) {
                return [*self UTF8String];
            }
            return self.description();
        }
    
    template <typename S> inline
    typename std::enable_if_t<objc::traits::is_selector<S>::value,
        const std::string>
        stringify(S s) {
            const objc::selector sel(s);
            return sel.str();
        }
    
} /* namespace im */

#endif /// LIBIMREAD_OBJC_RT_HH