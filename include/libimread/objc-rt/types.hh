/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_TYPES_HH
#define LIBIMREAD_OBJC_RT_TYPES_HH

#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include <array>
#include <utility>
#include <functional>
#include <type_traits>

#ifdef __APPLE__
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
    
} /* namespace objc */


#endif /// LIBIMREAD_OBJC_RT_TYPES_HH