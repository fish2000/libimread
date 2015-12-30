/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_SELECTOR_HH
#define LIBIMREAD_OBJC_RT_SELECTOR_HH

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
        selector(objc::selector&& other) noexcept
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
        
        friend std::ostream& operator<<(std::ostream& os, const objc::selector& s) {
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
    
} /* namespace objc */

/// string suffix for inline declaration of objc::selector objects
/// ... e.g. create an inline wrapper for a `yoDogg:` selector like so:
///     objc::selector yodogg = "yoDogg:"_SEL;

inline objc::selector operator"" _SEL(const char* name) {
    return objc::selector(name);
}


#endif /// LIBIMREAD_OBJC_RT_SELECTOR_HH