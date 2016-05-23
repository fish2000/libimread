/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_SELECTOR_HH
#define LIBIMREAD_OBJC_RT_SELECTOR_HH

#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <type_traits>
#include "types.hh"

namespace objc {
    
    /// Straightforward wrapper around an objective-c selector (the SEL type).
    /// + Constructable from, and convertable to, common string types
    /// + Overloaded for equality testing
    
    struct selector {
        
        types::selector sel;
        
        explicit selector(const std::string& name);
        explicit selector(const char* name);
        explicit selector(NSString* name);
        
        selector(types::selector s);
        selector(const objc::selector& other);
        selector(objc::selector&& other) noexcept;
        
        objc::selector& operator=(const objc::selector& other);
        objc::selector& operator=(types::selector other);
        
        bool operator==(const objc::selector& s) const;
        bool operator!=(const objc::selector& s) const;
        bool operator==(const types::selector& s) const;
        bool operator!=(const types::selector& s) const;
        
        const char* c_str() const;
        std::string str() const;
        NSString* ns_str() const;
        CFStringRef cf_str() const;
        
        friend std::ostream& operator<<(std::ostream& os, const objc::selector& s) {
            return os << "@selector( " << s.str() << " )";
        }
        
        std::size_t hash() const;
        void swap(objc::selector& other) noexcept;
        void swap(types::selector& other) noexcept;
        
        operator types::selector() const;
        operator std::string() const;
        operator const char*() const;
        operator char*() const;
        operator NSString*() const;
        operator CFStringRef() const;
        
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

objc::selector operator"" _SEL(const char* name);

#endif /// LIBIMREAD_OBJC_RT_SELECTOR_HH