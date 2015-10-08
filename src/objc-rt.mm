/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/objc-rt.hh>

namespace objc {
    
    // __attribute__((__always_inline__))
    types::boolean boolean(bool value) { return types::boolean(value ? YES : NO); }
    
    // __attribute__((__always_inline__))
    bool to_bool(types::boolean value) { return bool(value == YES); }
    
}

namespace std {
    
    template <>
    void swap(objc::selector& s0, objc::selector& s1) {
        s0.swap(s1);
    }
    
    template <>
    void swap(objc::id& s0, objc::id& s1) {
        s0.swap(s1);
    }
    
    
}