/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/objc-rt/namespace-std.hh>

namespace std {
    
    template <>
    void swap(objc::selector& s0, objc::selector& s1) {
        s0.swap(s1);
    }
    
    template <>
    void swap(objc::id& s0, objc::id& s1)
         noexcept(is_nothrow_move_constructible<objc::id>::value &&
                  is_nothrow_move_assignable<objc::id>::value) {
        s0.swap(s1);
    }
    
}