/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_NAMESPACE_STD_HH
#define LIBIMREAD_OBJC_RT_NAMESPACE_STD_HH

#include "types.hh"
#include "selector.hh"
#include "message-args.hh"
#include "traits.hh"
#include "object.hh"
#include "message.hh"

namespace std {
    
    /// std::swap() specialization for objc::selector and objc::id
    
    template <>
    void swap(objc::selector& s0, objc::selector& s1);
    
    template <>
    void swap(objc::id& s0, objc::id& s1)
         noexcept(is_nothrow_move_constructible<objc::id>::value &&
                  is_nothrow_move_assignable<objc::id>::value);
    
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

#endif /// LIBIMREAD_OBJC_RT_NAMESPACE_STD_HH