/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_NAMESPACE_IM_HH
#define LIBIMREAD_OBJC_RT_NAMESPACE_IM_HH

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
#include "selector.hh"
#include "message-args.hh"
#include "traits.hh"
#include "object.hh"
#include "message.hh"

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

#endif /// LIBIMREAD_OBJC_RT_NAMESPACE_IM_HH