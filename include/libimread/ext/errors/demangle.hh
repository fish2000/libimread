/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Adapted from http://stackoverflow.com/a/31633962/298171

#pragma once
#include <typeinfo>

namespace terminator {
    
    /// static mutex ensures we only are demangle()-ing
    /// one thing at a time, because the implement uses a bunch of
    /// private global stuff and I don't feel like rewriting it
    // static std::mutex mangle_barrier;
    
    /// actual function to demangle an allegedly mangled thing
    char const* demangle(char const* const symbol) noexcept;
    
    /// convenience function to get the name of a type
    template <typename Arg>
    char const* nameof(Arg argument) {
        try {
            return demangle(typeid(argument).name());
        } catch (std::bad_typeid const&) {
            return "<unknown>";
        }
    }
    
} /* namespace terminator */
