/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Adapted from http://stackoverflow.com/a/31633962/298171

#include <libimread/ext/errors/demangle.hh>

#include <cstdlib>
#include <cxxabi.h>
#include <mutex>
#include <memory>

namespace terminator {
    
    namespace {
        
        /// define the (already-declared) static mutex,
        /// to keep the demangler from reentering itself
        /// ... which that does sound dirty, actually
        static std::mutex mangle_barrier;
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wglobal-constructors"
        #pragma clang diagnostic ignored "-Wexit-time-destructors"
        std::unique_ptr<char, decltype(std::free)&> demangled_name{ nullptr, std::free };
        #pragma clang diagnostic pop
        
    }
    
    char const* demangle(char const* const symbol) noexcept {
        if (!symbol) { return "<null>"; }
        std::lock_guard<std::mutex> lock(mangle_barrier);
        int status = -4;
        demangled_name.reset(
            abi::__cxa_demangle(symbol,
                                demangled_name.get(),
                                nullptr, &status));
        return ((status == 0) ? demangled_name.release() : symbol);
    }

} /* namespace terminator */
