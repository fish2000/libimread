/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Adapted from http://stackoverflow.com/a/31633962/298171

#ifndef LIBIMREAD_EXT_ERRORS_TERMINATOR_HH_
#define LIBIMREAD_EXT_ERRORS_TERMINATOR_HH_

#include "demangle.hh"
#include "backtrace.hh"

#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <exception>
#include <memory>
#include <typeinfo>
#include <cxxabi.h>

namespace {
    
    __attribute__((noreturn))
    void backtrace_on_terminate() noexcept;
    
    // static_assert(std::is_same<std::terminate_handler,
    //                            decltype(&backtrace_on_terminate)>{},
    //                            "Type mismatch on return from backtrace_on_terminate()!");
    
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wglobal-constructors"
    #pragma clang diagnostic ignored "-Wexit-time-destructors"
    std::unique_ptr<std::remove_pointer_t<std::terminate_handler>,
                    decltype(std::set_terminate)&> terminate_handler {
                        std::set_terminate(std::get_terminate()),
                        std::set_terminate };
    #pragma clang diagnostic pop
    
    __attribute__((noreturn))
    void backtrace_on_terminate() noexcept {
        std::set_terminate(terminate_handler.release()); /// avoid infinite looping
        terminator::backtrace(std::clog);
        if (std::exception_ptr ep = std::current_exception()) {
            try {
                std::rethrow_exception(ep);
            } catch (std::exception const& e) {
                std::clog << "backtrace: caught an unhandled exception: "
                          << e.what()
                          << std::endl;
            } catch (...) {
                if (std::type_info* et = abi::__cxa_current_exception_type()) {
                    std::clog << "backtrace: unhandled exception type: "
                              << terminator::demangle(et->name())
                              << std::endl;
                } else {
                    std::clog << "backtrace: unhandled exception of unknown type" << std::endl;
                }
            }
        }
        std::_Exit(EXIT_FAILURE);
    }

}


namespace terminator {
    
    bool set(std::terminate_handler h) {
        terminate_handler.release();
        terminate_handler.reset(std::set_terminate(h));
        return terminate_handler.get() != nullptr;
    }
    
    std::terminate_handler get() {
        return terminate_handler.get();
    }
    
    bool setup() {
        static bool did_setup = false;
        if (!did_setup) {
            did_setup = terminator::set(backtrace_on_terminate);
        }
        return did_setup;
    }
    
}

#endif /// LIBIMREAD_EXT_ERRORS_TERMINATOR_HH_