
#include <typeinfo>

#include <libimread/libimread.hpp>
#include <libimread/ext/errors/terminator.hh>
#include <libimread/errors.hh>
#include <libimread/interleaved.hh>

using namespace im;

#include "include/catch.hpp"

namespace {
    
    TEST_CASE("[terminator] Display an exception with imread_raise_default()",
              "[terminator-imread-raise-default]")
    {
        CHECK_THROWS_AS(
            imread_raise_default(ProgrammingError),
            ProgrammingError);
    }
    
    TEST_CASE("[terminator] Display an exception with imread_raise()",
              "[terminator-imread-raise]")
    {
        CHECK_THROWS_AS(
            imread_raise(ProgrammingError, "This is a programming error,", "... dogg"),
            ProgrammingError);
    }
    
    #define TYPE_ID(thingy) typeid(thingy).name()
    
    TEST_CASE("[terminator] Demangle some mangled-up shit",
              "[terminator-demangle-shit]")
    {
        /// sources of the pre-mangled shit
        using RGB = im::color::RGB;
        using RGBA = im::color::RGBA;
        using IRGB = im::InterleavedImage<RGB>;
        using IRGBA = im::InterleavedImage<RGBA>;
        
        /// demangle names from typeid() results
        std::string one = terminator::demangle(TYPE_ID(RGB));
        std::string two = terminator::demangle(TYPE_ID(RGBA));
        std::string three = terminator::demangle(TYPE_ID(IRGB));
        std::string four = terminator::demangle(TYPE_ID(IRGBA));
        
        WTF("Demangling name 01:\t\t", TYPE_ID(RGB),     one);
        WTF("Demangling name 02:\t\t", TYPE_ID(RGBA),    two);
        WTF("Demangling name 03:\t\t", TYPE_ID(IRGB),    three);
        WTF("Demangling name 04:\t\t", TYPE_ID(IRGBA),   four);
        
    }
    
    /// Uncomment this to manually test out libimread/ext/errors/terminator.hh
    
    // TEST_CASE("[terminator] Directly call std::terminate()", "[terminator-directly-call-std-terminate]") {
    //     std::terminate();
    // }
    
}
