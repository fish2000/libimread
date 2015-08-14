
#include <libimread/libimread.hpp>
#include <libimread/ext/errors/terminator.hh>
#include <libimread/errors.hh>

using namespace im;

#include "include/catch.hpp"

TEST_CASE("[terminator] Display an exception with imread_raise_default()", "[terminator-imread-raise-default]") {
    CHECK_THROWS_AS(
        imread_raise_default(ProgrammingError),
        ProgrammingError);
}

TEST_CASE("[terminator] Display an exception with imread_raise()", "[terminator-imread-raise]") {
    CHECK_THROWS_AS(
        imread_raise(ProgrammingError, "This is a programming error,", "... dogg"),
        ProgrammingError);
}

/// Uncomment this to manually test out libimread/ext/errors/terminator.hh

// TEST_CASE("[terminator] Directly call std::exit(-1)", "[terminator-std-exit]") {
//     std::terminate();
// }