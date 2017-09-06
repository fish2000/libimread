#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "exception.h"

// import literals
// we use string operator "" s() - the 's' suffix
// of const char * string makes it std::string. For example:
// auto str = "hello"s;
// str variable is std::string type. This needs C++14 support.
using namespace std::literals;
// using namespace inicpp;

TEST(exceptions, generic) {
    inicpp::exception ex;
    EXPECT_EQ(ex.what(), "Generic inicpp exception"s);
    inicpp::exception ex_init("text");
    EXPECT_EQ(ex_init.what(), "text"s);
}

TEST(exceptions, parser) {
    inicpp::parser_exception ex("message");
    EXPECT_EQ(ex.what(), "message"s);
}

TEST(exceptions, bad_cast) {
    inicpp::bad_cast_exception ex_init("text");
    EXPECT_EQ(ex_init.what(), "text"s);
    inicpp::bad_cast_exception ex_from_to("integer", "boolean");
    EXPECT_EQ(ex_from_to.what(), "Bad conversion from: integer to: boolean"s);
}

TEST(exceptions, not_found) {
    inicpp::not_found_exception ex(5);
    EXPECT_EQ(ex.what(), "Element on index: 5 was not found"s);
    inicpp::not_found_exception ex_name("name");
    EXPECT_EQ(ex_name.what(), "Element: name not found in container"s);
}

TEST(exceptions, ambiguity) {
    inicpp::ambiguity_exception ex("elname");
    EXPECT_EQ(ex.what(), "Ambiguous element with name: elname"s);
}

TEST(exceptions, validation) {
    inicpp::validation_exception ex("message");
    EXPECT_EQ(ex.what(), "message"s);
}

TEST(exceptions, invalid_type) {
    inicpp::invalid_type_exception ex("message");
    EXPECT_EQ(ex.what(), "message"s);
}

TEST(exceptions, not_implemented) {
    inicpp::not_implemented_exception ex;
    EXPECT_EQ(ex.what(), "Not implemented"s);
}
