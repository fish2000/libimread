#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "types.h"

// using namespace inicpp;

TEST(inicpp::types, get_option_enum_type) {
    EXPECT_EQ(inicpp::get_option_enum_type<inicpp::boolean_ini_t>(),    inicpp::option_type::boolean_e);
    EXPECT_EQ(inicpp::get_option_enum_type<inicpp::enum_ini_t>(),       inicpp::option_type::enum_e);
    EXPECT_EQ(inicpp::get_option_enum_type<inicpp::float_ini_t>(),      inicpp::option_type::float_e);
    EXPECT_EQ(inicpp::get_option_enum_type<inicpp::signed_ini_t>(),     inicpp::option_type::signed_e);
    EXPECT_EQ(inicpp::get_option_enum_type<inicpp::unsigned_ini_t>(),   inicpp::option_type::unsigned_e);
    EXPECT_EQ(inicpp::get_option_enum_type<inicpp::string_ini_t>(),     inicpp::option_type::string_e);
    EXPECT_EQ(inicpp::get_option_enum_type<const char*>(),              inicpp::option_type::invalid_e);
}
