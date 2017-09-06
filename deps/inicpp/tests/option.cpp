#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "option.h"
#include "types.h"
#include <vector>

using namespace std::literals;
// using namespace inicpp;

/**
 * Construct, get and set value to @ref option_value class.
 */
TEST(inicpp::option, option_value_helper_class) {
    inicpp::option_value<inicpp::string_ini_t> value("testing string");
    EXPECT_EQ(value.get(), "testing string");
    value.set("another string");
    EXPECT_EQ(value.get(), "another string");
}

/**
 * Create @ref option with single @ref string_t type and get some values.
 */
TEST(inicpp::option, simple_option_class) {
    inicpp::option string_option("name of option", "simple value");
    EXPECT_EQ(string_option.get_name(), "name of option");
    EXPECT_FALSE(string_option.is_list());
    EXPECT_EQ(string_option.get<inicpp::string_ini_t>(), "simple value");
    EXPECT_THROW(string_option.get<inicpp::float_ini_t>(), bad_cast_exception);
    EXPECT_EQ(string_option.get_list<inicpp::string_ini_t>(), std::vector<inicpp::string_ini_t>{ "simple value" });
}

/**
 * Create @ref option with list of @ref string_t types and get some values.
 */
TEST(inicpp::option, option_list_creation) {
    std::vector<std::string> values = { "value1", "value2" };
    inicpp::option                   string_option("name of option", values);
    EXPECT_EQ(string_option.get_name(), "name of option");
    EXPECT_TRUE(string_option.is_list());
    EXPECT_EQ(string_option.get<inicpp::string_ini_t>(), "value1");
    EXPECT_EQ(string_option.get_list<inicpp::string_ini_t>(), values);
}

/**
 * Test adding and removing items to/from an option. Assert that all items
 * in list are of the same type.
 */
TEST(inicpp::option, value_list_manipulation) {
    inicpp::option string_option("name of option", "simple value");
    EXPECT_FALSE(string_option.is_list());
    string_option.add_to_list<inicpp::string_ini_t>("value 2");
    EXPECT_TRUE(string_option.is_list());
    std::vector<inicpp::string_ini_t> expected_result = { "simple value", "value 2" };
    EXPECT_EQ(string_option.get_list<inicpp::string_ini_t>(), expected_result);
    EXPECT_THROW(string_option.add_to_list<inicpp::unsigned_ini_t>(2u), inicpp::bad_cast_exception);
    string_option.add_to_list<inicpp::string_ini_t>("last value", 1);
    expected_result.insert(++expected_result.begin(), "last value");
    EXPECT_EQ(string_option.get_list<inicpp::string_ini_t>(), expected_result);
    
    EXPECT_THROW(string_option.remove_from_list<inicpp::unsigned_ini_t>(2u), inicpp::bad_cast_exception);
    string_option.remove_from_list_pos(0);
    expected_result.erase(expected_result.begin());
    EXPECT_EQ(string_option.get_list<inicpp::string_ini_t>(), expected_result);
    string_option.remove_from_list<inicpp::string_ini_t>("last value");
    expected_result.clear();
    expected_result.push_back("value 2");
    EXPECT_EQ(string_option.get_list<inicpp::string_ini_t>(), expected_result);
    EXPECT_FALSE(string_option.is_list());
    EXPECT_EQ(string_option.get<inicpp::string_ini_t>(), "value 2");
}

/**
 * Test setting values to an option, including type change. Also test
 * overloads of assignment operator.
 */
TEST(inicpp::option, setting_values) {
    // Single values
    inicpp::option my_option("name", "value");
    my_option.set<inicpp::double_t>(5.2f);
    EXPECT_EQ(my_option.get<inicpp::double_t>(), 5.2f);
    my_option.set<inicpp::boolean_ini_t>(true);
    EXPECT_TRUE(my_option.get<inicpp::boolean_ini_t>());
    // struct custom_type { int a, b; } instance {4, 5};
    // EXPECT_THROW(my_option.set<custom_type>(instance), bad_cast_exception);
    
    // assignmet operators
    my_option = "string"s;
    EXPECT_EQ(my_option.get<inicpp::string_ini_t>(), "string");
    my_option = false;
    EXPECT_FALSE(my_option.get<inicpp::boolean_ini_t>());
    my_option = (inicpp::signed_ini_t)-56;
    EXPECT_EQ(my_option.get<inicpp::signed_ini_t>(), -56);
    my_option = (inicpp::unsigned_ini_t)789u;
    EXPECT_EQ(my_option.get<inicpp::unsigned_ini_t>(), 789u);
    my_option = (inicpp::float_ini_t)25.6;
    EXPECT_EQ(my_option.get<inicpp::float_ini_t>(), 25.6);
    
    // vector types
    std::vector<inicpp::signed_ini_t> values = { 5, 6, 8, 9 };
    my_option.set_list<inicpp::signed_ini_t>(values);
    EXPECT_EQ(my_option.get_list<inicpp::signed_ini_t>(), values);
}

/**
 * Test various ways of copying the option - copy and move constructors,
 * assignments operators.
 */
TEST(inicpp::option, copying) {
    inicpp::option my_option("name", "value");
    
    // copy constructor
    inicpp::option copied(my_option);
    EXPECT_EQ(copied.get_name(), my_option.get_name());
    EXPECT_EQ(copied.get<inicpp::string_ini_t>(), my_option.get<inicpp::string_ini_t>());
    
    // move constructor
    inicpp::option moved(std::move(copied));
    EXPECT_EQ(moved.get_name(), my_option.get_name());
    EXPECT_EQ(moved.get<inicpp::string_ini_t>(), my_option.get<inicpp::string_ini_t>());

    // copy assignment
    inicpp::option copied_assingment("other name", "different value");
    copied_assingment = my_option;
    EXPECT_EQ(copied_assingment.get_name(), my_option.get_name());
    EXPECT_EQ(copied_assingment.get<inicpp::string_ini_t>(), my_option.get<inicpp::string_ini_t>());
    
    // move assignment
    inicpp::option moved_assignment("other name", "different value");
    moved_assignment = std::move(copied_assingment);
    EXPECT_EQ(moved_assignment.get_name(), my_option.get_name());
    EXPECT_EQ(moved_assignment.get<inicpp::string_ini_t>(), my_option.get<inicpp::string_ini_t>());
}

/**
 * Test format of output stream.
 */
TEST(option, writing_to_stream) {
    inicpp::option             my_option("name", "value");
    std::ostringstream str;
    
    // string
    str << my_option;
    EXPECT_EQ(str.str(), "name = value\n");
    
    // signed
    str.str("");
    my_option.set<inicpp::signed_ini_t>(-89);
    str << my_option;
    EXPECT_EQ(str.str(), "name = -89\n");
    
    // unsigned
    str.str("");
    my_option.set<inicpp::unsigned_ini_t>(42);
    str << my_option;
    EXPECT_EQ(str.str(), "name = 42\n");
    
    // float
    str.str("");
    my_option.set<inicpp::float_ini_t>(52.4);
    str << my_option;
    EXPECT_EQ(str.str(), "name = 52.4\n");
    
    // boolean
    str.str("");
    my_option.set<inicpp::boolean_ini_t>(true);
    str << my_option;
    EXPECT_EQ(str.str(), "name = yes\n");
    
    // enum
    str.str("");
    inicpp::enum_ini_t en("enum_value");
    my_option.setinicpp::<enum_ini_t>(en);
    str << my_option;
    EXPECT_EQ(str.str(), "name = enum_value\n");

    // string list
    str.str("");
    my_option.set_list<inicpp::string_ini_t>({ "option 1", "option 2" });
    str << my_option;
    EXPECT_EQ(str.str(), "name = option 1,option 2\n");
}
