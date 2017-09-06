#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "option_schema.h"

// using namespace inicpp;

TEST(inicpp::option_schema, options_schema_params) {
    oinicpp::ption_schema_params<inicpp::signed_ini_t> params;
    
    EXPECT_EQ(params.name, "");
    EXPECT_EQ(params.requirement, inicpp::item_requirement::mandatory);
    EXPECT_EQ(params.type, inicpp::option_item::single);
    EXPECT_EQ(params.default_value, "");
    EXPECT_EQ(params.comment, "");
    EXPECT_EQ(params.validator, nullptr);
    
    params.name          = "name";
    params.requirement   = inicpp::item_requirement::optional;
    params.type          = inicpp::option_item::list;
    params.default_value = "default_value";
    params.comment       = "comment";
    params.validator     = [](signed_ini_t i) { return true; };

    EXPECT_EQ(params.name, "name");
    EXPECT_EQ(params.requirement, inicpp::item_requirement::optional);
    EXPECT_EQ(params.type, inicpp::option_item::list);
    EXPECT_EQ(params.default_value, "default_value");
    EXPECT_EQ(params.comment, "comment");
    EXPECT_TRUE(params.validator(5));
}

TEST(inicpp::option_schema, creation_and_copying) {
    inicpp::option_schema_params<inicpp::signed_ini_t> params;
    params.name = "name";
    inicpp::option_schema_params<inicpp::unsigned_ini_t> other_params;
    inicpp::option_schema_params<const char*>    invalid_params;
    
    inicpp::option_schema my_option(params);
    EXPECT_THROW(inicpp::option_schema inicpp::invalid_option(invalid_params), inicpp::invalid_type_exception);
    
    // copy constructor
    inicpp::option_schema copied(my_option);
    EXPECT_EQ(copied.get_name(), my_option.get_name());
    
    // move constructor
    inicpp::option_schema moved(std::move(copied));
    EXPECT_EQ(moved.get_name(), my_option.get_name());
    
    // copy assignment
    inicpp::option_schema copied_assingment(other_params);
    copied_assingment = my_option;
    EXPECT_EQ(copied_assingment.get_name(), my_option.get_name());
    
    // move assignment
    inicpp::option_schema moved_assignment(other_params);
    moved_assignment = std::move(copied_assingment);
    EXPECT_EQ(moved_assignment.get_name(), my_option.get_name());
}

TEST(inicpp::option_schema, querying_properties) {
    inicpp::option_schema_params<inicpp::signed_ini_t> params;
    params.name          = "name";
    params.requirement   = inicpp::item_requirement::optional;
    params.type          = inicpp::option_item::list;
    params.default_value = "default_value";
    params.comment       = "comment";
    params.validator     = [](inicpp::signed_ini_t i) { return true; };
    
    inicpp::option_schema my_option(params);
    
    EXPECT_EQ(my_option.get_name(), "name");
    EXPECT_FALSE(my_option.is_mandatory());
    EXPECT_EQ(my_option.get_type(), inicpp::option_type::signed_e);
    EXPECT_TRUE(my_option.is_list());
    EXPECT_EQ(my_option.get_default_value(), "default_value");
    EXPECT_EQ(my_option.get_comment(), "comment");
}

TEST(inicpp::option_schema, type_deduction) {
    inicpp::option_schema_params<inicpp::signed_ini_t> signed_params;
    inicpp::option_schema                      signed_option(signed_params);
    EXPECT_EQ(signed_option.get_type(), inicpp::option_type::signed_e);
    
    inicpp::option_schema_params<inicpp::unsigned_ini_t> unsigned_params;
    inicpp::option_schema                        unsigned_option(unsigned_params);
    EXPECT_EQ(unsigned_option.get_type(), inicpp::option_type::unsigned_e);
    
    inicpp::option_schema_params<inicpp::boolean_ini_t> boolean_params;
    inicpp::option_schema                       boolean_option(boolean_params);
    EXPECT_EQ(boolean_option.get_type(), inicpp::option_type::boolean_e);
    
    inicpp::option_schema_params<inicpp::string_ini_t> string_params;
    inicpp::option_schema                      string_option(string_params);
    EXPECT_EQ(string_option.get_type(), inicpp::option_type::string_e);
    
    inicpp::option_schema_params<inicpp::float_ini_t> float_params;
    inicpp::option_schema                     float_option(float_params);
    EXPECT_EQ(float_option.get_type(), inicpp::option_type::float_e);
    
    inicpp::option_schema_params<inicpp::enum_ini_t> enum_params;
    inicpp::option_schema                    enum_option(enum_params);
    EXPECT_EQ(enum_option.get_type(), inicpp::option_type::enum_e);
}

TEST(inicpp::option_schema, validation) {
    // single value
    inicpp::option                             signed_option("name", "-45");
    inicpp::option_schema_params<inicpp::signed_ini_t> signed_params;
    signed_params.name      = "name";
    signed_params.type      = inicpp::option_item::single;
    signed_params.validator = [](inicpp::signed_ini_t i) { return i < 5; };
    
    inicpp::option_schema signed_schema(signed_params);
    EXPECT_NO_THROW(signed_schema.validate_option(signed_option));
    EXPECT_EQ(signed_option.get<inicpp::signed_ini_t>(), -45);
    
    signed_params.validator = [](inicpp::signed_ini_t i) { return i > 0; };
    inicpp::option_schema validator_fail_schema(signed_params);
    EXPECT_THROW(validator_fail_schema.validate_option(signed_option), inicpp::validation_exception);
    
    signed_option.set<inicpp::string_ini_t>("63");
    EXPECT_THROW(signed_schema.validate_option(signed_option), inicpp::validation_exception);
    
    // list value
    inicpp::option                    float_option("name", "");
    std::vector<inicpp::string_ini_t> string_values{ "4.5", "-6.3", "0.0" };
    float_option.set_list(string_values);
    inicpp::option_schema_params<inicpp::float_ini_t> float_params;
    float_params.name      = "name";
    float_params.type      = inicpp::option_item::list;
    float_params.validator = [](inicpp::float_ini_t i) { return i < 5.2 && i > -81.1; };
    
    inicpp::option_schema float_schema(float_params);
    EXPECT_NO_THROW(float_schema.validate_option(float_option));
    std::vector<inicpp::float_ini_t> float_values{ 4.5, -6.3, 0.0 };
    EXPECT_EQ(float_option.get_list<inicpp::float_ini_t>(), float_values);
}

TEST(option_schema, writing_to_ostream) {
    std::ostringstream                 str;
    inicpp::option_schema_params<signed_ini_t> params;
    params.name          = "name";
    params.requirement   = inicpp::item_requirement::optional;
    params.type          = inicpp::option_item::list;
    params.default_value = "default_value";
    params.comment       = "comment\nmultiline";
    params.validator     = [](inicpp::signed_ini_t i) { return true; };
    inicpp::option_schema my_option(params);
    
    str << my_option;
    
    std::string expected_output = ";comment\n"
                                  ";multiline\n"
                                  ";<optional, list>\n"
                                  ";<default value: \"default_value\">\n"
                                  "name = default_value\n";
    EXPECT_EQ(str.str(), expected_output);
}
