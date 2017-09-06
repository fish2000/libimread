#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "section_schema.h"

// using namespace inicpp;

TEST(inicpp::section_schema, sections_schema_params) {
    inicpp::section_schema_params params;
    
    EXPECT_EQ(params.name, "");
    EXPECT_EQ(params.requirement, inicpp::item_requirement::mandatory);
    EXPECT_EQ(params.comment, "");
    
    params.name        = "name";
    params.requirement = inicpp::item_requirement::optional;
    params.comment     = "comment";
    
    EXPECT_EQ(params.name, "name");
    EXPECT_EQ(params.requirement, inicpp::item_requirement::optional);
    EXPECT_EQ(params.comment, "comment");
}

TEST(inicpp::section_schema, creation_and_copying) {
    inicpp::section_schema_params params;
    params.name = "name";
    inicpp::section_schema_params other_params;
    
    inicpp::section_schema my_section(params);
    
    // copy constructor
    inicpp::section_schema copied(my_section);
    EXPECT_EQ(copied.get_name(), my_section.get_name());
    
    // move constructor
    inicpp::section_schema moved(std::move(copied));
    EXPECT_EQ(moved.get_name(), my_section.get_name());
    
    // copy assignment
    inicpp::section_schema copied_assingment(other_params);
    copied_assingment = my_section;
    EXPECT_EQ(copied_assingment.get_name(), my_section.get_name());
    
    // move assignment
    inicpp::section_schema moved_assignment(other_params);
    moved_assignment = std::move(copied_assingment);
    EXPECT_EQ(moved_assignment.get_name(), my_section.get_name());
}

TEST(inicpp::section_schema, querying_properties) {
    inicpp::section_schema_params params;
    params.name        = "name";
    params.requirement = inicpp::item_requirement::optional;
    params.comment     = "comment";
    
    inicpp::section_schema my_section(params);
    
    EXPECT_EQ(my_section.get_name(), "name");
    EXPECT_FALSE(my_section.is_mandatory());
    EXPECT_EQ(my_section.get_comment(), "comment");
}

TEST(inicpp::section_schema, adding_and_removing_options) {
    inicpp::section_schema_params params;
    params.name = "section";
    inicpp::section_schema my_section(params);
    
    inicpp::option_schema_params<inicpp::signed_ini_t> opt_params;
    opt_params.name          = "name";
    opt_params.requirement   = inicpp::item_requirement::optional;
    opt_params.type          = inicpp::option_item::list;
    opt_params.default_value = "default_value";
    opt_params.comment       = "comment";
    opt_params.validator     = [](inicpp::signed_ini_t i) { return true; };
    inicpp::option_schema my_option(opt_params);
    
    EXPECT_NO_THROW(my_section.add_option(my_option));
    EXPECT_THROW(my_section.add_option(my_option), inicpp::ambiguity_exception);
    
    opt_params.name = "other";
    EXPECT_NO_THROW(my_section.add_option(opt_params));
    
    EXPECT_THROW(my_section.remove_option("unknown name"), inicpp::not_found_exception);
    EXPECT_NO_THROW(my_section.remove_option(my_option.get_name()));
    EXPECT_NO_THROW(my_section.remove_option(opt_params.name));
}

TEST(inicpp::section_schema, validation) {
    // prepare section schema
    inicpp::section_schema_params params;
    params.name = "section";
    inicpp::section_schema my_section(params);
    
    inicpp::option_schema_params<inicpp::signed_ini_t> opt1_params;
    opt1_params.name          = "name";
    opt1_params.requirement   = inicpp::item_requirement::mandatory;
    opt1_params.type          = inicpp::option_item::single;
    opt1_params.default_value = "42";
    inicpp::option_schema my_option1(opt1_params);
    
    inicpp::option_schema_params<inicpp::unsigned_ini_t> opt2_params;
    opt2_params.name          = "opt_2_name";
    opt2_params.requirement   = inicpp::item_requirement::optional;
    opt2_params.type          = inicpp::option_item::single;
    opt2_params.default_value = "567";
    inicpp::option_schema my_option2(opt2_params);
    
    my_section.add_option(my_option1);
    my_section.add_option(my_option2);
    
    // prepare configuration section
    inicpp::section sect("section");
    
    inicpp::option opt1("name", "-85");
    inicpp::option opt2("some_name", "some_value");
    
    sect.add_option(opt1);
    sect.add_option(opt2);
    
    // perform validation
    EXPECT_NO_THROW(my_section.validate_section(sect, inicpp::schema_mode::relaxed));
    
    // test if options are validated
    EXPECT_EQ(sect["name"].get<inicpp::signed_ini_t>(), -85);
    
    // test if default values are added from schema and validated (type changed)
    EXPECT_EQ(sect["opt_2_name"].get<inicpp::unsigned_ini_t>(), 567u);
    
    // test if exception is thrown with strict mode on unknown option
    EXPECT_THROW(my_section.validate_section(sect, inicpp::schema_mode::strict), inicpp::validation_exception);
}

TEST(section_schema, writing_to_ostream) {
    std::ostringstream    str;
    inicpp::section_schema_params params;
    params.name        = "name";
    params.requirement = inicpp::item_requirement::optional;
    params.comment     = "comment\nmultiline";
    sinicpp::ection_schema my_section(params);
    
    inicpp::option_schema_params<inicpp::string_ini_t> opt_params;
    opt_params.name          = "opt_name";
    opt_params.requirement   = inicpp::item_requirement::optional;
    opt_params.type          = inicpp::option_item::list;
    opt_params.default_value = "default,value";
    inicpp::option_schema my_option(opt_params);
    my_section.add_option(my_option);

    str << my_section;

    std::string expected_output = ";comment\n"
                                  ";multiline\n"
                                  ";<optional>\n"
                                  "[name]\n"
                                  ";<optional, list>\n"
                                  ";<default value: \"default,value\">\n"
                                  "opt_name = default,value\n";
    EXPECT_EQ(str.str(), expected_output);
}
