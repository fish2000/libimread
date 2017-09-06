#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "option_schema.h"
#include "schema.h"
#include "section_schema.h"

// using namespace inicpp;

TEST(inicpp::schema, creation_and_assignments) {
    // normal construction
    inicpp::schema schm;
    EXPECT_EQ(schm.size(), 0u);
    inicpp::section_schema_params sect_params;
    sect_params.name = "name";
    schm.add_section(sect_params);
    EXPECT_EQ(schm.size(), 1u);
    EXPECT_TRUE(schm.contains("name"));
    
    // copy constructor and assignment
    inicpp::schema copy_constructed(schm);
    EXPECT_EQ(copy_constructed.size(), 1u);
    EXPECT_TRUE(copy_constructed.contains("name"));
    inicpp::schema copy_assigned = schm;
    EXPECT_EQ(copy_assigned.size(), 1u);
    EXPECT_TRUE(copy_assigned.contains("name"));
    
    // move constructor and assignment
    inicpp::schema move_constructed(std::move(copy_constructed));
    EXPECT_EQ(move_constructed.size(), 1u);
    EXPECT_TRUE(move_constructed.contains("name"));
    inicpp::schema move_assigned = std::move(copy_assigned);
    EXPECT_EQ(move_assigned.size(), 1u);
    EXPECT_TRUE(move_assigned.contains("name"));
}

TEST(inicpp::schema, adding_and_querying_sections) {
    // create testing section in schema
    inicpp::schema                schm;
    inicpp::section_schema_params sect1_params;
    sect1_params.name = "name";
    schm.add_section(sect1_params);
    inicpp::section_schema_params sect2_params;
    sect2_params.name = "other";
    schm.add_section(sect2_params);
    
    // assert prerequisites and contains method
    EXPECT_EQ(schm.size(), 2u);
    EXPECT_THROW(schm.add_section(sect2_params), inicpp::ambiguity_exception);
    EXPECT_TRUE(schm.contains("name"));
    EXPECT_TRUE(schm.contains("other"));
    EXPECT_FALSE(schm.contains("random"));
    
    // operator []
    EXPECT_EQ(schm[0].get_name(), "name");
    EXPECT_EQ(schm[1].get_name(), "other");
    EXPECT_THROW(schm[2].get_name(), inicpp::not_found_exception);
    EXPECT_EQ(schm["name"].get_name(), "name");
    EXPECT_EQ(schm["other"].get_name(), "other");
    EXPECT_THROW(schm["random"].get_name(), inicpp::not_found_exception);
}

TEST(schema, adding_and_querying_options) {
    // create testing schema
    inicpp::schema                schm;
    inicpp::section_schema_params sect1_params;
    sect1_params.name = "name";
    schm.add_section(sect1_params);
    
    // add some options
    inicpp::option_schema_params<inicpp::string_ini_t> opt1_params;
    opt1_params.name = "opt1";
    inicpp::option_schema opt1_schm(opt1_params);
    EXPECT_THROW(schm.add_option("random", opt1_schm), inicpp::not_found_exception);
    EXPECT_NO_THROW(schm.add_option("name", opt1_schm));
    EXPECT_EQ(schm["name"].size(), 1u);
    EXPECT_THROW(schm.add_option("name", opt1_params), inicpp::ambiguity_exception);
    EXPECT_THROW(schm.add_option("random", opt1_params), inicpp::not_found_exception);
}

TEST(schema, validate_config) {
    // create testing schema
    inicpp::schema schm;
    
    inicpp::section_schema_params sect1_params;
    sect1_params.name        = "name";
    sect1_params.requirement = inicpp::item_requirement::mandatory;
    schm.add_section(sect1_params);
    
    inicpp::section_schema_params sect2_params;
    sect2_params.name        = "default";
    sect2_params.requirement = inicpp::item_requirement::optional;
    schm.add_section(sect2_params);
    
    inicpp::option_schema_params<inicpp::signed_ini_t> opt1_params;
    opt1_params.name          = "opt";
    opt1_params.requirement   = inicpp::item_requirement::optional;
    opt1_params.type          = inicpp::option_item::single;
    opt1_params.default_value = "42";
    schm.add_option("name", opt1_params);
    
    // create config
    inicpp::config  conf;
    section sect("name");
    section sect2("name2");
    inicpp::option  opt("opt", "258");
    sect.add_option(opt);
    conf.add_section(sect);
    conf.add_section(sect2);
    
    // perform validation
    EXPECT_NO_THROW(schm.validate_config(conf, inicpp::schema_mode::relaxed));
    
    // test if options are validated
    EXPECT_EQ(conf["name"]["opt"].get<inicpp::signed_ini_t>(), 258);
    
    // test if not specified sections are added
    EXPECT_TRUE(conf.contains("default"));
    
    // test if exception is thrown with strict mode on unknown section
    EXPECT_THROW(schm.validate_config(conf, inicpp::schema_mode::strict), inicpp::validation_exception);
}

TEST(schema, stream_output) {
    // create testing schema
    inicpp::schema                schm;
    inicpp::section_schema_params sect1_params;
    sect1_params.name        = "name";
    sect1_params.comment     = "multiline\ncomment";
    sect1_params.requirement = inicpp::item_requirement::mandatory;
    schm.add_section(sect1_params);
    inicpp::section_schema_params sect2_params;
    sect2_params.name        = "other";
    sect2_params.comment     = "other section";
    sect2_params.requirement = inicpp::item_requirement::optional;
    schm.add_section(sect2_params);

    // test expectations
    std::string expected = ""
                           ";multiline\n"
                           ";comment\n"
                           ";<mandatory>\n"
                           "[name]\n"
                           ";other section\n"
                           ";<optional>\n"
                           "[other]\n";

    std::ostringstream str;
    str << schm;
    EXPECT_EQ(str.str(), expected);
}
