#include "inicpp.h"
#include <string>
#include <iostream>
#include <sstream>

// using namespace inicpp;

std::string get_config() {
    std::string conf_str = ""
                           "[Section 1]\n"
                           "; comment\n"
                           "Option 1 = -1285\n"
                           "Option 2 = 5, 25,856\n"
                           "float1 = +4.1234565E+45\n"
                           "unknown_option = haha\n";
    return conf_str;
}

int main(void) {
    std::cout << "Create config schema" << std::endl;
    std::cout << "--------------------" << std::endl;
    inicpp::schema schm;
    
    // add section 'Section 1'
    inicpp::section_schema_params section_1_params;
    section_1_params.name        = "Section 1";
    section_1_params.comment     = "comment";
    section_1_params.requirement = inicpp::item_requirement::optional;
    schm.add_section(section_1_params);
    
    // add options to 'Section 1'
    inicpp::option_schema_params<inicpp::signed_ini_t> option_1_params;
    option_1_params.name        = "Option 1";
    option_1_params.type        = inicpp::option_item::single;
    option_1_params.requirement = inicpp::item_requirement::mandatory;
    
    option_1_params.validator   = [](inicpp::signed_ini_t i) {
        return i < 0;
    }; // valid values are only negative ones
    
    option_1_params.comment       = "Important option\nshould be negative";
    option_1_params.default_value = "-1";
    schm.add_option("Section 1", option_1_params);
    
    inicpp::option_schema_params<inicpp::unsigned_ini_t> option_2_params;
    option_2_params.name        = "Option 2";
    option_2_params.type        = inicpp::option_item::list;
    option_2_params.requirement = inicpp::item_requirement::mandatory;
    schm.add_option("Section 1", option_2_params);
    
    // this option is not specified in config -> default value will be added
    inicpp::option_schema_params<inicpp::signed_ini_t> option_3_params;
    option_3_params.name          = "Option 3";
    option_3_params.type          = inicpp::option_item::single;
    option_3_params.requirement   = inicpp::item_requirement::optional;
    option_3_params.default_value = "-42";
    schm.add_option("Section 1", option_3_params);
    
    inicpp::option_schema_params<inicpp::float_ini_t> option_4_params;
    option_4_params.name        = "float1";
    option_4_params.type        = inicpp::option_item::single;
    option_4_params.requirement = inicpp::item_requirement::mandatory;
    schm.add_option("Section 1", option_4_params);
    
    std::cout << "done..." << std::endl << std::endl;
    
    std::cout << "Load and validate config in relaxed mode" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    inicpp::config conf = inicpp::parser::load(get_config(), schm, inicpp::schema_mode::relaxed);
    std::cout << "done..." << std::endl << std::endl;
    
    std::cout << "Check, if options are properly typed" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "  'Option 1' is inicpp::signed_ini_t type with value '"
              << conf["Section 1"]["Option 1"].get<inicpp::signed_ini_t>() << "'" << std::endl;
    
    // 'Option 1' is not enum_ini_t type, it's compile time error
    // conf["Section 1"]["Option 1"].get<inicpp::enum_ini_t>();
    std::cout << "  'Option 2' is a list of inicpp::unsigned_ini_t with "
              << conf["Section 1"]["Option 2"].get_list<inicpp::unsigned_ini_t>().size() << " values"
              << std::endl;
    
    std::cout << "  'Option 3' was added with default value '"
              << conf["Section 1"]["Option 3"].get<inicpp::signed_ini_t>() << "' and correct type"
              << std::endl;
    
    std::cout << "  'float1' option has value '" << conf["Section 1"]["float1"].get<inicpp::float_ini_t>()
              << std::endl;
    
    std::cout << "  'unknown_option' was left as string with value '"
              << conf["Section 1"]["unknown_option"].get<inicpp::string_ini_t>() << "'" << std::endl;
    
    std::cout << "done..." << std::endl << std::endl;
    
    std::cout << "Validation with strict mode fails due to 'unknown_option'" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    
    try {
        inicpp::parser::load(get_config(), schm, inicpp::schema_mode::strict);
    } catch (inicpp::validation_exception&) { std::cout << "  Strict mode validation failed" << std::endl; }
    std::cout << "done..." << std::endl << std::endl;
    
    std::cout << "Write default configuration from schema to stream" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::ostringstream str;
    str << schm;
    std::cout << str.str();
    std::cout << "done..." << std::endl << std::endl;
    
    std::cout << "Write current configuration with comments from schema to stream" << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    str.str("");
    inicpp::parser::save(conf, schm, str);
    std::cout << str.str();
    std::cout << "done..." << std::endl << std::endl;
}
