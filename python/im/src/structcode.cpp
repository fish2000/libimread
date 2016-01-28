
#include <iostream>
#include <sstream>
#include <cctype>
#include <tuple>
#include "structcode.hpp"

namespace structcode {
    
    field_namer::field_namer()
        :idx(0)
        {}
    
    int field_namer::next() { return idx++; }
    
    void field_namer::add(std::string const& field_name) {
        field_name_vector.push_back(field_name);
    }
    
    bool field_namer::has(std::string const& field_name) {
        for (auto fn = std::begin(field_name_vector);
                  fn != std::end(field_name_vector); ++fn) {
            if (std::string(*fn) == field_name) {
                return true;
            }
        }
        return false;
    }
    
    std::string field_namer::operator()() {
        char str[5];
        while (true) {
            std::sprintf(str, "f%i", next());
            std::string dummy_name(str);
            if (!has(dummy_name)) {
                add(dummy_name);
                return dummy_name;
            }
        }
    }
    
    /// static initializers
    const stringmap_t structcodemaps::byteorder = structcodemaps::init_byteorder();
    const stringmap_t structcodemaps::native = structcodemaps::init_native();
    const stringmap_t structcodemaps::standard = structcodemaps::init_standard();
    
    shapevec_t parse_shape(std::string shapecode) {
        std::string segment;
        shapevec_t shape_elems;
        while (shapecode.find(",", 0) != std::string::npos) {
            std::size_t pos = shapecode.find(",", 0);
            segment = shapecode.substr(0, pos);
            shapecode.erase(0, pos+1);
            shape_elems.push_back(std::stoi(segment));
        }
        shape_elems.push_back(std::stoi(shapecode));
        return shape_elems;
    }
    
    parse_result_t parse(std::string structcode, bool toplevel) {
        stringvec_t tokens;
        structcode_t fields;
        field_namer field_names;
        
        // std::string byteorder = "@";
        std::string byteorder = "";
        std::size_t itemsize = 1;
        shapevec_t shape = { 0 };
        const shapevec_t noshape = { 0 };
        
        while (true) {
            if (structcode.size() == 0) { break; }
            switch (structcode[0]) {
                case '{': {
                    structcode.erase(0, 1);
                    int pos = 1;
                    std::size_t siz;
                    for (siz = 0; pos && (siz != structcode.size()); ++siz) {
                        if (structcode[siz] == '{') { ++pos; }
                        if (structcode[siz] == '}') { --pos; }
                    }
                    if (pos) { break; } /// too many open-brackets
                    std::string temp = structcode.substr(0, siz-1);
                    
                    std::string endianness;
                    stringvec_t parsetokens;
                    structcode_t pairvec;
                    std::tie(endianness, parsetokens, pairvec) = parse(temp, toplevel=false);
                    
                    structcode.erase(0, siz+1);
                    for (std::size_t idx = 0; idx < pairvec.size(); ++idx) {
                        fields.push_back(pairvec[idx]);
                    }
                }
                break;
                case '(': {
                    structcode.erase(0, 1);
                    int pos = 1;
                    std::size_t siz;
                    for (siz = 0; pos && (siz != structcode.size()); ++siz) {
                        if (structcode[siz] == '(') { ++pos; }
                        if (structcode[siz] == ')') { --pos; }
                    }
                    if (pos) { break; } /// too many open-parens
                    std::string shapestr = structcode.substr(0, siz-1);
                    shape = parse_shape(shapestr);
                    structcode.erase(0, siz);
                }
                break;
                case '*':
                {
                    /// SECRET NOOp
                    structcode.erase(0, 1);
                    std::size_t pos = structcode.find("*", 0);
                    std::string explicit_name = structcode.substr(0, pos);
                    field_names.add(explicit_name);
                    structcode.erase(0, pos+1);
                }
                break;
                case '@':
                case '=':
                case '<':
                case '>':
                case '^':
                case '!':
                {
                    try {
                        byteorder = structcodemaps::byteorder.at(structcode.substr(0, 1));
                    } catch (const std::out_of_range &err) {
                        std::cerr    << ">>> Byte order symbol not found: "
                                     << structcode.substr(0, 1) << std::endl
                                     << ">>> Exception message: "
                                     << err.what() << std::endl;
                    }
                    structcode.erase(0, 1);
                    tokens.push_back(byteorder);
                    // fields.push_back(std::make_pair("__byteorder__", byteorder));
                }    
                break;
                case ' ':
                {
                    /// NOP
                    structcode.erase(0, 1);
                }
                break;
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                {
                    std::size_t siz;
                    char digit;
                    int still_digits = 1;
                    for (siz = 0; still_digits && (siz < structcode.size()); siz++) {
                        digit = structcode.c_str()[siz];
                        still_digits = std::isdigit(digit) && digit != '(';
                    }
                    std::string numstr = structcode.substr(0, siz);
                    itemsize = static_cast<std::size_t>(std::stol(numstr));
                    structcode.erase(0, siz);
                    if (!std::isdigit(numstr.back())) {
                        structcode = std::string(&numstr.back()) + structcode;
                    }
                }
                break;
                default:
                std::size_t codelen = 1;
                std::string code = "";
                std::string name = "";
                std::string dtypechar = "";
                
                if (structcode.substr(0, codelen) == "Z") {
                    /// add next character
                    codelen++;
                }
                
                code += structcode.substr(0, codelen);
                structcode.erase(0, codelen);
                
                /// field name
                if (structcode.substr(0, 1) == ":") {
                    structcode.erase(0, 1);
                    std::size_t pos = structcode.find(":", 0);
                    name = structcode.substr(0, pos);
                    field_names.add(name);
                    structcode.erase(0, pos+1);
                }
            
                name = name.size() ? name : field_names();
            
                if (byteorder == "@" || byteorder == "^") {
                    try {
                        dtypechar = structcodemaps::native.at(code);
                    } catch (const std::out_of_range &err) {
                        std::cerr    << ">>> Native structcode symbol not found: "
                                     << code << std::endl
                                     << ">>> Exception message: "
                                     << err.what() << std::endl;
                        try {
                            dtypechar = structcodemaps::standard.at(code);
                        } catch (const std::out_of_range &err) {
                            std::cerr    << ">>> Subsequent standard symbol lookup failed: "
                                         << code << std::endl
                                         << ">>> Exception message: "
                                         << err.what() << std::endl;
                        }
                        break;
                     }
                } else {
                    try {
                        dtypechar = structcodemaps::standard.at(code);
                    } catch (const std::out_of_range &err) {
                        std::cerr    << ">>> Standard structcode symbol not found: "
                                     << code << std::endl
                                     << ">>> Exception message: "
                                     << err.what() << std::endl;
                        try {
                            dtypechar = structcodemaps::native.at(code);
                        } catch (const std::out_of_range &err) {
                            std::cerr    << ">>> Subsequent native structcode symbol lookup failed: "
                                         << code << std::endl
                                         << ">>> Exception message: "
                                         << err.what() << std::endl;
                        }
                        break;
                    }
                }
                
                const char last = dtypechar.back();
                if (last == 'U' || last == 'S' || last == 'V') {
                    if (itemsize > 1) {
                        std::ostringstream outstream;
                        outstream << itemsize;
                        dtypechar += outstream.str();
                    }
                    itemsize = 1;
                }
                
                if (shape != noshape) {
                    std::ostringstream outstream;
                    outstream << "(";
                    for (auto shape_elem = std::begin(shape);
                              shape_elem != std::end(shape); ++shape_elem) {
                        outstream << *shape_elem;
                        if (shape_elem + 1 != std::end(shape)) {
                            outstream << ", ";
                        }
                    }
                    outstream << ")";
                    dtypechar = outstream.str() + dtypechar;
                } else if (itemsize > 1) {
                    dtypechar = std::to_string(itemsize) + dtypechar;
                }
                
                fields.push_back(std::make_pair(name, dtypechar));
                tokens.push_back(dtypechar);
                
                itemsize = 1;
                shape = noshape;
                
                break;
            }
        }
        
        return std::make_tuple(std::move(byteorder),
                               std::move(tokens),
                               std::move(fields));
        
    }

} /// namespace structcode
