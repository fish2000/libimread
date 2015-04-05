
#include "typecode.hpp"
using namespace std;

namespace typecode {
    
    /// static initializers
    const map<int, NPY_TYPES> typecodemaps::integral = typecodemaps::init_integral_map();
    const map<NPY_TYPES, NPY_TYPECHAR> typecodemaps::character = typecodemaps::init_typecode_character_map();
    const map<NPY_TYPES, string> typecodemaps::literal = typecodemaps::init_typecode_literal_map();
    
    NPY_TYPECHAR typechar(NPY_TYPES typecode) {
        try {
            return typecodemaps::character.at(typecode);
        } catch (const out_of_range &err) {
            cerr    << ">>> Type character not found for typecode: "
                    << typecode << "\n>>> Exception message: "
                    << err.what() << "\n";
            return typecodemaps::character.at(NPY_USERDEF);
        }
    }
    NPY_TYPECHAR typechar(unsigned int typecode) {
        return typecode::typechar(static_cast<NPY_TYPES>(typecode));
    }
    
    string name(NPY_TYPES typecode) {
        try {
            return typecodemaps::literal.at(typecode);
        } catch (const out_of_range &err) {
            cerr    << ">>> Typecode literal not found for typecode: "
                    << typecode << "\n>>> Exception message: "
                    << err.what() << "\n";
            return typecodemaps::literal.at(NPY_USERDEF);
        }
    }
    string name(unsigned int typecode) {
        return typecode::name(static_cast<NPY_TYPES>(typecode));
    }
}
