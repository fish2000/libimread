
#include <iostream>
#include "typecode.hpp"

namespace typecode {
    
    intmap_t typecodemaps::init_integral_map() {
        intmap_t _integral_map = {
            { NPY_BOOL,         ENUM_NPY_BOOL::value }, 
            { NPY_BYTE,         ENUM_NPY_BYTE::value }, 
            { NPY_HALF,         ENUM_NPY_HALF::value }, 
            { NPY_SHORT,        ENUM_NPY_SHORT::value }, 
            { NPY_INT,          ENUM_NPY_INT::value }, 
            { NPY_LONG,         ENUM_NPY_LONG::value }, 
            { NPY_LONGLONG,     ENUM_NPY_LONGLONG::value }, 
            { NPY_UBYTE,        ENUM_NPY_UBYTE::value }, 
            { NPY_USHORT,       ENUM_NPY_USHORT::value }, 
            { NPY_UINT,         ENUM_NPY_UINT::value }, 
            { NPY_ULONG,        ENUM_NPY_ULONG::value }, 
            { NPY_ULONGLONG,    ENUM_NPY_ULONGLONG::value }, 
            { NPY_CFLOAT,       ENUM_NPY_CFLOAT::value }, 
            { NPY_CDOUBLE,      ENUM_NPY_CDOUBLE::value }, 
            { NPY_FLOAT,        ENUM_NPY_FLOAT::value }, 
            { NPY_DOUBLE,       ENUM_NPY_DOUBLE::value }, 
            { NPY_CLONGDOUBLE,  ENUM_NPY_CLONGDOUBLE::value }, 
            { NPY_LONGDOUBLE,   ENUM_NPY_LONGDOUBLE::value }
        };
        return _integral_map;
    }
    
    charmap_t typecodemaps::init_typecode_character_map() {
        charmap_t _typecode_character_map = {
            { NPY_BOOL,         NPY_BOOLLTR },
            { NPY_BYTE,         NPY_BYTELTR },
            { NPY_UBYTE,        NPY_UBYTELTR },
            { NPY_SHORT,        NPY_SHORTLTR },
            { NPY_USHORT,       NPY_USHORTLTR },
            { NPY_INT,          NPY_INTLTR },
            { NPY_UINT,         NPY_UINTLTR },
            { NPY_LONG,         NPY_LONGLTR },
            { NPY_ULONG,        NPY_ULONGLTR },
            { NPY_LONGLONG,     NPY_LONGLONGLTR },
            { NPY_ULONGLONG,    NPY_ULONGLONGLTR },
            { NPY_FLOAT,        NPY_FLOATLTR },
            { NPY_DOUBLE,       NPY_DOUBLELTR },
            { NPY_LONGDOUBLE,   NPY_LONGDOUBLELTR },
            { NPY_CFLOAT,       NPY_CFLOATLTR },
            { NPY_CDOUBLE,      NPY_CDOUBLELTR },
            { NPY_CLONGDOUBLE,  NPY_CLONGDOUBLELTR },
            { NPY_OBJECT,       NPY_OBJECTLTR },
            { NPY_STRING,       NPY_STRINGLTR },
            { NPY_UNICODE,      NPY_UNICODELTR },
            { NPY_VOID,         NPY_VOIDLTR },
            { NPY_DATETIME,     NPY_DATETIMELTR },
            { NPY_HALF,         NPY_HALFLTR },
            { NPY_TIMEDELTA,    NPY_TIMEDELTALTR },
            { NPY_CHAR,         NPY_CHARLTR },
            { NPY_USERDEF,      NPY_BOOLLTR }, /// bah
        };
        return _typecode_character_map;
    }
    
    stringmap_t typecodemaps::init_typecode_literal_map() {
        stringmap_t _typecode_literal_map = {
            { NPY_BOOL,         "NPY_BOOL" },
            { NPY_BYTE,         "NPY_BYTE" },
            { NPY_UBYTE,        "NPY_UBYTE" },
            { NPY_SHORT,        "NPY_SHORT" },
            { NPY_USHORT,       "NPY_USHORT" },
            { NPY_INT,          "NPY_INT" },
            { NPY_UINT,         "NPY_UINT" },
            { NPY_LONG,         "NPY_LONG" },
            { NPY_ULONG,        "NPY_ULONG" },
            { NPY_LONGLONG,     "NPY_LONGLONG" },
            { NPY_ULONGLONG,    "NPY_ULONGLONG" },
            { NPY_FLOAT,        "NPY_FLOAT" },
            { NPY_DOUBLE,       "NPY_DOUBLE" },
            { NPY_LONGDOUBLE,   "NPY_LONGDOUBLE" },
            { NPY_CFLOAT,       "NPY_CFLOAT" },
            { NPY_CDOUBLE,      "NPY_CDOUBLE" },
            { NPY_CLONGDOUBLE,  "NPY_CLONGDOUBLE" },
            { NPY_OBJECT,       "NPY_OBJECT" },
            { NPY_STRING,       "NPY_STRING" },
            { NPY_UNICODE,      "NPY_UNICODE" },
            { NPY_VOID,         "NPY_VOID" },
            { NPY_DATETIME,     "NPY_DATETIME" },
            { NPY_HALF,         "NPY_HALF" },
            { NPY_TIMEDELTA,    "NPY_TIMEDELTA" },
            { NPY_CHAR,         "NPY_CHAR" },
            { NPY_USERDEF,      "?" }
        };
        return _typecode_literal_map;
    }
    
    /// static initializers
    const intmap_t typecodemaps::integral = typecodemaps::init_integral_map();
    const charmap_t typecodemaps::character = typecodemaps::init_typecode_character_map();
    const stringmap_t typecodemaps::literal = typecodemaps::init_typecode_literal_map();
    
    NPY_TYPECHAR typechar(NPY_TYPES typecode) {
        try {
            return typecodemaps::character.at(typecode);
        } catch (const std::out_of_range& err) {
            std::cerr    << ">>> Type character not found for typecode: "
                         << typecode << std::endl << ">>> Exception message: "
                         << err.what() << std::endl;
            return typecodemaps::character.at(NPY_USERDEF);
        }
    }
    
    NPY_TYPECHAR typechar(unsigned int typecode) {
        return typecode::typechar(static_cast<NPY_TYPES>(typecode));
    }
    
    std::string name(NPY_TYPES typecode) {
        try {
            return typecodemaps::literal.at(typecode);
        } catch (const std::out_of_range& err) {
            std::cerr    << ">>> Typecode literal not found for typecode: "
                         << typecode << std::endl << ">>> Exception message: "
                         << err.what() << std::endl;
            return typecodemaps::literal.at(NPY_USERDEF);
        }
    }
    
    std::string name(unsigned int typecode) {
        return typecode::name(static_cast<NPY_TYPES>(typecode));
    }
}
