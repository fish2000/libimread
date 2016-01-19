#ifndef PyImgC_TYPECODE_H
#define PyImgC_TYPECODE_H

/// This was originally part of PLIIO/PyImgC

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <unordered_map>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <Python.h>
#include <numpy/ndarraytypes.h>


namespace std {
    
    template <>
    struct hash<NPY_TYPES> {
        
        typedef NPY_TYPES argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& typecode) const {
            return static_cast<result_type>(typecode);
        }
        
    };
    
}


namespace typecode {
    
    using ENUM_NPY_BOOL         = std::integral_constant<NPY_TYPES, NPY_BOOL>;
    using ENUM_NPY_BYTE         = std::integral_constant<NPY_TYPES, NPY_BYTE>;
    using ENUM_NPY_HALF         = std::integral_constant<NPY_TYPES, NPY_HALF>;
    using ENUM_NPY_SHORT        = std::integral_constant<NPY_TYPES, NPY_SHORT>;
    using ENUM_NPY_INT          = std::integral_constant<NPY_TYPES, NPY_INT>;
    using ENUM_NPY_LONG         = std::integral_constant<NPY_TYPES, NPY_LONG>;
    using ENUM_NPY_LONGLONG     = std::integral_constant<NPY_TYPES, NPY_LONGLONG>;
    using ENUM_NPY_UBYTE        = std::integral_constant<NPY_TYPES, NPY_UBYTE>;
    using ENUM_NPY_USHORT       = std::integral_constant<NPY_TYPES, NPY_USHORT>;
    using ENUM_NPY_UINT         = std::integral_constant<NPY_TYPES, NPY_UINT>;
    using ENUM_NPY_ULONG        = std::integral_constant<NPY_TYPES, NPY_ULONG>;
    using ENUM_NPY_ULONGLONG    = std::integral_constant<NPY_TYPES, NPY_ULONGLONG>;
    using ENUM_NPY_CFLOAT       = std::integral_constant<NPY_TYPES, NPY_CFLOAT>;
    using ENUM_NPY_CDOUBLE      = std::integral_constant<NPY_TYPES, NPY_CDOUBLE>;
    using ENUM_NPY_FLOAT        = std::integral_constant<NPY_TYPES, NPY_FLOAT>;
    using ENUM_NPY_DOUBLE       = std::integral_constant<NPY_TYPES, NPY_DOUBLE>;
    using ENUM_NPY_CLONGDOUBLE  = std::integral_constant<NPY_TYPES, NPY_CLONGDOUBLE>;
    using ENUM_NPY_LONGDOUBLE   = std::integral_constant<NPY_TYPES, NPY_LONGDOUBLE>;
    
    using intmap_t = std::unordered_map<int, NPY_TYPES>;
    using charmap_t = std::unordered_map<NPY_TYPES, NPY_TYPECHAR>;
    using stringmap_t = std::unordered_map<NPY_TYPES, std::string>;
    
    struct typecodemaps {
    
        static intmap_t init_integral_map() {
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
        
        static charmap_t init_typecode_character_map() {
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
        
        static stringmap_t init_typecode_literal_map() {
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
        
        static const intmap_t integral;
        static const charmap_t character;
        static const stringmap_t literal;
    };
    
    NPY_TYPECHAR typechar(NPY_TYPES typecode);
    NPY_TYPECHAR typechar(unsigned int typecode);
    
    std::string name(NPY_TYPES typecode);
    std::string name(unsigned int typecode);
}


#endif