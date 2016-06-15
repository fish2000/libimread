#ifndef PyImgC_TYPECODE_H
#define PyImgC_TYPECODE_H

/// This was originally part of PLIIO/PyImgC

#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <unordered_map>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <Python.h>
#define NO_IMPORT_ARRAY
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
    
        static intmap_t init_integral_map();
        static charmap_t init_typecode_character_map();
        static stringmap_t init_typecode_literal_map();
        
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