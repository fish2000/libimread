
/// SPECIOUS, QUESTIONABLE, DEBATEWORTHY, &c...

namespace dtype {
    
    
    using typecode = NPY_TYPES;
    using Halide::Type;
    
    typecode from_typestruct(Type ts) {
        switch (ts.code) {
            
            case Type::UInt:
                switch (ts.bits) {
                    case 1: return NPY_BOOL;
                    case 8: return NPY_UINT8;
                    case 16: return NPY_UINT16;
                    case 32: return NPY_UINT32;
                    case 64: return NPY_UINT64;
                }
                return NPY_NOTYPE;
        
            case Type::Int:
            switch (ts.bits) {
                case 1: return NPY_BOOL;
                case 8: return NPY_INT8;
                case 16: return NPY_INT16;
                case 32: return NPY_INT32;
                case 64: return NPY_INT64;
            }
            return NPY_NOTYPE;
        
            case Type::Float:
            switch (ts.bits) {
                // case 1: return NPY_BOOL;
                // case 8: return NPY_UINT8;
                case 16: return NPY_HALF;
                case 32: return NPY_FLOAT32;
                case 64: return NPY_FLOAT64;
                // case WTF: return NPY_LONGDOUBLE ????
            }
            return NPY_NOTYPE;
    
            /// WELL FUCK
            case Type::Handle: return NPY_VOID;
        }
        return NPY_NOTYPE;
    }
    
    Type from_typecode(typecode tc) {
        Type t;
        if (PyArray_ValidType(tc) != NPY_TRUE) { return t; }
        if (PyTypeNum_ISBOOL(tc)) { return Halide::Bool(); }
        if (!PyTypeNum_ISNUMBER(tc)) { t.code = Type::Handle; }
        if (PyTypeNum_ISFLOAT(tc)) { t.code = Type::Float; }
        if (PyTypeNum_ISINTEGER(tc) && PyTypeNum_ISUNSIGNED(tc)) { t.code = Type::UInt; }
        if (PyTypeNum_ISINTEGER(tc) && PyTypeNum_ISSIGNED(tc)) { t.code = Type::UInt; }
        
    }
    
    
    switch (PyArray_TYPE(array)) {
        case NPY_UINT8:
        case NPY_INT8:
            return 8;
        case NPY_UINT16:
        case NPY_INT16:
            return 16;
        case NPY_UINT32:
        case NPY_INT32:
            return 32;
        case NPY_UINT64:
        case NPY_INT64:
            return 64;
        default:
            throw ProgrammingError();
    }
    
    PyTypeNum_ISUNSIGNED(number);
    PyDataType_ISUNSIGNED(dtypestruct);
    PyArray_ISUNSIGNED(array);
    
    
    PyTypeNum_ISINTEGER();
    PyTypeNum_ISFLOAT();
    PyTypeNum_ISBOOL();
    PyTypeNum_ISNUMBER();
    
    PyArray_ValidType() == NPY_TRUE
    
}
