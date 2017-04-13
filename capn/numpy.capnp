
@0xa4ff3451b6f64e62;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("im::numpy");

# enum NPY_TYPES {    NPY_BOOL=0,
#                     NPY_BYTE, NPY_UBYTE,
#                     NPY_SHORT, NPY_USHORT,
#                     NPY_INT, NPY_UINT,
#                     NPY_LONG, NPY_ULONG,
#                     NPY_LONGLONG, NPY_ULONGLONG,
#                     NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
#                     NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
#                     NPY_OBJECT=17,
#                     NPY_STRING, NPY_UNICODE,
#                     NPY_VOID,
#                     /*
#                      * New 1.6 types appended, may be integrated
#                      * into the above in 2.0.
#                      */
#                     NPY_DATETIME, NPY_TIMEDELTA, NPY_HALF,
#                     NPY_NTYPES,
#                     NPY_NOTYPE,
#                     NPY_CHAR,      /* special flag */
#                     NPY_USERDEF=256,  /* leave room for characters */
#                     /* The number of types not including the new 1.6 types */
#                     NPY_NTYPES_ABI_COMPATIBLE=21
# };

enum NumpyTypes {

    npyBOOL                 @0;
    npyBYTE                 @1;
    npyUBYTE                @2;
    npySHORT                @3;
    npyUSHORT               @4;
    npyLONG                 @5;
    npyULONG                @6;
    npyLONGLONG             @7;
    npyULONGLONG            @8;
    npyFLOAT                @9;
    npyDOUBLE               @10;
    npyLONGDOUBLE           @11;
    npyCFLOAT               @12;
    npyCDOUBLE              @13;
    npyCLONGDOUBLE          @14;

    npyDummy0               @15;
    npyDummy1               @16;

    npyOBJECT               @17;
    npySTRING               @18;
    npyUNICODE              @19;
    npyVOID                 @20;
    npyDATETIME             @21;
    npyTIMEDELTA            @22;
    npyHALF                 @23;
    npyNTYPES               @24;
    npyNOTYPE               @25;
    npyCHAR                 @26;
    npyUSERDEF              @27;

    # npyUSERDEF     @256;          # WAT
    #NPY_NTYPES_ABI_COMPATIBLE @21; # double WAT

    npyNtypesABICompatible  @28;
}
