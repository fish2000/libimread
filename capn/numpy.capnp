
@0xa4ff3451b6f64e62;

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
#
#                     NPY_NTYPES,
#                     NPY_NOTYPE,
#                     NPY_CHAR,      /* special flag */
#                     NPY_USERDEF=256,  /* leave room for characters */
#
#                     /* The number of types not including the new 1.6 types */
#                     NPY_NTYPES_ABI_COMPATIBLE=21
# };

enum NPY_TYPES {
    NPY_BOOL        @0;
    NPY_BYTE        @1;
    NPY_UBYTE       @2;
    NPY_SHORT       @3;
    NPY_USHORT      @4;
    NPY_LONG        @5;
    NPY_ULONG       @6;
    NPY_LONGLONG    @7;
    NPY_ULONGLONG   @8;
    NPY_FLOAT       @9;
    NPY_DOUBLE      @10;
    NPY_LONGDOUBLE  @11;
    NPY_CFLOAT      @12;
    NPY_CDOUBLE     @13;
    NPY_CLONGDOUBLE @14;
    npy_dummy0      @15;
    npy_dummy1      @16;
    NPY_OBJECT      @17;
    NPY_STRING      @18;
    NPY_UNICODE     @19;
    NPY_VOID        @20;
    NPY_DATETIME    @21;
    NPY_TIMEDELTA   @22;
    NPY_HALF        @23;
    NPY_NTYPES      @24;
    NPY_NOTYPE      @25;
    NPY_CHAR        @26;
    NPY_USERDEF     @256; # WAT
    
    #NPY_NTYPES_ABI_COMPATIBLE @21; # double WAT
}
