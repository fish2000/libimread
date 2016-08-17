
@0xb6c5a11aca2271f3;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("halide");

struct TypedBuffer {
    buffer      @0 :Buffer;
    type        @1 :TypeCode        = halideTypeUInt;
    ndims       @2 :UInt8           = 3;
}

struct Buffer {
    host        @0 :Data;
    dev         @1 :UInt64          = 0;
    extent      @2 :List(Int32)     = [ 0, 0, 0, 0 ];
    stride      @3 :List(Int32)     = [ 0, 0, 0, 0 ];
    min         @4 :List(Int32)     = [ 0, 0, 0, 0 ];
    elemSize    @5 :Int32           = 1;
    hostDirty   @6 :Bool            = false;
    devDirty    @7 :Bool            = false;
}

# typedef enum halide_type_code_t
# #if __cplusplus >= 201103L
# : uint8_t
# #endif
# {
#     halide_type_int = 0,   //!< signed integers
#     halide_type_uint = 1,  //!< unsigned integers
#     halide_type_float = 2, //!< floating point numbers
#     halide_type_handle = 3 //!< opaque pointer type (void *)
# } halide_type_code_t;

enum TypeCode {
    halideTypeInt       @0;
    halideTypeUInt      @1;
    halideTypeFloat     @2;
    halideTypeHandle    @3;
}

