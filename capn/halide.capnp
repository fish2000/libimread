
@0xb6c5a11aca2271f3;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("halide");

struct TypedBuffer {
    buffer      @0 :Buffer;
    type        @1 :TypeCode        = halideTypeUInt;
    ndims       @2 :UInt8           = 3;
}

# Direct translation of halide buffer_t struct --
# when packed with the Cap'n, this *should* have
# the same layout as an actual in-memory buffer_t:
struct BufferT {
    dev         @0 :UInt64          = 0;
    host        @1 :Data            = 0x"00";
    extent      @2 :List(Int32)     = [ 0, 0, 0, 0 ];
    stride      @3 :List(Int32)     = [ 0, 0, 0, 0 ];
    min         @4 :List(Int32)     = [ 0, 0, 0, 0 ];
    elemSize    @5 :Int32           = 1;
    hostDirty   @6 :Bool            = false;
    devDirty    @7 :Bool            = false;
}

# Rearranged/simplified/syntax-sugared the buffer_t data fields
struct Buffer {
    host        @0 :Data            = 0x"00";
    dev         @1 :UInt64          = 0;
    extent      @2 :DimList         = (x = 0, y = 0, p = 0);
    stride      @3 :DimList         = (x = 0, y = 0, p = 0);
    min         @4 :DimList         = (x = 0, y = 0, p = 0);
    elemSize    @5 :Int32           = 1;
    hostDirty   @6 :Bool            = false;
    devDirty    @7 :Bool            = false;
}

# Rough equivalent of:
#
#   struct DimList {
#       int32_t x = 0,
#               y = 0,
#               p = 0,
#               v = 0;
#   };
#
struct DimList {
    x           @0 :Int32           = 0; # width
    y           @1 :Int32           = 0; # height
    p           @2 :Int32           = 0; # planes
    v           @3 :Int32           = 0; # frames (not currently used)
}

# typedef enum halide_type_code_t : uint8_t {
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

