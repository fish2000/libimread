
@0xb6c5a11aca2271f3;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("halide");

struct TypedBuffer {
    buffer      @0 :Buffer;
    type        @1 :TypeCode        = uint;
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
    extent      @2 :DimList         = (x = 0, y = 0, p = 0);
    stride      @3 :DimList         = (x = 0, y = 0, p = 0);
    min         @4 :DimList         = (x = 0, y = 0, p = 0);
    elemSize    @5 :Int32           = 1;
    hostDirty   @6 :Bool            = false;
    devDirty    @7 :Bool            = false;
    dev         @1 :UInt64          = 0;
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

# derived from:
# typedef enum halide_type_code_t : uint8_t {
#     halide_type_int = 0,   //!< signed integers
#     halide_type_uint = 1,  //!< unsigned integers
#     halide_type_float = 2, //!< floating point numbers
#     halide_type_handle = 3 //!< opaque pointer type (void *)
# } halide_type_code_t;

enum TypeCode {
    int                 @0;
    uint                @1;
    float               @2;
    handle              @3;
}

struct Type {
    code        @0 :TypeCode        = uint;
    bits        @1 :UInt8           = 8;
    lanes       @2 :UInt16          = 1; # 1 for scalar types
}

# derived from:
# enum halide_error_code_t {
#     halide_error_code_success = 0,
#     halide_error_code_generic_error = -1,
#     halide_error_code_explicit_bounds_too_small = -2,
#     halide_error_code_bad_elem_size = -3,
#     halide_error_code_access_out_of_bounds = -4,
#     halide_error_code_buffer_allocation_too_large = -5,
#     halide_error_code_buffer_extents_too_large = -6,
#     halide_error_code_constraints_make_required_region_smaller = -7,
#     halide_error_code_constraint_violated = -8,
#     halide_error_code_param_too_small = -9,
#     halide_error_code_param_too_large = -10,
#     halide_error_code_out_of_memory = -11,
#     halide_error_code_buffer_argument_is_null = -12,
#     halide_error_code_debug_to_file_failed = -13,
#     halide_error_code_copy_to_host_failed = -14,
#     halide_error_code_copy_to_device_failed = -15,
#     halide_error_code_device_malloc_failed = -16,
#     halide_error_code_device_sync_failed = -17,
#     halide_error_code_device_free_failed = -18,
#     halide_error_code_no_device_interface = -19,
#     halide_error_code_matlab_init_failed = -20,
#     halide_error_code_matlab_bad_param_type = -21,
#     halide_error_code_internal_error = -22,
#     halide_error_code_device_run_failed = -23,
#     halide_error_code_unaligned_host_ptr = -24,
#     halide_error_code_bad_fold = -25,
#     halide_error_code_fold_factor_too_small = -26,
# };

enum ErrorCode {
    success                                 @0;
    genericError                            @1;
    explicitBoundsTooSmall                  @2;
    badElemSize                             @3;
    accessOutOfBounds                       @4;
    bufferAllocationTooLarge                @5;
    bufferExtentsTooLarge                   @6;
    constraintsMakeRequiredRegionSmaller    @7;
    constraintViolated                      @8;
    paramTooSmall                           @9;
    paramTooLarge                           @10;
    outOfMemory                             @11;
    bufferArgumentIsNull                    @12;
    debugToFileFailed                       @13;
    copyToHostFailed                        @14;
    copyToDeviceFailed                      @15;
    deviceMallocFailed                      @16;
    deviceSyncFailed                        @17;
    deviceFreeFailed                        @18;
    noDeviceInterface                       @19;
    matlabInitFailed                        @20;
    matlabBadParamType                      @21;
    internalError                           @22;
    deviceRunFailed                         @23;
    unalignedHostPtr                        @24;
    badFold                                 @25;
    foldFactorTooSmall                      @26;
}

# derived from:
# struct halide_scalar_value_t {
#     union {
#         bool b;
#         int8_t i8;
#         int16_t i16;
#         int32_t i32;
#         int64_t i64;
#         uint8_t u8;
#         uint16_t u16;
#         uint32_t u32;
#         uint64_t u64;
#         float f32;
#         double f64;
#         void *handle;
#     } u;
# };

struct Scalar {
    value :union {
        b       @0  :Bool;
        i8      @1  :Int8;
        i16     @2  :Int16;
        i32     @3  :Int32;
        i64     @4  :Int64;
        u8      @5  :UInt8;
        u16     @6  :UInt16;
        u32     @7  :UInt32;
        u64     @8  :UInt64;
        f32     @9  :Float32;
        f64     @10 :Float64;
        handle  @11 :Data;
        nil     @12 :Void;
    }
}

# derived from:
# enum halide_argument_kind_t {
#     halide_argument_kind_input_scalar = 0,
#     halide_argument_kind_input_buffer = 1,
#     halide_argument_kind_output_buffer = 2
# };

enum ArgumentKind {
    inputScalar     @0;
    inputBuffer     @1;
    outputBuffer    @2;
}

# derived from:
# struct halide_filter_argument_t {
#     const char *name;       // name of the argument; will never be null or empty.
#     int32_t kind;           // actually halide_argument_kind_t
#     int32_t dimensions;     // always zero for scalar arguments
#     halide_type_t type;
#     // These pointers should always be null for buffer arguments,
#     // and *may* be null for scalar arguments. (A null value means
#     // there is no def/min/max specified for this argument.)
#     const struct halide_scalar_value_t *def;
#     const struct halide_scalar_value_t *min;
#     const struct halide_scalar_value_t *max;
# };

struct FilterArgument {
    name        @0 :Text;           # name of argument
    kind        @1 :ArgumentKind    = inputScalar;
    dimensions  @2 :Int32           = 0; # zero for scalars
    type        @3 :Type            = (code = uint, bits = 8);
    def         @4 :Scalar          = (value = (nil = void));
    min         @5 :Scalar          = (value = (nil = void));
    max         @6 :Scalar          = (value = (nil = void));
}

struct FilterMetadata {
    version         @0 :Int32                   = 0; # always 0
    numArguments    @1 :Int32                   = 1; # always >=1
    arguments       @2 :List(FilterArgument);
    target          @3 :Text;                   # name of Halide target (from Target::to_string)
    name            @4 :Text;                   # name of filter
}