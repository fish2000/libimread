
add_definitions(
    -fobjc-abi-version=3
    -fobjc-arc -fobjc-call-cxx-cdtors
    -fno-objc-arc-exceptions
    -std=c++1z -stdlib=libc++
    -x objective-c++
    -Wno-nullability-completeness
    -DWITH_SCHEMA
    -O3 -funroll-loops -mtune=native
    -fstrict-aliasing)
