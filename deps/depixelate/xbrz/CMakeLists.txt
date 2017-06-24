
add_definitions(
    -Wall -Werror
    -Wno-unused-function
    -std=c++14 -stdlib=libc++ -O3
    -mtune=native -fstrict-aliasing
    -x c++)

set(xbrz_srcs
    "xbrz.cpp")

set(xbrz_hdrs
    "debugbreak.h"
    "xbrz_config.h"
    "xbrz.h")

add_library(
    depixelate_xbrz
    OBJECT ${xbrz_srcs} ${xbrz_hdrs})
set_target_properties(
    depixelate_xbrz
    PROPERTIES LIBRARY_OUTPUT_NAME "depixelate_xbrz")
set_property(
    TARGET depixelate_xbrz
    PROPERTY POSITION_INDEPENDENT_CODE 1)

# Add shared and static library targets, based on the “OBJECT” library,
# respectively called “depixelate_xbrz_shared” and “depixelate_xbrz_static”:
# add_library(
#     depixelate_xbrz_shared
#     SHARED $<TARGET_OBJECTS:depixelate_xbrz>)
# add_library(
#     depixelate_xbrz_static
#     STATIC $<TARGET_OBJECTS:depixelate_xbrz>)
# set_target_properties(
#     depixelate_xbrz_shared
#     PROPERTIES LIBRARY_OUTPUT_NAME "depixelate_xbrz")
# set_target_properties(
#     depixelate_xbrz_static
#     PROPERTIES ARCHIVE_OUTPUT_NAME "depixelate_xbrz")
# target_link_libraries(depixelate_xbrz_shared)
# target_link_libraries(depixelate_xbrz_static)

# Install the shared and static library target outputs:
# install(TARGETS depixelate_xbrz_shared
#         DESTINATION lib)
# install(TARGETS depixelate_xbrz_static
#         DESTINATION lib)