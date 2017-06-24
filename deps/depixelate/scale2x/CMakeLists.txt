
add_definitions(
    -Wall -Werror
    -Wno-unused-function
    -std=gnu11 -O3
    -mtune=native -fstrict-aliasing
    -x c)

set(scale2x_srcs
    "pixel.c"
    "scale2x.c"
    "scale3x.c"
    "scalebit.c")

set(scale2x_hdrs
    "pixel.h"
    "portable.h"
    "scale2x.h"
    "scale3x.h"
    "scalebit.h")

add_library(
    depixelate_scale2x
    OBJECT ${scale2x_srcs} ${scale2x_hdrs})
set_target_properties(
    depixelate_scale2x
    PROPERTIES LIBRARY_OUTPUT_NAME "depixelate_scale2x")
set_property(
    TARGET depixelate_scale2x
    PROPERTY POSITION_INDEPENDENT_CODE 1)

# Add shared and static library targets, based on the “OBJECT” library,
# respectively called “depixelate_scale2x_shared” and “depixelate_scale2x_static”:
# add_library(
#     depixelate_scale2x_shared
#     SHARED $<TARGET_OBJECTS:depixelate_scale2x>)
# add_library(
#     depixelate_scale2x_static
#     STATIC $<TARGET_OBJECTS:depixelate_scale2x>)
# set_target_properties(
#     depixelate_scale2x_shared
#     PROPERTIES LIBRARY_OUTPUT_NAME "depixelate_scale2x")
# set_target_properties(
#     depixelate_scale2x_static
#     PROPERTIES ARCHIVE_OUTPUT_NAME "depixelate_scale2x")
# target_link_libraries(depixelate_scale2x_shared)
# target_link_libraries(depixelate_scale2x_static)

# Install the shared and static library target outputs:
# install(TARGETS depixelate_scale2x_shared
#         DESTINATION lib)
# install(TARGETS depixelate_scale2x_static
#         DESTINATION lib)