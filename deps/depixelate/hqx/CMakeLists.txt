
find_package(PkgConfig REQUIRED)
pkg_search_module(IL REQUIRED IL)
pkg_search_module(ILU REQUIRED ILU)
pkg_search_module(ILUT REQUIRED ILUT)

include_directories(${IL_INCLUDE_DIRS}
                    ${ILU_INCLUDE_DIRS}
                    ${ILUT_INCLUDE_DIRS})

add_definitions(
    -Wall -Werror
    -Wno-unused-function
    -Wno-absolute-value
    -std=gnu99 -O3
    -mtune=native -fstrict-aliasing
    -x c)

set(hqx_srcs
    "hq2x.c"
    "hq3x.c"
    "hq4x.c"
    "hqx_init.c"
    "hqx.c")

set(hqx_hdrs
    "hqx_common.h"
    "hqx.h")

add_library(
    depixelate_hqx
    OBJECT ${hqx_srcs} ${hqx_hdrs})
set_target_properties(
    depixelate_hqx
    PROPERTIES LIBRARY_OUTPUT_NAME "depixelate_hqx")
set_property(
    TARGET depixelate_hqx
    PROPERTY POSITION_INDEPENDENT_CODE 1)

# Add shared and static library targets, based on the “OBJECT” library,
# respectively called “depixelate_hqx_shared” and “depixelate_hqx_static”:
add_library(
    depixelate_hqx_shared
    SHARED $<TARGET_OBJECTS:depixelate_hqx>)
add_library(
    depixelate_hqx_static
    STATIC $<TARGET_OBJECTS:depixelate_hqx>)
set_target_properties(
    depixelate_hqx_shared
    PROPERTIES LIBRARY_OUTPUT_NAME "depixelate_hqx")
set_target_properties(
    depixelate_hqx_static
    PROPERTIES ARCHIVE_OUTPUT_NAME "depixelate_hqx")
target_link_libraries(depixelate_hqx_shared ${IL_LIBRARIES}
                                            ${ILU_LIBRARIES}
                                            ${ILUT_LIBRARIES})
target_link_libraries(depixelate_hqx_static ${IL_LIBRARIES}
                                            ${ILU_LIBRARIES}
                                            ${ILUT_LIBRARIES})

# Install the shared and static library target outputs:
# install(TARGETS depixelate_hqx_shared
#         DESTINATION lib)
# install(TARGETS depixelate_hqx_static
#         DESTINATION lib)