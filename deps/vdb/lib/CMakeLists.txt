cmake_minimum_required(VERSION 3.3)
project(vdb-client LANGUAGES C CXX)

set(vdb_client_VERSION_MAJOR "0")
set(vdb_client_VERSION_MINOR "1")
set(vdb_client_VERSION_PATCH "0")

set(LIB_NAME vdb-client)

set(CPP_FILES
    vdb.cpp)

set(PUBLIC_HEADERS
    vdb.h
    vdb_impl.h)

set(PRIVATE_HEADERS)
# set(PUBLIC_HEADER_DST_LOCATION "${OF_INST_INC_PATH}/${LIB_NAME}")

add_library(${LIB_NAME}-static STATIC
            ${CPP_FILES} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})
add_library(${LIB_NAME}-shared SHARED
            ${CPP_FILES} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})
set_target_properties(${LIB_NAME}-static PROPERTIES
                      ARCHIVE_OUTPUT_NAME ${LIB_NAME})
set_target_properties(${LIB_NAME}-shared PROPERTIES
                      LIBRARY_OUTPUT_NAME ${LIB_NAME})

install(TARGETS ${LIB_NAME}-static DESTINATION lib)
install(TARGETS ${LIB_NAME}-shared DESTINATION lib)
install(FILES ${PUBLIC_HEADERS} DESTINATION include/vdb)
