
set(LIB_NAME vdb-client)

set(CPP_FILES
    vdb.cpp)

set(PUBLIC_HEADERS
    vdb.h
    vdb_impl.h)

set(PRIVATE_HEADERS)
# set(PUBLIC_HEADER_DST_LOCATION "${OF_INST_INC_PATH}/${LIB_NAME}")

add_library(${LIB_NAME} SHARED ${CPP_FILES} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})
install(TARGETS ${LIB_NAME} DESTINATION lib)
install(FILES ${PUBLIC_HEADERS} DESTINATION include/vdb)
