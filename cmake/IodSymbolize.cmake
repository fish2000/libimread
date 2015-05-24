

# add_custom_command(
#     OUTPUT ${TEST_INCLUDE_DIR}/test_data.hpp
#     COMMAND ${TEST_SCRIPT_DIR}/scan-test-data.py > ${TEST_INCLUDE_DIR}/test_data.hpp)
# add_custom_target("test_data_header"
#     DEPENDS ${TEST_INCLUDE_DIR}/test_data.hpp)
# set_source_files_properties(
#     ${TEST_INCLUDE_DIR}/test_data.hpp
#     PROPERTIES GENERATED TRUE)
# add_executable("test_${PROJECT_NAME}" ${TEST_SOURCES})
# add_dependencies("test_${PROJECT_NAME}" "test_data_header")


# macro(IOD_SYMBOLIZE_DIR directory)
# endmacro()
#
# macro(ADD_IOD_SYMBOLS)
# endmacro()

macro(IOD_SYMBOLIZE dir0 dir1 header)
add_custom_command(
    OUTPUT ${header}
    COMMAND symbolizer --verbose ${dir0} ${dir1} -o ${header})
add_custom_target("iod_symbolize"
    DEPENDS ${header})
set_source_files_properties(
    ${header}
    PROPERTIES GENERATED TRUE)
endmacro()