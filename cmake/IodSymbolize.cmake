
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