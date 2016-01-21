
include(FindPythonInterp)

# function(JOIN VALUES GLUE OUTPUT)
#   string(REPLACE ";" "${GLUE}" _TMP_STR "${VALUES}")
#   set(${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
# endfunction()

macro(IOD_SYMBOLIZE header)
    set(symbolizer_args ${ARGN})
    list(LENGTH symbolizer_args num_dir_args)
    if(NOT ${num_dir_args} GREATER 0)
        message(FATAL_ERROR
            "IOD_SYMBOLIZE() called without directories")
    endif()
    list(INSERT symbolizer_args 0 symbolizer --verbose)
    list(APPEND symbolizer_args -o ${header})
    add_custom_command(
        OUTPUT ${header}
        COMMAND ${symbolizer_args})
    add_custom_target("iod_symbolize"
        DEPENDS ${header})
    set_source_files_properties(
        ${header}
        PROPERTIES GENERATED TRUE)
endmacro()