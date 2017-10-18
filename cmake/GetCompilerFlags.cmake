macro (GET_COMPILER_FLAGS TARGET VAR)
    set(COMPILER_FLAGS "")
    
    # Get flags from “add_definitions()”, re-escape quotes
    get_target_property(TARGET_DEFS ${TARGET} COMPILE_DEFINITIONS)
    get_directory_property(DIRECTORY_DEFS COMPILE_DEFINITIONS)
    foreach (DEF ${TARGET_DEFS} ${DIRECTORY_DEFS})
        if (DEF)
            string(REPLACE "\"" "\\\"" DEF "${DEF}")
            list(APPEND COMPILER_FLAGS "-D${DEF}")
        endif ()
    endforeach ()
    
    # Get flags from “include_directories()”
    get_target_property(TARGET_INCLUDEDIRS ${TARGET} INCLUDE_DIRECTORIES)
    foreach (DIR ${TARGET_INCLUDEDIRS})
        if (DIR)
            list(APPEND COMPILER_FLAGS "-I${DIR}")
        endif ()
    endforeach ()
    
    # Get build-type-specific flags
    string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE_SUFFIX)
    separate_arguments(GLOBAL_FLAGS UNIX_COMMAND
            "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE_SUFFIX}}")
    list(APPEND COMPILER_FLAGS ${GLOBAL_FLAGS})
    
    # Add -std= flag if appropriate -- WOW HOW HACKY IS THIS SHIT
    get_target_property(STANDARD ${TARGET} CXX_STANDARD)
    if ((NOT "${STANDARD}" STREQUAL NOTFOUND) AND (NOT "${STANDARD}" STREQUAL ""))
        list(APPEND COMPILER_FLAGS "-std=c++${STANDARD}")
    endif ()
    set(${VAR} "${COMPILER_FLAGS}")
endmacro ()