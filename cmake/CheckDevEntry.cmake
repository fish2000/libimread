
macro(check_dev_entry dev_entry var)
    # set(options ${ARGN})
    # list(LENGTH options num_options)
    # string(TOUPPER ${dev_entry} dev_entry_upcase)
    message(STATUS "Checking for /dev/${dev_entry}")
    if(EXISTS "/dev/${dev_entry}")
        message(STATUS "Checking for /dev/${dev_entry} - found")
        set(${var} TRUE
            CACHE INTERNAL
            "Dev entry ${dev_entry} presence")
    else()
        message(STATUS "Checking for /dev/${dev_entry} - not found")
        set(${var} FALSE
            CACHE INTERNAL
            "Dev entry ${dev_entry} presence")
    endif()
endmacro()