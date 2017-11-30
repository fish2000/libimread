
macro(check_dev_entry dev_entry var)
    message(STATUS "Checking for /dev/${dev_entry}")
    if(EXISTS "/dev/${dev_entry}")
        message(STATUS "Checking for /dev/${dev_entry} - found")
        set(${var} 1
            CACHE INTERNAL
            "Dev entry ${dev_entry} presence")
    else()
        message(STATUS "Checking for /dev/${dev_entry} - not found")
        set(${var} 0
            CACHE INTERNAL
            "Dev entry ${dev_entry} presence")
    endif()
endmacro()