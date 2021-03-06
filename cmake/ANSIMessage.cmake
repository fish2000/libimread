# based on this: https://stackoverflow.com/a/19578320/298171

if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColorReset  "${Esc}[m")
    set(ColorBold   "${Esc}[1m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
    set(BoldRed     "${Esc}[1;31m")
    set(BoldGreen   "${Esc}[1;32m")
    set(BoldYellow  "${Esc}[1;33m")
    set(BoldBlue    "${Esc}[1;34m")
    set(BoldMagenta "${Esc}[1;35m")
    set(BoldCyan    "${Esc}[1;36m")
    set(BoldWhite   "${Esc}[1;37m")
else()
    set(ColorReset  "")
    set(ColorBold   "")
    set(Red         "")
    set(Green       "")
    set(Yellow      "")
    set(Blue        "")
    set(Magenta     "")
    set(Cyan        "")
    set(White       "")
    set(BoldRed     "")
    set(BoldGreen   "")
    set(BoldYellow  "")
    set(BoldBlue    "")
    set(BoldMagenta "")
    set(BoldCyan    "")
    set(BoldWhite   "")
endif()

function(ansi_message)
    list(GET ARGV 0 MessageType)
    if(MessageType STREQUAL FATAL_ERROR)
        list(REMOVE_AT ARGV 0)
        message(${MessageType} "${BoldRed}${ARGV}${ColorReset}")
    elseif(MessageType STREQUAL SEND_ERROR)
        list(REMOVE_AT ARGV 0)
        message(${MessageType} "${Red}${ARGV}${ColorReset}")
    elseif(MessageType STREQUAL WARNING)
        list(REMOVE_AT ARGV 0)
        message(${MessageType} "${BoldYellow}${ARGV}${ColorReset}")
    elseif(MessageType STREQUAL AUTHOR_WARNING)
        list(REMOVE_AT ARGV 0)
        message(${MessageType} "${BoldCyan}${ARGV}${ColorReset}")
    elseif(MessageType STREQUAL STATUS)
        list(REMOVE_AT ARGV 0)
        message(${MessageType} "${Cyan}${ARGV}${ColorReset}")
    else()
        message("${ARGV}")
    endif()
endfunction(ansi_message)