# Install script for directory: /Users/fish/Dropbox/libimread/apps/libimread-config

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/fish/Dropbox/libimread/xcode/fmwk/apps/libimread-config/Debug/imread-config")
    if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/usr/local/lib"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/docopt/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/crossguid/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/MABlockClosure/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/imagecompression/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/SSZipArchive/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      endif()
    endif()
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/fish/Dropbox/libimread/xcode/fmwk/apps/libimread-config/Release/imread-config")
    if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/usr/local/lib"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/docopt/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/crossguid/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/MABlockClosure/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/imagecompression/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/SSZipArchive/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      endif()
    endif()
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/fish/Dropbox/libimread/xcode/fmwk/apps/libimread-config/MinSizeRel/imread-config")
    if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/usr/local/lib"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/docopt/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/crossguid/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/MABlockClosure/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/imagecompression/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/SSZipArchive/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      endif()
    endif()
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/fish/Dropbox/libimread/xcode/fmwk/apps/libimread-config/RelWithDebInfo/imread-config")
    if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/usr/local/lib"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/docopt/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/crossguid/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/MABlockClosure/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/imagecompression/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/SSZipArchive/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/imread-config")
      endif()
    endif()
  endif()
endif()

