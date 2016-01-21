# Install script for directory: /Users/fish/Dropbox/libimread

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
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/fish/Dropbox/libimread/xcode/fmwk/Debug/libimread.dylib")
    if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/usr/local/lib"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/crossguid/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/MABlockClosure/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/imagecompression/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/SSZipArchive/Debug"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      endif()
    endif()
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/fish/Dropbox/libimread/xcode/fmwk/Release/libimread.dylib")
    if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/usr/local/lib"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/crossguid/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/MABlockClosure/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/imagecompression/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/SSZipArchive/Release"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      endif()
    endif()
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/fish/Dropbox/libimread/xcode/fmwk/MinSizeRel/libimread.dylib")
    if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/usr/local/lib"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/crossguid/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/MABlockClosure/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/imagecompression/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/SSZipArchive/MinSizeRel"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      endif()
    endif()
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/fish/Dropbox/libimread/xcode/fmwk/RelWithDebInfo/libimread.dylib")
    if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/usr/local/lib"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/crossguid/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/MABlockClosure/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/imagecompression/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/fish/Dropbox/libimread/xcode/fmwk/deps/SSZipArchive/RelWithDebInfo"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimread.dylib")
      endif()
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libimread" TYPE DIRECTORY FILES "/Users/fish/Dropbox/libimread/xcode/fmwk/libimread/" FILES_MATCHING REGEX "/libimread\\.hpp$")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libimread/private" TYPE DIRECTORY FILES "/Users/fish/Dropbox/libimread/include/libimread/private/" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libimread" TYPE DIRECTORY FILES "/Users/fish/Dropbox/libimread/include/libimread/" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libimread" TYPE DIRECTORY FILES "/Users/fish/Dropbox/libimread/include/libimread/" FILES_MATCHING REGEX "/[^/]*\\.hh$")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libimread" TYPE DIRECTORY FILES "/Users/fish/Dropbox/libimread/include/libimread/" FILES_MATCHING REGEX "/[^/]*\\.hpp$")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libimread" TYPE DIRECTORY FILES "/Users/fish/Dropbox/libimread/cmake/" FILES_MATCHING REGEX "/[^/]*libimreadconfig\\.cmake$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/fish/Dropbox/libimread/xcode/fmwk/deps/cmake_install.cmake")
  include("/Users/fish/Dropbox/libimread/xcode/fmwk/apps/cmake_install.cmake")
  include("/Users/fish/Dropbox/libimread/xcode/fmwk/tests/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/fish/Dropbox/libimread/xcode/fmwk/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
